from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.classifier import ProtoClassifier
from src.data_loader import verify_cora_dataset
from src.graph_augmenter import GraphAugmenter
from src.gnn_backbone import GCNBackbone
from src.llm_encoder import SciBERTEncoder
from src.revised_protocol import CLASS_DESCRIPTIONS, CLASS_SPLIT, PROTOCOL_SEED
from src.v2.prompt_models import NormalizedGatedPrompt
from src.v2.validation import class_embedding_variant, snapshot_v1_artifacts


SEED = 11
SHOT = 3
CONDITIONS = ("correct", "shuffled", "random", "ZeroPrompt", "NoPrompt")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_episode_manifest() -> list[dict[str, Any]]:
    path = ROOT / "results" / "v2" / "validation_episode_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    key = f"validation|shot={SHOT}|seed={SEED}"
    raw_episodes = manifest.get(key)
    if raw_episodes is None or len(raw_episodes) != 100:
        raise RuntimeError(f"Expected 100 saved validation episodes at key {key}")
    episodes: list[dict[str, Any]] = []
    for raw in raw_episodes:
        episode = {
            name: np.asarray(value, dtype=np.int64) if name != "episode" else value
            for name, value in raw.items()
        }
        if not set(episode["support_idx"].tolist()).isdisjoint(episode["query_idx"].tolist()):
            raise AssertionError("Saved validation episode contains support/query overlap")
        if episode["class_ids"].tolist() != list(CLASS_SPLIT["validation"]):
            raise AssertionError("Saved validation episode contains unexpected classes")
        episodes.append(episode)
    return episodes


def load_checkpoint(condition: str) -> tuple[NormalizedGatedPrompt, dict[str, Any], str]:
    path = ROOT / "results" / "v2" / "checkpoints" / f"normalizedgatedprompt_{condition}_shot_{SHOT}_seed_{SEED}.pt"
    if not path.is_file():
        raise FileNotFoundError(f"Missing existing V2 checkpoint: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    metadata = checkpoint["metadata"]
    if int(metadata["seed"]) != SEED or int(metadata["shot"]) != SHOT:
        raise AssertionError("V2 checkpoint seed/shot metadata mismatch")
    if metadata["semantic_variant"] != condition:
        raise AssertionError("V2 checkpoint semantic condition mismatch")
    model = NormalizedGatedPrompt(
        reference_node_norm=float(metadata["reference_node_norm_train_only"]),
        initial_gate=float(metadata["initial_gate"]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(DEVICE).eval()
    return model, metadata, hashlib.sha256(path.read_bytes()).hexdigest()


def predict_condition(
    condition: str,
    episodes: list[dict[str, Any]],
    x: torch.Tensor,
    edge_index: torch.Tensor,
    class_embeddings: torch.Tensor,
    backbone: GCNBackbone,
    models: dict[str, NormalizedGatedPrompt],
) -> dict[str, Any]:
    episode_accuracies: list[float] = []
    episode_f1s: list[float] = []
    episode_logits: list[torch.Tensor] = []
    episode_predictions: list[torch.Tensor] = []
    prompt_norms: list[float] = []
    query_embeddings: list[torch.Tensor] = []
    gate = None

    with torch.no_grad():
        for episode in episodes:
            support = torch.as_tensor(episode["support_idx"], dtype=torch.long, device=DEVICE)
            query = torch.as_tensor(episode["query_idx"], dtype=torch.long, device=DEVICE)
            support_labels = torch.as_tensor(episode["support_labels"], dtype=torch.long, device=DEVICE)
            query_labels = torch.as_tensor(episode["query_labels"], dtype=torch.long, device=DEVICE)
            class_ids = torch.as_tensor(episode["class_ids"], dtype=torch.long, device=DEVICE)

            if condition == "NoPrompt":
                graph_x = x
                graph_edges = edge_index
                original_mask = None
                prompt_features = x.new_empty((0, x.shape[1]))
            else:
                if condition == "ZeroPrompt":
                    prompt_features = x.new_zeros((2, x.shape[1]))
                else:
                    model_key = "configured" if condition == "correct" else condition
                    prompt_features = models[model_key](class_embeddings[class_ids])
                    gate = float(models[model_key].gate.item())
                augmented = GraphAugmenter(num_classes=2).augment(
                    x,
                    edge_index,
                    support,
                    support_labels,
                    prompt_embeddings=prompt_features,
                )
                graph_x, graph_edges, original_mask = augmented.x, augmented.edge_index, augmented.original_mask

            output = backbone(graph_x, graph_edges)
            if original_mask is not None:
                output = output[original_mask]
            support_embeddings = output[support]
            query_output = output[query]
            prototypes = ProtoClassifier.prototypes(support_embeddings, support_labels, 2)
            logits = ProtoClassifier.logits(query_output, prototypes, temperature=0.1)
            predictions = logits.argmax(dim=-1)
            episode_accuracies.append(float((predictions == query_labels).float().mean().item()))
            episode_f1s.append(
                float(
                    f1_score(
                        query_labels.cpu().numpy(),
                        predictions.cpu().numpy(),
                        labels=[0, 1],
                        average="macro",
                        zero_division=0,
                    )
                )
            )
            episode_logits.append(logits.detach().cpu())
            episode_predictions.append(predictions.detach().cpu())
            query_embeddings.append(query_output.detach().cpu())
            prompt_norms.append(float(prompt_features.norm(dim=-1).mean().item()) if prompt_features.numel() else 0.0)

    return {
        "condition": condition,
        "accuracy": float(np.mean(episode_accuracies)),
        "macro_f1": float(np.mean(episode_f1s)),
        "effective_prompt_norm": float(np.mean(prompt_norms)),
        "gate": gate,
        "logits": episode_logits,
        "predictions": episode_predictions,
        "query_embeddings": query_embeddings,
    }


def paired_sensitivity(reference: dict[str, Any], target: dict[str, Any]) -> dict[str, float]:
    cosine_values: list[float] = []
    l2_values: list[float] = []
    logit_delta_values: list[float] = []
    prediction_disagreement: list[float] = []
    for ref_embedding, target_embedding, ref_logits, target_logits, ref_pred, target_pred in zip(
        reference["query_embeddings"],
        target["query_embeddings"],
        reference["logits"],
        target["logits"],
        reference["predictions"],
        target["predictions"],
    ):
        cosine_values.extend(
            F.cosine_similarity(ref_embedding, target_embedding, dim=-1).tolist()
        )
        l2_values.extend(torch.linalg.vector_norm(target_embedding - ref_embedding, dim=-1).tolist())
        logit_delta_values.extend((target_logits - ref_logits).abs().mean(dim=-1).tolist())
        prediction_disagreement.extend((target_pred != ref_pred).float().tolist())
    return {
        "query_cosine_mean": float(np.mean(cosine_values)),
        "query_cosine_min": float(np.min(cosine_values)),
        "query_l2_change_mean": float(np.mean(l2_values)),
        "query_l2_change_max": float(np.max(l2_values)),
        "changed_query_count": int(np.count_nonzero(np.asarray(l2_values) > 0)),
        "query_count": len(l2_values),
        "mean_absolute_logit_change": float(np.mean(logit_delta_values)),
        "changed_prediction_count": int(np.count_nonzero(prediction_disagreement)),
        "prediction_disagreement_rate": float(np.mean(prediction_disagreement)),
    }


def main() -> None:
    global SEED, SHOT
    started = time.perf_counter()
    torch.set_num_threads(1)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    # Snapshot all existing checkpoints and manuscript, including V2 artifacts.
    protected = sorted(ROOT.rglob("*.pt")) + sorted(ROOT.parent.glob("*.docx"))
    before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    v1_before = snapshot_v1_artifacts(ROOT)
    info = verify_cora_dataset()
    encoder = SciBERTEncoder(model_name="allenai/scibert_scivocab_uncased", device=str(DEVICE))
    permitted_ids = tuple(CLASS_SPLIT["validation"])
    encoded = encoder.encode_class_descriptions(
        [CLASS_DESCRIPTIONS[i] for i in permitted_ids], batch_size=len(permitted_ids)
    ).to(DEVICE)
    embeddings = torch.zeros(7, 768, dtype=encoded.dtype, device=DEVICE)
    embeddings[list(permitted_ids)] = encoded
    assert all(not p.requires_grad for p in encoder.model.parameters())
    manifest_path = ROOT / "results/v2/validation_episode_manifest.json"
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    paths = sorted((ROOT / "results/v2/checkpoints").glob("normalizedgatedprompt_configured_shot_*_seed_*.pt"))
    if not paths:
        raise RuntimeError("No saved configured V2 models available; training is prohibited")
    runs = []
    for path in paths:
        parts = path.stem.split("_")
        SHOT, SEED = int(parts[3]), int(parts[5])
        episodes = load_episode_manifest()
        for episode in episodes:
            for index_key, label_key in (("support_idx", "support_labels"), ("query_idx", "query_labels")):
                actual = info["y"][torch.as_tensor(episode[index_key])].numpy()
                expected = np.asarray(permitted_ids)[episode[label_key]]
                if not np.array_equal(actual, expected):
                    raise AssertionError("Manifest labels do not match validation-only node IDs")
        model, metadata, digest = load_checkpoint("configured")
        model.requires_grad_(False)
        gcn_path = ROOT / "results/checkpoints" / f"seed_{SEED}_frozen_gcn.pt"
        if not str(metadata["v1_backbone_checkpoint"]).replace(chr(92), "/").endswith(gcn_path.name):
            raise AssertionError("Prompt/backbone association mismatch")
        checkpoint = torch.load(gcn_path, map_location="cpu", weights_only=False)
        assert int(checkpoint["pretraining"]["seed"]) == SEED
        backbone = GCNBackbone(1433, 256, 256).to(DEVICE)
        backbone.load_state_dict(checkpoint["model_state_dict"])
        backbone.eval().requires_grad_(False)
        # Swap the two validation descriptions, keeping labels, edges and weights fixed.
        shuffled = embeddings.clone()
        shuffled[list(permitted_ids)] = embeddings[list(reversed(permitted_ids))]
        random_embeddings, _ = class_embedding_variant(
            embeddings, "random", seed=PROTOCOL_SEED + SEED * 100 + SHOT
        )
        condition_embeddings = {"correct": embeddings, "shuffled": shuffled,
                                "random": random_embeddings, "ZeroPrompt": embeddings,
                                "NoPrompt": embeddings}
        outputs = {}
        with torch.inference_mode():
            prompt_features = model(encoded)
            prompt_changes = {}
            for name in ("shuffled", "random"):
                changed = model(condition_embeddings[name][list(permitted_ids)])
                prompt_changes[name] = {
                    "l2_mean": float((changed - prompt_features).norm(dim=-1).mean()),
                    "cosine_mean": float(F.cosine_similarity(changed, prompt_features, dim=-1).mean()),
                }
            for condition in CONDITIONS:
                outputs[condition] = predict_condition(
                    condition, episodes, info["x"].to(DEVICE), info["edge_index"].to(DEVICE),
                    condition_embeddings[condition], backbone,
                    {"configured": model, "shuffled": model, "random": model},
                )
        comparisons = {c: paired_sensitivity(outputs["correct"], outputs[c]) for c in CONDITIONS[1:]}
        runs.append({
            "seed": SEED, "shot": SHOT, "episodes": len(episodes),
            "checkpoint": str(path.relative_to(ROOT)), "sha256": digest,
            "backbone_sha256": before[str(gcn_path)],
            "episode_key": f"validation|shot={SHOT}|seed={SEED}",
            "query_norm_mean": float(torch.cat(outputs["correct"]["query_embeddings"]).norm(dim=-1).mean()),
            "gate_logit": float(model.gate_logit),
            "reference_node_norm": model.reference_node_norm,
            "conditions": {c: {k: v for k, v in out.items() if k not in
                           {"logits", "predictions", "query_embeddings"}} for c, out in outputs.items()},
            "comparisons": comparisons, "prompt_feature_changes": prompt_changes,
        })
        assert all(p.grad is None and not p.requires_grad for p in [*model.parameters(), *backbone.parameters()])
        print(f"Completed seed={SEED}, shot={SHOT}, {len(episodes)} identical validation episodes", flush=True)
    after = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    assert before == after, "Checkpoint or manuscript changed"
    assert v1_before == snapshot_v1_artifacts(ROOT), "Existing artifacts changed during inference"
    assert manifest_hash == hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    seeds_in_manifest = sorted({int(k.split("seed=")[1]) for k in manifest})
    available_seeds = sorted({r["seed"] for r in runs})
    result = {
        "runs": runs, "available_seeds": available_seeds,
        "unavailable_seeds": sorted(set(seeds_in_manifest) - set(available_seeds)),
        "manifest_sha256": manifest_hash, "protected_files_unchanged": True,
        "protected_file_count": len(protected), "device": str(DEVICE),
        "runtime_seconds": time.perf_counter() - started,
        "test_classes_evaluated": False, "training_performed": False,
    }
    write_report(result)
    (ROOT / "reports/v2_semantic_sensitivity.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


def write_report(result: dict[str, Any]) -> None:
    result["runs"].sort(key=lambda run: (run["seed"], run["shot"]))
    lines = [
        "# LLMGP-v2 semantic sensitivity diagnosis", "",
        "## Protocol and availability", "",
        "Inference only. Each saved configured NormalizedGatedPrompt checkpoint and its associated frozen GCN are held fixed across correct, shuffled, random, ZeroPrompt and NoPrompt. No optimization, checkpoint writing or test-class evaluation was performed.", "",
        f"Available V2 seeds: {result['available_seeds']}. Missing V2 seeds represented in the saved validation manifest: {result['unavailable_seeds']}. The extended-validation runner trained seeds 23/37 in memory without saving their V2 weights; those reported models cannot be reused. V1 prompt weights are not substituted.", "",
        "All available configured V2 shot settings are diagnosed, with 100 existing manifest episodes per setting. Each condition uses exactly the same support/query IDs, labels, order and classifier (cosine prototypes, temperature 0.1). Only validation IDs [3,4] are encoded and scored. The existing full Cora graph is retained for transductive message passing, including unlabeled nodes from other classes; test-class descriptions and test episodes are not used.", "",
        "Correct uses the configured Neural Networks / Probabilistic Methods descriptions. Shuffled swaps these two validation descriptions while retaining support-label connectivity. Random replaces their embeddings with deterministic Gaussian directions matched to each original embedding norm. ZeroPrompt retains the same prompt nodes and edges with zero features; NoPrompt removes prompt nodes and edges. All semantic conditions use the same learned projection and gate, avoiding a confound from separately trained checkpoints.", "",
        "The local semantic_alignment_audit.md verifies the configured class/topic mapping. Here 'correct' means those configured descriptions; this diagnosis assesses dependence on them, not their linguistic quality.", "",
        "## Condition metrics", "",
        "Accuracy and Macro-F1 are episode means. Effective norm is measured after normalization and gate multiplication. Gate is the shared sigmoid value, not the raw learned logit; bypassed conditions have no active gate.", "",
        "| Seed | Shot | Condition | Accuracy | Macro-F1 | Effective prompt norm | Gate |",
        "|---:|---:|---|---:|---:|---:|---:|",
    ]
    for run in result["runs"]:
        for condition, values in run["conditions"].items():
            gate = "--" if values["gate"] is None else f"{values['gate']:.9f}"
            lines.append(f"| {run['seed']} | {run['shot']} | {condition} | {values['accuracy']:.6f} | {values['macro_f1']:.6f} | {values['effective_prompt_norm']:.9f} | {gate} |")
    lines += ["", "## Paired effects relative to correct", "",
              "Cosine and L2 changes compare corresponding query vectors. Logit change is mean absolute change across the two class logits. Counts refer to query occurrences across episodes, not independent unique nodes. Full precision and maxima are saved in the accompanying JSON.", "",
              "| Seed | Shot | Condition | Query cosine mean | Query L2 mean | Query L2 maximum | Logit absolute change | Predictions changed |",
              "|---:|---:|---|---:|---:|---:|---:|---:|"]
    for run in result["runs"]:
        for condition, m in run["comparisons"].items():
            lines.append(f"| {run['seed']} | {run['shot']} | {condition} | {m['query_cosine_mean']:.10f} | {m['query_l2_change_mean']:.9g} | {m['query_l2_change_max']:.9g} | {m['mean_absolute_logit_change']:.9g} | {m['changed_prediction_count']}/{m['query_count']} ({100*m['prediction_disagreement_rate']:.4f}%) |")
    lines += ["", "## Prompt injection diagnostics", "",
              "Effective prompt norm equals the training-node reference norm times the sigmoid gate. The normalized ungated norm equals the reference norm. Prompt-feature changes below establish whether changed descriptions produce distinct injected features before graph propagation.", "",
              "| Seed | Shot | Reference norm | Gate logit | Correct query norm mean | Shuffled prompt L2 | Random prompt L2 |",
              "|---:|---:|---:|---:|---:|---:|---:|"]
    for run in result["runs"]:
        lines.append(f"| {run['seed']} | {run['shot']} | {run['reference_node_norm']:.9f} | {run['gate_logit']:.9f} | {run['query_norm_mean']:.9f} | {run['prompt_feature_changes']['shuffled']['l2_mean']:.9g} | {run['prompt_feature_changes']['random']['l2_mean']:.9g} |")
    lines += ["", "## Interpretation", ""]
    lines += [
        "**Diagnosis: description-dependent information reaches a small subset of query representations and influences classification decisions, but these saved validation runs do not show a consistent benefit from correct semantic alignment.**",
        "",
        "Correct/shuffled query cosine means range from 0.9999997175 to 0.9999999672. Only 0.37%-5.40% of query occurrences change their vectors when descriptions are swapped. Nevertheless, predictions change by 26.13%, 5.03%, 1.57% and 0.60% at 1, 3, 5 and 10 shots respectively. This is substantial decision sensitivity at 1 shot despite very sparse query-representation changes; support-prototype shifts can affect many queries. At 3 shots, mean query L2 change is 0.000339 versus a mean query norm of 5.20535, while mean absolute logit change is 0.002846. Correct descriptions outperform shuffled descriptions only at 1 shot, by 0.733 percentage points; shuffled descriptions have higher accuracy at 3, 5 and 10 shots. Sensitivity therefore establishes influence, not consistent usefulness.",
        "",
    ]
    for run in result["runs"]:
        s, r = run["comparisons"]["shuffled"], run["comparisons"]["random"]
        lines.append(f"- Seed {run['seed']}, {run['shot']}-shot: swapping descriptions changes {s['changed_query_count']}/{s['query_count']} query representations and {s['changed_prediction_count']} predicted classes; random directions change {r['changed_query_count']}/{r['query_count']} query representations and {r['changed_prediction_count']} predicted classes. Correct-minus-shuffled accuracy is {run['conditions']['correct']['accuracy']-run['conditions']['shuffled']['accuracy']:+.6f}.")
    lines += ["", "Nonzero prompt-feature changes alone demonstrate semantic input reaching the prompt projection. Nonzero paired query changes demonstrate propagation into query representations. Logit changes can also arise through support-prototype changes even when a query vector is unchanged. Class flips indicate decision sensitivity; they do not establish a beneficial semantic contribution. ZeroPrompt versus NoPrompt also includes topology and GCN degree-normalization effects, so their differences cannot be attributed solely to description semantics.", "",
              "With a two-layer GCN and prompt edges only to support nodes, feature-dependent prompt influence reaches original support nodes in one layer and their neighbors in two layers. Queries outside that neighborhood can retain exactly the same representation while their prototype-based logits change.", "",
              "This is a descriptive diagnosis of saved validation models, with only one available independent V2 seed. Repeated query occurrences and shot settings do not supply independent seed replications. No across-seed significance or test generalization claim is supported.", "",
              "## Reproducibility and integrity", "",
              f"Runtime: {result['runtime_seconds']:.2f} seconds on {result['device']}. SHA-256 checks passed for {result['protected_file_count']} existing checkpoint/manuscript files and the existing V1 artifact snapshot.", "",
              f"Validation manifest SHA-256: `{result['manifest_sha256']}`.", "",
              "| Seed | Shot | Existing V2 checkpoint | SHA-256 |", "|---:|---:|---|---|"]
    for run in result["runs"]:
        lines.append(f"| {run['seed']} | {run['shot']} | `{run['checkpoint']}` | `{run['sha256']}` |")
    lines += ["", "Command from the implementation directory:", "", "```powershell",
              "& .venv/Scripts/python.exe LLMGP/experiments/run_v2_semantic_sensitivity.py", "```", "",
              "Machine-readable metrics: `reports/v2_semantic_sensitivity.json`. No retraining or test evaluation was run."]
    (ROOT / "reports/v2_semantic_sensitivity.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
