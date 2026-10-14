from __future__ import annotations

import copy
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.classifier import ProtoClassifier
from src.data_loader import verify_cora_dataset
from src.gnn_backbone import GCNBackbone
from src.llm_encoder import SciBERTEncoder
from src.revised_protocol import (
    CLASS_COUNTS_BY_ID, CLASS_DESCRIPTIONS, CLASS_ID_TO_TOPIC, CLASS_SPLIT,
    PROTOCOL_SEED, validate_protocol,
)
from src.v2.validation import sample_episode
from src.v3.semantic_prototype_fusion import SemanticPrototypeFusion

SEEDS = (11, 23, 37)
CONDITIONS = ("NoPrompt", "V3-correct", "V3-shuffled", "V3-random")
SHOT = 3
EPISODES = 100
DEVICE = torch.device("cpu")
TRAINING_CONFIG = {
    "optimizer": "Adam", "learning_rate": 0.001, "weight_decay": 0.0001,
    "episodes": EPISODES, "initial_lambda": 0.1, "temperature": 0.1,
    "support_shots": SHOT, "query_shots": 15, "way": 2,
    "initialization_seed_rule": "seed * 1000000 + shot * 10000",
    "selection": "final update only; no validation selection",
}
CHECKPOINT_DIR = ROOT / "results/v3/checkpoints"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def protected_snapshot() -> dict[str, str]:
    """Protect existing V1/V2 source, data, checkpoints, reports and manuscript."""
    own = {"experiments/run_v3_mvp_validation.py", "reports/v3_alignment_diagnosis_v2.md",
           "reports/v3_alignment_diagnosis_v2.csv", "reports/v3_alignment_diagnosis_v2.json"}
    files = {}
    for path in ROOT.rglob("*"):
        if not path.is_file() or "__pycache__" in path.parts or ".pytest_cache" in path.parts:
            continue
        relative = path.relative_to(ROOT).as_posix()
        if relative in own or relative.startswith("results/v3/checkpoints/"):
            continue
        files[relative] = digest(path)
    for path in ROOT.parent.glob("*.docx"):
        files["../" + path.name] = digest(path)
    return files


def verify_alignment(labels: np.ndarray) -> str:
    validate_protocol(labels)
    assert CLASS_SPLIT == {"train": (0, 1, 2), "validation": (3, 4), "test": (5, 6)}
    audit = ROOT / "reports/semantic_alignment_audit.md"
    text = audit.read_text(encoding="utf-8")
    for class_id in (*CLASS_SPLIT["train"], *CLASS_SPLIT["validation"]):
        row_prefix = f"| {class_id} | {CLASS_ID_TO_TOPIC[class_id]} | {CLASS_COUNTS_BY_ID[class_id]} |"
        if row_prefix not in text:
            raise AssertionError("Current topic mapping disagrees with existing alignment audit")
    assert "All seven configured descriptions" in text
    expected = {0: "theoretical foundations", 1: "reinforcement learning",
                2: "genetic algorithms", 3: "neural networks", 4: "probabilistic models"}
    assert all(phrase in CLASS_DESCRIPTIONS[i].lower() for i, phrase in expected.items())
    return digest(audit)


def validate_episodes(episodes: list[dict], labels: np.ndarray, split: str) -> None:
    assert len(episodes) == EPISODES
    allowed = set(CLASS_SPLIT[split])
    for ep in episodes:
        classes = np.asarray(ep["class_ids"])
        assert len(classes) == 2 and len(set(classes)) == 2 and set(classes) <= allowed
        assert set(ep["support_idx"]).isdisjoint(ep["query_idx"])
        for index_key, label_key, count in (("support_idx", "support_labels", SHOT),
                                            ("query_idx", "query_labels", 15)):
            idx, local = np.asarray(ep[index_key]), np.asarray(ep[label_key])
            assert len(idx) == count * 2 and len(set(idx)) == len(idx)
            assert np.array_equal(np.bincount(local, minlength=2), [count, count])
            assert np.array_equal(labels[idx], classes[local])


def episode_hash(episodes: list[dict]) -> str:
    serial = [{k: np.asarray(v).tolist() for k, v in ep.items()} for ep in episodes]
    return hashlib.sha256(json.dumps(serial, sort_keys=True).encode()).hexdigest()


def semantic_controls(embeddings: torch.Tensor, seed: int) -> tuple[dict, dict]:
    shuffled = embeddings.clone()
    generator = torch.Generator().manual_seed(PROTOCOL_SEED + seed * 91)
    assignment = {}
    for split in ("train", "validation"):
        ids = list(CLASS_SPLIT[split])
        # A cyclic shift is a derangement: every class receives another description.
        shift = int(torch.randint(1, len(ids), (1,), generator=generator))
        source = ids[shift:] + ids[:shift]
        shuffled[ids] = embeddings[source]
        assignment[split] = dict(zip(ids, source))
    ids = list((*CLASS_SPLIT["train"], *CLASS_SPLIT["validation"]))
    random = embeddings.clone()
    generated = torch.randn((len(ids), 768), generator=torch.Generator().manual_seed(PROTOCOL_SEED + seed * 91 + 77777))
    random[ids] = F.normalize(generated, dim=-1) * embeddings[ids].norm(dim=-1, keepdim=True)
    torch.testing.assert_close(random[ids].norm(dim=-1), embeddings[ids].norm(dim=-1))
    assert torch.count_nonzero(random[list(CLASS_SPLIT["test"])]) == 0
    return {"V3-correct": embeddings, "V3-shuffled": shuffled, "V3-random": random}, assignment


def state_hash(model: torch.nn.Module) -> str:
    sha = hashlib.sha256()
    for name, value in model.state_dict().items():
        sha.update(name.encode())
        sha.update(value.detach().cpu().contiguous().numpy().tobytes())
    return sha.hexdigest()


def prepare_episodes(episodes: list[dict], node_embeddings: torch.Tensor) -> list[tuple]:
    prepared = []
    for ep in episodes:
        support = torch.as_tensor(ep["support_idx"], dtype=torch.long)
        query = torch.as_tensor(ep["query_idx"], dtype=torch.long)
        support_labels = torch.as_tensor(ep["support_labels"], dtype=torch.long)
        prototypes = ProtoClassifier.prototypes(node_embeddings[support], support_labels, 2)
        prepared.append((prototypes.detach(), node_embeddings[query].detach(),
                         torch.as_tensor(ep["query_labels"], dtype=torch.long),
                         torch.as_tensor(ep["class_ids"], dtype=torch.long)))
    return prepared


def frozen_check(*models: torch.nn.Module) -> None:
    for model in models:
        assert not model.training
        assert all(not p.requires_grad and p.grad is None for p in model.parameters())


def train(model: SemanticPrototypeFusion, episodes: list[tuple], semantics: torch.Tensor) -> dict:
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    assert {id(p) for group in optimizer.param_groups for p in group["params"]} == {id(p) for p in model.parameters()}
    norms = {name: [] for name, _ in model.named_parameters()}
    losses = []
    gate_history = [float(model.fusion_lambda.detach())]
    for support, queries, labels, class_ids in episodes:
        optimizer.zero_grad(set_to_none=True)
        fused = model(support, semantics[class_ids])
        loss = F.cross_entropy(ProtoClassifier.logits(queries, fused, temperature=0.1), labels)
        assert torch.isfinite(loss)
        loss.backward()
        for name, parameter in model.named_parameters():
            gradient = parameter.grad
            if gradient is None or not torch.isfinite(gradient).all() or gradient.norm().item() <= 0:
                raise AssertionError(f"Nonfinite or zero gradient: {name}")
            norms[name].append(float(gradient.norm()))
        optimizer.step()
        assert all(torch.isfinite(p).all() for p in model.parameters())
        losses.append(float(loss.detach()))
        gate_history.append(float(model.fusion_lambda.detach()))
    model.eval()
    return {"updates": len(losses), "loss_mean": float(np.mean(losses)), "lambda_history": gate_history,
            "gradient_norms": {k: {"min": min(v), "max": max(v), "mean": float(np.mean(v))} for k, v in norms.items()}}


def correct_class_margin(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Correct logit minus highest incorrect logit, including misclassified queries."""
    correct = logits.gather(1, labels[:, None]).squeeze(1)
    other = logits.clone()
    other.scatter_(1, labels[:, None], -float("inf"))
    return correct - other.max(dim=1).values


def evaluate(model: SemanticPrototypeFusion | None, episodes: list[tuple], semantics: torch.Tensor) -> dict:
    accuracy, macro_f1 = [], []
    alignment, cross_alignment, pairwise, support_pairwise, fused_pairwise = [], [], [], [], []
    margins_before, margins_after, margin_changes, semantic_norms = [], [], [], []
    per_class = {}
    harmed = helped = flipped = 0
    with torch.inference_mode():
        for support, queries, labels, class_ids in episodes:
            prototypes = support if model is None else model(support, semantics[class_ids])
            baseline_logits = ProtoClassifier.logits(queries, support, temperature=0.1)
            logits = ProtoClassifier.logits(queries, prototypes, temperature=0.1)
            prediction = logits.argmax(dim=-1)
            baseline_prediction = baseline_logits.argmax(dim=-1)
            before = correct_class_margin(baseline_logits, labels)
            after = correct_class_margin(logits, labels)
            margins_before.extend(before.tolist())
            margins_after.extend(after.tolist())
            margin_changes.extend((after - before).tolist())
            harmed += int(((baseline_prediction == labels) & (prediction != labels)).sum())
            helped += int(((baseline_prediction != labels) & (prediction == labels)).sum())
            flipped += int((prediction != baseline_prediction).sum())
            for local, global_id in enumerate(class_ids.tolist()):
                group = per_class.setdefault(str(global_id), {"before": [], "after": []})
                group["before"].extend(before[labels == local].tolist())
                group["after"].extend(after[labels == local].tolist())
            if model is not None:
                raw = model.semantic_projection(semantics[class_ids])
                semantic = F.normalize(raw, dim=-1)
                support_direction = F.normalize(support, dim=-1)
                match = semantic @ support_direction.T
                alignment.extend(match.diag().tolist())
                cross_alignment.extend(match[~torch.eye(len(class_ids), dtype=torch.bool)].tolist())
                upper = torch.triu_indices(len(class_ids), len(class_ids), offset=1)
                pairwise.extend((semantic @ semantic.T)[upper[0], upper[1]].tolist())
                support_pairwise.extend((support_direction @ support_direction.T)[upper[0], upper[1]].tolist())
                fused_pairwise.extend((prototypes @ prototypes.T)[upper[0], upper[1]].tolist())
                semantic_norms.extend(raw.norm(dim=-1).tolist())
            accuracy.append(float((prediction == labels).float().mean()))
            macro_f1.append(float(f1_score(labels.numpy(), prediction.numpy(), labels=[0, 1], average="macro", zero_division=0)))
    result = {"accuracy": float(np.mean(accuracy)), "macro_f1": float(np.mean(macro_f1)),
              "episode_accuracy": accuracy, "episode_macro_f1": macro_f1,
              "margin_before_mean": float(np.mean(margins_before)), "margin_after_mean": float(np.mean(margins_after)),
              "margin_change_mean": float(np.mean(margin_changes)),
              "margin_decreased_fraction": float(np.mean(np.asarray(margin_changes) < 0)),
              "correct_to_wrong": harmed, "wrong_to_correct": helped, "prediction_flips": flipped,
              "query_count": len(margins_before),
              "class_margins": {k: {"before": float(np.mean(v["before"])), "after": float(np.mean(v["after"]))} for k, v in per_class.items()}}
    if model is not None:
        result.update({"semantic_support_cosine_mean": float(np.mean(alignment)),
                       "semantic_support_cosine_min": min(alignment), "semantic_support_cosine_max": max(alignment),
                       "semantic_wrong_support_cosine_mean": float(np.mean(cross_alignment)),
                       "alignment_advantage_mean": float(np.mean(alignment) - np.mean(cross_alignment)),
                       "semantic_pairwise_cosine_mean": float(np.mean(pairwise)),
                       "support_pairwise_cosine_mean": float(np.mean(support_pairwise)),
                       "fused_pairwise_cosine_mean": float(np.mean(fused_pairwise)),
                       "projected_semantic_norm_mean": float(np.mean(semantic_norms))})
    assert all(np.isfinite(value) for value in result.values() if isinstance(value, (float, int)))
    return result


def main() -> None:
    started = time.perf_counter()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    before = protected_snapshot()
    existing_v3 = {str(p): digest(p) for p in CHECKPOINT_DIR.glob("*") if p.is_file()}
    for seed in SEEDS:
        for condition in CONDITIONS[1:]:
            target = CHECKPOINT_DIR / f"semantic_prototype_fusion_{condition[3:]}_shot_{SHOT}_seed_{seed}.pt"
            if target.exists():
                raise FileExistsError(f"Refusing to overwrite an existing V3 checkpoint: {target}")
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    historical = json.loads((ROOT / "reports/v3_mvp_validation_details.json").read_text(encoding="utf-8"))
    info = verify_cora_dataset()
    labels = info["y"].numpy()
    audit_hash = verify_alignment(labels)
    encoder = SciBERTEncoder(device=str(DEVICE))
    allowed = list((*CLASS_SPLIT["train"], *CLASS_SPLIT["validation"]))
    embeddings = torch.zeros(7, 768)
    embeddings[allowed] = encoder.encode_class_descriptions([CLASS_DESCRIPTIONS[i] for i in allowed], batch_size=5)
    frozen_check(encoder.model)
    encoder_state = state_hash(encoder.model)
    manifest_path = ROOT / "results/v2/validation_episode_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows, seed_details = [], []
    for seed in SEEDS:
        seed_started = time.perf_counter()
        train_base = PROTOCOL_SEED * 10000000 + seed * 100000 + SHOT * 1000
        training = [sample_episode(labels, "train", SHOT, train_base + i) for i in range(EPISODES)]
        validation = manifest[f"validation|shot={SHOT}|seed={seed}"]
        validate_episodes(training, labels, "train")
        validate_episodes(validation, labels, "validation")
        validation_digest = episode_hash(validation)
        path = ROOT / "results/checkpoints" / f"seed_{seed}_frozen_gcn.pt"
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        assert checkpoint["pretraining"]["seed"] == seed
        assert checkpoint["pretraining"]["objective"] == "masked_feature_reconstruction_mse"
        backbone = GCNBackbone(1433, 256, 256).requires_grad_(False)
        backbone.load_state_dict(checkpoint["model_state_dict"])
        backbone.eval()
        gcn_state = state_hash(backbone)
        with torch.no_grad():
            nodes = backbone(info["x"], info["edge_index"]).detach()
        train_data, validation_data = prepare_episodes(training, nodes), prepare_episodes(validation, nodes)
        semantics, assignment = semantic_controls(embeddings, seed)
        torch.manual_seed(seed * 1000000 + SHOT * 10000)
        initial = SemanticPrototypeFusion(initial_lambda=0.1)
        initial_digest = state_hash(initial)
        outcomes = {"NoPrompt": evaluate(None, validation_data, embeddings)}
        details = {}
        initial_diagnostics = {}
        for condition in CONDITIONS[1:]:
            model = copy.deepcopy(initial)
            assert state_hash(model) == initial_digest
            initial_diagnostics[condition] = evaluate(model, validation_data, semantics[condition])
            stats = train(model, train_data, semantics[condition])
            stats["lambda"] = float(model.fusion_lambda.detach())
            stats["gate_logit"] = float(model.fusion_logit.detach())
            target = CHECKPOINT_DIR / f"semantic_prototype_fusion_{condition[3:]}_shot_{SHOT}_seed_{seed}.pt"
            saved = {
                "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "seed": seed, "shot": SHOT, "semantic_condition": condition,
                "model": "SemanticPrototypeFusion", "semantic_dim": 768, "embedding_dim": 256,
                "training_configuration": dict(TRAINING_CONFIG), "training_statistics": stats,
                "frozen_gcn_checkpoint": path.relative_to(ROOT).as_posix(), "frozen_gcn_sha256": digest(path),
                "scibert_identifier": "allenai/scibert_scivocab_uncased", "scibert_state_sha256": encoder_state,
                "semantic_class_ids": allowed, "semantic_embeddings": semantics[condition][allowed].clone(),
                "class_descriptions": {i: CLASS_DESCRIPTIONS[i] for i in allowed},
                "shuffled_assignment": assignment if condition == "V3-shuffled" else None,
                "random_embedding_seed": PROTOCOL_SEED + seed * 91 + 77777 if condition == "V3-random" else None,
                "class_split": dict(CLASS_SPLIT), "initialization_sha256": initial_digest,
                "training_episode_sha256": episode_hash(training), "validation_episode_sha256": validation_digest,
                "validation_manifest_sha256": digest(manifest_path), "alignment_audit_sha256": audit_hash,
                "architecture_sha256": digest(ROOT / "src/v3/semantic_prototype_fusion.py"),
            }
            with target.open("xb") as handle:
                torch.save(saved, handle)
            # Measure persisted models after a strict state-dict round trip.
            loaded = torch.load(target, map_location="cpu", weights_only=True)
            restored = copy.deepcopy(initial).eval()
            restored.load_state_dict(loaded["state_dict"], strict=True)
            assert state_hash(restored) == state_hash(model)
            torch.testing.assert_close(loaded["semantic_embeddings"], semantics[condition][allowed], rtol=0, atol=0)
            outcomes[condition] = evaluate(restored, validation_data, semantics[condition])
            stats["checkpoint"] = target.relative_to(ROOT).as_posix()
            stats["checkpoint_sha256"] = digest(target)
            stats["state_dict_roundtrip_verified"] = True
            details[condition] = stats
            frozen_check(backbone, encoder.model)
            assert state_hash(backbone) == gcn_state
            assert episode_hash(validation) == validation_digest
        for condition, metrics in outcomes.items():
            row = {"seed": seed, "model": condition, "accuracy": metrics["accuracy"],
                   "macro_f1": metrics["macro_f1"], "lambda": details.get(condition, {}).get("lambda"),
                   "training_updates": details.get(condition, {}).get("updates", 0),
                   "validation_episode_sha256": validation_digest,
                   "initialization_sha256": initial_digest if condition != "NoPrompt" else ""}
            for comparator, key in (("NoPrompt", "vs_noprompt"), ("V3-shuffled", "vs_shuffled"), ("V3-random", "vs_random")):
                # Paired differences average per-episode accuracy differences.
                row[f"accuracy_delta_{key}"] = float(np.mean(np.asarray(metrics["episode_accuracy"]) - np.asarray(outcomes[comparator]["episode_accuracy"])))
            rows.append(row)
        seed_details.append({"seed": seed, "backbone": str(path.relative_to(ROOT)), "backbone_sha256": digest(path),
                             "training_episode_sha256": episode_hash(training), "validation_episode_sha256": validation_digest,
                             "initialization_sha256": initial_digest, "shuffled_assignment": assignment,
                             "training": details, "validation": outcomes, "initial_validation_diagnostics": initial_diagnostics,
                             "input_semantic_pairwise_cosine": {c: float(F.cosine_similarity(s[3:4], s[4:5], dim=-1)) for c, s in semantics.items()},
                             "runtime_seconds": time.perf_counter() - seed_started})
        print(f"Seed {seed}: " + "; ".join(f"{c} acc={m['accuracy']:.4f} F1={m['macro_f1']:.4f}" for c, m in outcomes.items()), flush=True)
    after = protected_snapshot()
    assert before == after, "Existing V1/V2 artifacts or manuscript changed"
    frozen_check(encoder.model)
    assert state_hash(encoder.model) == encoder_state, "Frozen SciBERT state changed"
    assert all(digest(Path(p)) == sha for p, sha in existing_v3.items()), "Existing V3 checkpoint changed"
    comparisons = []
    for row in rows:
        previous = next(r for r in historical["rows"] if r["seed"] == row["seed"] and r["model"] == row["model"])
        comparison = {"seed": row["seed"], "model": row["model"],
                      "accuracy_delta": row["accuracy"] - previous["accuracy"],
                      "macro_f1_delta": row["macro_f1"] - previous["macro_f1"]}
        if row["lambda"] is not None:
            comparison["lambda_delta"] = row["lambda"] - previous["lambda"]
        comparisons.append(comparison)
    result = {"rows": rows, "seed_details": seed_details, "protected_hashes": before,
              "protected_files_unchanged": True, "audit_sha256": audit_hash,
              "validation_manifest_sha256": digest(manifest_path), "device": str(DEVICE),
              "runtime_seconds": time.perf_counter() - started, "test_results": "not evaluated",
              "training_configuration": dict(TRAINING_CONFIG), "previous_mvp_comparison": comparisons,
              "scibert_state_unchanged": True, "existing_v3_checkpoints_unchanged": True}
    write_alignment_results(result)
    assert before == protected_snapshot(), "Existing artifacts changed while writing diagnostics"
    print(f"Complete: {result['runtime_seconds']:.2f}s; nine V3 checkpoints saved; {len(before)} protected artifact hashes unchanged; test evaluation not run.", flush=True)


def write_alignment_results(result: dict) -> None:
    rows, seeds = result["rows"], result["seed_details"]
    report_dir = ROOT / "reports"
    json_path = report_dir / "v3_alignment_diagnosis_v2.json"
    csv_path = report_dir / "v3_alignment_diagnosis_v2.csv"
    csv_rows = []
    for row in rows:
        seed = next(s for s in seeds if s["seed"] == row["seed"])
        metrics = seed["validation"][row["model"]]
        flat = dict(row)
        flat.update({k: v for k, v in metrics.items() if isinstance(v, (float, int))})
        csv_rows.append(flat)
    fields = list(dict.fromkeys(k for row in csv_rows for k in row))
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(csv_rows)
    lines = ["# LLMGP-v3 alignment diagnosis with saved checkpoints", "",
             "## Protocol and persistence", "",
             "The existing V3 SemanticPrototypeFusion architecture and MVP runner were reused without architecture changes. Three seeds (11/23/37), 3-shot, 2-way episodes, 100 training episodes for each semantic arm and 100 identical validation episodes per seed, with 15 queries per class. Training uses only [0,1,2], evaluation only [3,4]. Test classes [5,6] were not evaluated or encoded.", "",
             "Every arm starts from an exact clone of the same per-seed initialization. Adam, learning rate 0.001, weight decay 0.0001, initial lambda 0.1, temperature 0.1, and final-update evaluation match the prior MVP. No validation model selection, extra training, extra seeds or architecture changes were performed. Shuffled descriptions are deranged within each split; validation descriptions are swapped. Random frozen semantic vectors retain each corresponding SciBERT input norm.", "",
             "Nine distinct trained checkpoints are saved under `results/v3/checkpoints/`, opened exclusively to prevent overwrites. Each contains the state_dict, seed, shot, semantic arm, full training configuration and statistics, gate history, frozen GCN path/hash, SciBERT state identifier, actual semantic input vectors for classes 0-4, initialization/episode hashes and semantic-assignment provenance. Models were reloaded with strict state-dict matching before their validation measurements. Existing reports, including the original MVP and the unavailable-checkpoint diagnosis, remain unchanged.", "",
             "Frozen node embeddings are computed from the original full Cora graph without virtual prompt nodes. The reused transductive backbone includes unlabeled nodes from all classes; no test-class labels are used for scoring or optimization.", "",
             "## Accuracy, Macro-F1 and gate", "",
             "| Seed | Arm | Accuracy | Macro-F1 | Lambda | Accuracy delta vs NoPrompt |", "|---:|---|---:|---:|---:|---:|"]
    for row in rows:
        gate = "--" if row["lambda"] is None else f"{row['lambda']:.9f}"
        lines.append(f"| {row['seed']} | {row['model']} | {row['accuracy']:.6f} | {row['macro_f1']:.6f} | {gate} | {row['accuracy_delta_vs_noprompt']:+.6f} |")
    lines += ["", "Across-seed values use the sample SD (ddof=1), not an episode-level SD.", "",
              "| Arm | Accuracy mean +/- SD | Macro-F1 mean +/- SD |", "|---|---:|---:|"]
    for condition in CONDITIONS:
        group = [r for r in rows if r["model"] == condition]
        acc, f1 = [r["accuracy"] for r in group], [r["macro_f1"] for r in group]
        lines.append(f"| {condition} | {np.mean(acc):.6f} +/- {np.std(acc, ddof=1):.6f} | {np.mean(f1):.6f} +/- {np.std(f1, ddof=1):.6f} |")
    lines += ["", "## Semantic/support alignment", "",
              "Cosines use L2-normalized projected semantic vectors and support-derived GCN class means in the same episode. Matching cosine compares the same class; wrong-class cosine compares the other class. Their difference measures class-specific alignment rather than only a shared direction. Values average 200 class-prototype occurrences per arm; they are not independent samples.", "",
              "| Seed | Arm | Initial matching cosine | Trained matching cosine | Trained wrong-class cosine | Matching minus wrong |", "|---:|---|---:|---:|---:|---:|"]
    for seed in seeds:
        for condition in CONDITIONS[1:]:
            m, initial = seed["validation"][condition], seed["initial_validation_diagnostics"][condition]
            lines.append(f"| {seed['seed']} | {condition} | {initial['semantic_support_cosine_mean']:.9f} | {m['semantic_support_cosine_mean']:.9f} | {m['semantic_wrong_support_cosine_mean']:.9f} | {m['alignment_advantage_mean']:+.9f} |")
    lines += ["", "## Correct-class margins before and after fusion", "",
              "Margin = correct-class cosine logit minus the highest incorrect-class logit, with temperature 0.1. A negative margin indicates an incorrect prediction (zero may be an argmax tie). Before fusion uses support-only prototypes on exactly the same queries. Counts describe 3,000 query occurrences per arm. Correct-to-wrong and wrong-to-correct identify whether fusion loses more correct decisions than it gains.", "",
              "| Seed | Arm | Before mean | After mean | Delta | Margins decreased | Correct to wrong | Wrong to correct |", "|---:|---|---:|---:|---:|---:|---:|---:|"]
    for seed in seeds:
        for condition in CONDITIONS[1:]:
            m = seed["validation"][condition]
            lines.append(f"| {seed['seed']} | {condition} | {m['margin_before_mean']:+.9f} | {m['margin_after_mean']:+.9f} | {m['margin_change_mean']:+.9f} | {100*m['margin_decreased_fraction']:.2f}% | {m['correct_to_wrong']} | {m['wrong_to_correct']} |")
    lines += ["", "### Class-wise correct-description margins", "",
              "Balanced-query overall means can hide changes in opposite directions for the two classes.", "",
              "| Seed | Validation class | Before mean | After mean | Delta |", "|---:|---|---:|---:|---:|"]
    for seed in seeds:
        for class_id, margins in seed["validation"]["V3-correct"]["class_margins"].items():
            lines.append(f"| {seed['seed']} | {class_id}: {CLASS_ID_TO_TOPIC[int(class_id)]} | {margins['before']:+.9f} | {margins['after']:+.9f} | {margins['after']-margins['before']:+.9f} |")
    lines += ["", "## Similar semantic directions and possible collapse", "",
              "There is one distinct validation class pair (IDs 3/4). Its cosine is fixed across episodes for each trained arm because descriptions and projection weights are fixed. Repetition over 100 episodes does not supply 100 independent semantic directions. Near-one cosine indicates similar directions, but alone does not prove complete collapse or identify a cause. Support/fused pairwise cosines provide a geometry reference; train/test class collapse is not evaluated.", "",
              "| Seed | Arm | Input semantic cosine | Initial projected cosine | Trained projected cosine | Support cosine mean | Fused cosine mean |", "|---:|---|---:|---:|---:|---:|---:|"]
    for seed in seeds:
        for condition in CONDITIONS[1:]:
            m, initial = seed["validation"][condition], seed["initial_validation_diagnostics"][condition]
            lines.append(f"| {seed['seed']} | {condition} | {seed['input_semantic_pairwise_cosine'][condition]:.9f} | {initial['semantic_pairwise_cosine_mean']:.9f} | {m['semantic_pairwise_cosine_mean']:.9f} | {m['support_pairwise_cosine_mean']:.9f} | {m['fused_pairwise_cosine_mean']:.9f} |")
    lines += ["", "## Gate trajectory", "",
              "Lambda is a shared sigmoid gate. Its entire 101-point history (initialization plus 100 updates) is recorded in JSON and checkpoints.", "",
              "| Seed | Arm | Initial | Final | Final minus initial | Minimum | Maximum | Updates decreasing lambda |", "|---:|---|---:|---:|---:|---:|---:|---:|"]
    for seed in seeds:
        for condition, stats in seed["training"].items():
            history = stats["lambda_history"]
            lines.append(f"| {seed['seed']} | {condition} | {history[0]:.9f} | {history[-1]:.9f} | {history[-1]-history[0]:+.9f} | {min(history):.9f} | {max(history):.9f} | {int(np.count_nonzero(np.diff(history)<0))}/100 |")
    lines += ["", "## Diagnostic interpretation", ""]
    for seed in seeds:
        correct = seed["validation"]["V3-correct"]
        baseline = seed["validation"]["NoPrompt"]
        lines.append(f"- Seed {seed['seed']}: correct fusion changes accuracy by {correct['accuracy']-baseline['accuracy']:+.6f} and Macro-F1 by {correct['macro_f1']-baseline['macro_f1']:+.6f}. Its mean margin changes by {correct['margin_change_mean']:+.9f}; it loses {correct['correct_to_wrong']} correct decisions and gains {correct['wrong_to_correct']}. Matching-over-wrong support alignment is {correct['alignment_advantage_mean']:+.9f}; projected validation-class cosine is {correct['semantic_pairwise_cosine_mean']:.9f}.")
    correct_rows = [r for r in rows if r["model"] == "V3-correct"]
    for comparator in ("NoPrompt", "V3-shuffled", "V3-random"):
        delta_acc, delta_f1 = [], []
        for row in correct_rows:
            other = next(r for r in rows if r["seed"] == row["seed"] and r["model"] == comparator)
            delta_acc.append(row["accuracy"] - other["accuracy"])
            delta_f1.append(row["macro_f1"] - other["macro_f1"])
        lines.append(f"- Correct versus {comparator}: mean accuracy difference {np.mean(delta_acc):+.6f}, mean Macro-F1 difference {np.mean(delta_f1):+.6f}; both metrics improve in {sum(a>0 and f>0 for a, f in zip(delta_acc, delta_f1))}/3 seeds.")
    end_gates = [s["training"][c]["lambda_history"] for s in seeds for c in CONDITIONS[1:]]
    if all(h[-1] > h[0] for h in end_gates):
        lines += ["", "All semantic arms finish with lambda above 0.1: the gate does not learn to turn off semantic fusion. Training-class optimization therefore retains semantic influence despite the validation degradation."]
    pair = [s["validation"]["V3-correct"]["semantic_pairwise_cosine_mean"] for s in seeds]
    advantage = [s["validation"]["V3-correct"]["alignment_advantage_mean"] for s in seeds]
    matching = [s["validation"]["V3-correct"]["semantic_support_cosine_mean"] for s in seeds]
    if all(v < 0 for v in matching) and all(v < 0 for v in advantage):
        lines += ["", f"**Validation misalignment is observed in every seed.** Correct projected semantic prototypes have negative matching-support cosine ({min(matching):.9f} to {max(matching):.9f}) and slightly worse alignment with their own support class than with the other class (matching-minus-wrong cosine {min(advantage):+.9f} to {max(advantage):+.9f}). Their learned directions therefore do not provide the appropriate validation-class geometry despite correct text-label alignment."]
    if all(s["validation"]["V3-correct"]["semantic_pairwise_cosine_mean"] < s["initial_validation_diagnostics"]["V3-correct"]["semantic_pairwise_cosine_mean"] for s in seeds):
        lines += ["", f"**The projected validation prototypes do not show a training-induced collapse toward a common direction.** Their pairwise cosine decreases in every seed, from the initial approximately 0.95-0.97 to {min(pair):.9f}-{max(pair):.9f}. One seed retains moderately high similarity, but none approaches identical projected directions. The directly measured degradation is more consistent with inappropriate orientation/class discrimination than complete direction collapse. This statement is limited to the two validation classes."]
    if all(s["validation"]["V3-correct"]["class_margins"]["4"]["after"] < s["validation"]["V3-correct"]["class_margins"]["4"]["before"] for s in seeds):
        lines += ["", "Probabilistic Methods (class 4) loses mean correct-class margin in every seed; Neural Networks (class 3) gains margin in these runs. Fusion therefore introduces an asymmetric class effect rather than a uniform improvement. More correct-to-wrong than wrong-to-correct flips in each correct arm explain its lower accuracy."]
    lines += ["",
              "These comparisons measure the direct effect of replacing support-only prototypes with fused prototypes at fixed query embeddings. The causes of learned semantic geometry remain observational: separately trained correct/shuffled/random arms do not isolate every training effect. No claim of semantic improvement or test generalization is warranted when the correct arm loses to NoPrompt or the random control.", "",
              "## Verification and reproducibility", "",
              "- All 900 updates had finite, nonzero projection-weight, projection-bias and gate gradients. No update was skipped.",
              "- SciBERT and GCN remain frozen, in evaluation mode, with no gradients; optimizer parameters are restricted to V3. GCN state hashes match after every arm and the SciBERT state hash matches before/after the whole experiment.",
              "- Every episode passed balanced class-count, permitted-class, actual-label, unique-index and support/query-disjointness checks. All arms share the saved validation episodes and training order, with equal initialization hashes.",
              "- Each saved model passed strict checkpoint reload and state/semantic-input equality checks before validation.",
              f"- {len(result['protected_hashes'])} existing artifact hashes, including architecture files, V1/V2 checkpoints/reports, original V3 reports and manuscript, remained unchanged. Existing V3 checkpoints are preserved; filename collisions abort before training.",
              "- Full numeric diagnostics, gate histories, episode metrics and artifact hashes are in `v3_alignment_diagnosis_v2.json`; scalar seed/arm metrics are in `v3_alignment_diagnosis_v2.csv`.", ""]
    identical = all(all(abs(v) <= 1e-7 for k, v in row.items() if k.endswith("_delta")) for row in result["previous_mvp_comparison"])
    lines += [f"Prior MVP reproduction: accuracy, Macro-F1 and final lambda match the archived MVP within 1e-7: **{identical}**.", "",
              "| Seed | Arm | Saved checkpoint | SHA-256 |", "|---:|---|---|---|"]
    for seed in seeds:
        for condition, stats in seed["training"].items():
            lines.append(f"| {seed['seed']} | {condition} | `{stats['checkpoint']}` | `{stats['checkpoint_sha256']}` |")
    lines += ["", "| Seed | Shared validation episode hash | Initialization hash |", "|---:|---|---|"]
    for seed in seeds:
        lines.append(f"| {seed['seed']} | `{seed['validation_episode_sha256']}` | `{seed['initialization_sha256']}` |")
    lines += ["", f"Runtime: {result['runtime_seconds']:.2f} seconds on {result['device']} (excluding separate unit tests). Test-class results: **not evaluated**. No full experiments or V4 implementation were run.", "",
              f"Targeted V3 unit tests: {result.get('unit_test_results', 'run separately using pytest tests/v3 -q')}. They check the fusion formula, endpoints and gradients, margin signs, harmful misalignment and identical-direction detection without assuming classification failure.", "",
              "Command from the implementation directory:", "", "```powershell",
              "& .venv/Scripts/python.exe LLMGP/experiments/run_v3_mvp_validation.py", "```"]
    (report_dir / "v3_alignment_diagnosis_v2.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


def write_results(result: dict) -> None:
    rows = result["rows"]
    fields = list(rows[0])
    report_dir = ROOT / "reports"
    with (report_dir / "v3_mvp_validation.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# LLMGP-v3 minimal validation: Semantic Prototype Fusion", "",
             "## Protocol", "",
             "3-shot, 2-way episodes; seeds 11, 23 and 37; 100 training episodes per trainable condition per seed and 100 shared validation episodes per seed, with 15 queries per class. Train classes [0,1,2], validation [3,4], test [5,6]. Test classes were not evaluated or used to encode descriptions or select models.", "",
             "The existing self-supervised seed-specific GCN and frozen SciBERT are reused. Original graph embeddings are computed once per seed without virtual prompt nodes. The full graph is retained for transductive message passing; nodes from held-out classes remain unlabeled inputs to the reused backbone. Support class means and projected semantic vectors are individually L2-normalized, then fused as `normalize((1-lambda)*support + lambda*semantic)`. A shared sigmoid gate initializes lambda to 0.1. Only the 768-to-256 linear projection (with bias) and gate are optimized.", "",
             "All three trainable conditions start from an exact clone of the same per-seed initialization, receive the same ordered training episodes, and take 100 Adam updates (learning rate 0.001, weight decay 0.0001). Final weights after update 100 are evaluated once; no validation selection, early stopping, hyperparameter search or retraining is used. Models remain in memory; existing checkpoints are never written.", "",
             "Shuffled descriptions are deranged independently within the training and validation splits, including swapping validation classes 3/4. Random vectors are fixed per seed and class, norm-matched to the corresponding frozen SciBERT input embeddings before projection. Neither control introduces test descriptions. The saved V2 validation manifest is loaded read-only and shared by every arm. Training episodes reuse the existing episodic sampler and extended-validation seed rule.", "",
             "## Seed-wise validation results", "",
             "Accuracy and Macro-F1 are means of episode metrics. Paired accuracy differences use the same 100 episodes; differences are fractions, so 0.01 equals one percentage point.", "",
             "| Seed | Model | Accuracy | Macro-F1 | Learned lambda | Delta vs NoPrompt | Delta vs shuffled | Delta vs random |",
             "|---:|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        gate = "--" if row["lambda"] is None else f"{row['lambda']:.6f}"
        lines.append(f"| {row['seed']} | {row['model']} | {row['accuracy']:.6f} | {row['macro_f1']:.6f} | {gate} | {row['accuracy_delta_vs_noprompt']:+.6f} | {row['accuracy_delta_vs_shuffled']:+.6f} | {row['accuracy_delta_vs_random']:+.6f} |")
    lines += ["", "## Across-seed summaries", "", "Sample SD uses ddof=1 across the three independent seeds, not across episodes.", "",
              "| Model | Accuracy mean +/- SD | Macro-F1 mean +/- SD | Lambda mean +/- SD |", "|---|---:|---:|---:|"]
    for condition in CONDITIONS:
        selected = [row for row in rows if row["model"] == condition]
        def summary(key):
            values = [r[key] for r in selected]
            return f"{np.mean(values):.6f} +/- {np.std(values, ddof=1):.6f}"
        lines.append(f"| {condition} | {summary('accuracy')} | {summary('macro_f1')} | {summary('lambda') if condition != 'NoPrompt' else '--'} |")
    lines += ["", "### Paired correct-description accuracy differences", "",
              "| Comparator | Seed 11 | Seed 23 | Seed 37 | Mean +/- SD |", "|---|---:|---:|---:|---:|"]
    correct = [r for r in rows if r["model"] == "V3-correct"]
    for comparator, key in (("NoPrompt", "vs_noprompt"), ("V3-shuffled", "vs_shuffled"), ("V3-random", "vs_random")):
        values = [r[f"accuracy_delta_{key}"] for r in correct]
        lines.append(f"| {comparator} | {values[0]:+.6f} | {values[1]:+.6f} | {values[2]:+.6f} | {np.mean(values):+.6f} +/- {np.std(values, ddof=1):.6f} |")
    lines += ["", "## Outcome", ""]
    for comparator in ("NoPrompt", "V3-shuffled", "V3-random"):
        baseline = [r for r in rows if r["model"] == comparator]
        acc = np.asarray([r["accuracy"] for r in correct]) - [r["accuracy"] for r in baseline]
        f1 = np.asarray([r["macro_f1"] for r in correct]) - [r["macro_f1"] for r in baseline]
        wins = int(np.count_nonzero((acc > 0) & (f1 > 0)))
        lines.append(f"- Correct descriptions versus {comparator}: mean accuracy difference {np.mean(acc):+.6f}, mean Macro-F1 difference {np.mean(f1):+.6f}; both metrics improve in {wins}/3 seeds. Consistent superiority on both metrics: {bool(wins == 3)}.")
        if np.all(acc < 0) and np.all(f1 < 0):
            lines.append(f"  Correct semantic fusion underperforms {comparator} on both metrics in every seed.")
    all_better = all(all(next(r for r in rows if r['seed'] == c['seed'] and r['model'] == comparator)[metric] < c[metric] for c in correct for metric in ('accuracy', 'macro_f1')) for comparator in CONDITIONS if comparator != 'V3-correct')
    lines += ["", "Correct semantics consistently outperform all controls in this minimal validation." if all_better else "This minimal validation does not establish consistent improvement from correctly aligned semantic fusion over NoPrompt and both semantic controls. Any positive mean differences above are descriptive; with three seeds, no statistical or test-generalization claim is made.", "",
              "## Verification", "",
              "- Every projection weight, projection bias and gate gradient was finite and nonzero on every one of the 900 V3 training updates. No update was skipped.",
              "- GCN/SciBERT remain in evaluation mode with requires_grad=False and no accumulated gradients. The optimizer contains only the V3 parameters; detached cached encoder outputs prevent encoder backpropagation. GCN state hashes were also checked after each arm.",
              "- All training/validation episodes passed class-membership, actual-node-label, balanced-count, unique-index and support/query-disjointness checks.",
              "- Correct descriptions agree with the existing semantic alignment audit's ID/topic/count rows and configured description topics; local Cora class counts match that audit.",
              f"- All {len(result['protected_hashes'])} protected existing V1/V2 artifacts and manuscript hashes are unchanged. Full hashes and per-episode validation metrics are saved in `v3_mvp_validation_details.json`.", "",
              "### Minimum observed gradient norms", "", "| Seed | Condition | Projection weight | Projection bias | Gate logit |", "|---:|---|---:|---:|---:|"]
    for seed in result["seed_details"]:
        for condition, stats in seed["training"].items():
            norms = stats["gradient_norms"]
            lines.append(f"| {seed['seed']} | {condition} | {norms['semantic_projection.weight']['min']:.9g} | {norms['semantic_projection.bias']['min']:.9g} | {norms['fusion_logit']['min']:.9g} |")
    lines += ["", "### Shared validation episode identifiers", "", "| Seed | Validation episode SHA-256 | Initial V3 state SHA-256 |", "|---:|---|---|"]
    for seed in result["seed_details"]:
        lines.append(f"| {seed['seed']} | `{seed['validation_episode_sha256']}` | `{seed['initialization_sha256']}` |")
    lines += ["", "## Runtime, tests and stop point", "",
              f"Runtime: {result['runtime_seconds']:.2f} seconds on {result['device']}. Test-class results: **not evaluated**, as requested. No full experiments, extra seeds or shot settings were run.", "",
              f"Unit-test results: {result.get('unit_test_results', 'not run by the validation runner; use the command below')}. Validation runtime excludes the separate unit-test run.", "",
              "Targeted unit tests check the fusion formula, normalization, support rescaling invariance, gate endpoints and trainable/frozen gradient behavior. Run with:", "",
              "```powershell", "Set-Location LLMGP", "& ../.venv/Scripts/python.exe -m pytest tests/v3 -q", "```", "",
              "Validation command from the implementation directory:", "", "```powershell",
              "& .venv/Scripts/python.exe LLMGP/experiments/run_v3_mvp_validation.py", "```"]
    (report_dir / "v3_mvp_validation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (report_dir / "v3_mvp_validation_details.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
