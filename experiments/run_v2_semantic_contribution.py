from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import t
from torch.optim import Adam

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import verify_cora_dataset
from src.gnn_backbone import GCNBackbone
from src.llm_encoder import SciBERTEncoder
from src.prompt_builder import PromptBuilder
from src.revised_protocol import CLASS_DESCRIPTIONS, CLASS_SPLIT, PROTOCOL_SEED
from src.v2.prompt_models import NormalizedGatedPrompt
from src.v2.validation import (
    episode_forward,
    evaluate_episodes,
    sample_episode,
    snapshot_v1_artifacts,
    validate_trainable_gradients,
)


SEEDS = (11, 23, 37)
SHOT = 3
TRAIN_EPISODES = 100
VALIDATION_EPISODES = 100
INITIAL_GATE = 0.1
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEMANTIC_ARMS = ("correct", "shuffled", "random")


def ci95(values: list[float]) -> dict[str, float]:
    if len(values) < 2:
        raise ValueError("At least two seeds are required for an across-seed confidence interval")
    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1))
    margin = float(t.ppf(0.975, df=len(values) - 1) * sd / np.sqrt(len(values)))
    return {
        "mean": mean,
        "sd": sd,
        "ci95_lower": mean - margin,
        "ci95_upper": mean + margin,
    }


def configured_embeddings(encoder: SciBERTEncoder) -> tuple[torch.Tensor, dict[int, float]]:
    allowed_ids = (*CLASS_SPLIT["train"], *CLASS_SPLIT["validation"])
    texts = [CLASS_DESCRIPTIONS[class_id] for class_id in allowed_ids]
    encoded = encoder.encode_class_descriptions(texts, batch_size=len(texts))
    if encoded.shape != (len(allowed_ids), 768):
        raise AssertionError("Expected five 768-dimensional descriptions, excluding test classes")
    if any(parameter.requires_grad for parameter in encoder.model.parameters()):
        raise AssertionError("SciBERT parameters must remain frozen")
    embeddings = torch.zeros((7, 768), dtype=encoded.dtype)
    for offset, class_id in enumerate(allowed_ids):
        embeddings[class_id] = encoded[offset]
    return embeddings.to(DEVICE), {class_id: float(embeddings[class_id].norm().item()) for class_id in allowed_ids}


def variant_embeddings(
    configured: torch.Tensor,
    condition: str,
    seed: int,
    norms: dict[int, float],
) -> tuple[torch.Tensor, list[int] | None]:
    output = configured.clone()
    permitted = (*CLASS_SPLIT["train"], *CLASS_SPLIT["validation"])
    if condition == "correct":
        return output, None
    if condition == "shuffled":
        generator = torch.Generator(device=DEVICE).manual_seed(PROTOCOL_SEED + seed * 101)
        permutation = torch.randperm(len(permitted), generator=generator).tolist()
        if permutation == list(range(len(permitted))):
            permutation = permutation[1:] + permutation[:1]
        output[list(permitted)] = configured[[permitted[index] for index in permutation]]
        return output, [permitted[index] for index in permutation]
    if condition == "random":
        generator = torch.Generator(device=DEVICE).manual_seed(PROTOCOL_SEED + seed * 1009 + 31)
        random_vectors = torch.randn((len(permitted), 768), generator=generator, device=DEVICE)
        random_vectors = F.normalize(random_vectors, dim=-1)
        random_vectors = random_vectors * torch.tensor(
            [norms[class_id] for class_id in permitted], dtype=random_vectors.dtype, device=DEVICE
        ).unsqueeze(-1)
        output[list(permitted)] = random_vectors
        return output, None
    raise ValueError(f"Unknown semantic condition {condition}")


def load_gcn(seed: int) -> tuple[GCNBackbone, dict[str, Any]]:
    path = ROOT / "results" / "checkpoints" / f"seed_{seed}_frozen_gcn.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    metadata = checkpoint["pretraining"]
    if int(metadata["seed"]) != seed:
        raise AssertionError("Frozen GCN checkpoint seed mismatch")
    backbone = GCNBackbone(1433, 256, 256)
    backbone.load_state_dict(checkpoint["model_state_dict"])
    backbone.to(DEVICE).eval()
    for parameter in backbone.parameters():
        parameter.requires_grad = False
        parameter.grad = None
    return backbone, {
        "checkpoint": str(path.relative_to(ROOT)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def make_episode_pair(labels: np.ndarray, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train_start = PROTOCOL_SEED * 10_000_000 + seed * 100_000 + SHOT * 1_000
    validation_start = train_start + 50_000
    training = [
        sample_episode(labels, "train", SHOT, train_start + index)
        for index in range(TRAIN_EPISODES)
    ]
    validation = [
        sample_episode(labels, "validation", SHOT, validation_start + index)
        for index in range(VALIDATION_EPISODES)
    ]
    for episode in [*training, *validation]:
        if not set(episode["support_idx"].tolist()).isdisjoint(episode["query_idx"].tolist()):
            raise AssertionError("Support and query nodes overlap")
    if any(not set(episode["class_ids"].tolist()).issubset(CLASS_SPLIT["train"]) for episode in training):
        raise AssertionError("Training episode used a non-training class")
    if any(episode["class_ids"].tolist() != list(CLASS_SPLIT["validation"]) for episode in validation):
        raise AssertionError("Validation episodes must use classes [3,4]")
    return training, validation


def fit_fixed_budget(
    *,
    seed: int,
    class_embeddings: torch.Tensor,
    training_episodes: list[dict[str, Any]],
    data: Any,
    backbone: GCNBackbone,
    reference_norm: float,
) -> tuple[NormalizedGatedPrompt, dict[str, Any]]:
    # Same initialization per seed across all semantic arms.
    torch.manual_seed(seed * 1_000_000 + SHOT * 10_000 + 101)
    model = NormalizedGatedPrompt(reference_node_norm=reference_norm, initial_gate=INITIAL_GATE).to(DEVICE)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    x = data.x.to(DEVICE)
    edge_index = data.edge_index.to(DEVICE)
    losses: list[float] = []
    grad_norms: list[float] = []
    zero_gradient_episodes = 0
    for episode in training_episodes:
        loss, _, _, _ = episode_forward(episode, x, edge_index, class_embeddings, model, backbone)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if any(parameter.grad is not None for parameter in backbone.parameters()):
            raise AssertionError("Frozen GCN received gradients")
        if not validate_trainable_gradients(model):
            zero_gradient_episodes += 1
            continue
        grad_norms.append(float(torch.sqrt(sum(p.grad.detach().square().sum() for p in model.parameters())).item()))
        optimizer.step()
        losses.append(float(loss.detach().item()))
    if not losses:
        raise RuntimeError("All training episodes had zero trainable gradients")
    model.eval()
    return model, {
        "training_episodes_requested": len(training_episodes),
        "training_episodes_updated": len(losses),
        "zero_gradient_episodes": zero_gradient_episodes,
        "training_loss_mean": float(np.mean(losses)),
        "gradient_norm_mean": float(np.mean(grad_norms)),
        "gate": float(model.gate.item()),
    }


def main() -> None:
    started = time.perf_counter()
    torch.set_num_threads(1)
    before = snapshot_v1_artifacts(ROOT)
    info = verify_cora_dataset()
    labels = info["y"].cpu().numpy()
    encoder = SciBERTEncoder(model_name="allenai/scibert_scivocab_uncased", device=str(DEVICE))
    embeddings, embedding_norms = configured_embeddings(encoder)
    train_nodes = np.isin(labels, np.asarray(CLASS_SPLIT["train"], dtype=np.int64))
    reference_norm = float(info["x"][torch.from_numpy(train_nodes)].norm(dim=-1).mean().item())
    seed_rows: list[dict[str, Any]] = []
    gcn_identifiers: dict[int, dict[str, Any]] = {}
    semantic_permutations: dict[int, list[int]] = {}

    for seed in SEEDS:
        backbone, gcn_identifier = load_gcn(seed)
        gcn_identifiers[seed] = gcn_identifier
        training_episodes, validation_episodes = make_episode_pair(labels, seed)
        shuffled_for_check, permutation = variant_embeddings(embeddings, "shuffled", seed, embedding_norms)
        del shuffled_for_check
        semantic_permutations[seed] = permutation or []

        v1_path = ROOT / "results" / "checkpoints" / f"shot_{SHOT}_seed_{seed}_prompt.pt"
        v1_checkpoint = torch.load(v1_path, map_location="cpu", weights_only=False)
        v1_meta = v1_checkpoint["metadata"]
        if int(v1_meta["seed"]) != seed or int(v1_meta["shot"]) != SHOT:
            raise AssertionError("Saved V1 prompt checkpoint seed/shot mismatch")
        if not str(v1_meta["backbone_checkpoint"]).replace("\\", "/").endswith(f"seed_{seed}_frozen_gcn.pt"):
            raise AssertionError("Saved V1 prompt and GCN checkpoint seed mismatch")
        v1_prompt = PromptBuilder().to(DEVICE)
        v1_prompt.load_state_dict(v1_checkpoint["prompt_state_dict"])
        v1_prompt.eval()
        v1_metrics = evaluate_episodes(
            validation_episodes,
            info["x"].to(DEVICE),
            info["edge_index"].to(DEVICE),
            embeddings,
            v1_prompt,
            backbone,
            "connected",
        )
        seed_rows.append({"seed": seed, "condition": "V1 OriginalPrompt", **v1_metrics, "gate": None, "training_loss": None})

        for condition in SEMANTIC_ARMS:
            arm_embeddings, _ = variant_embeddings(embeddings, condition, seed, embedding_norms)
            model, training = fit_fixed_budget(
                seed=seed,
                class_embeddings=arm_embeddings,
                training_episodes=training_episodes,
                data=info["data"],
                backbone=backbone,
                reference_norm=reference_norm,
            )
            metrics = evaluate_episodes(
                validation_episodes,
                info["x"].to(DEVICE),
                info["edge_index"].to(DEVICE),
                arm_embeddings,
                model,
                backbone,
                "connected",
            )
            seed_rows.append(
                {
                    "seed": seed,
                    "condition": f"V2 {condition}",
                    **metrics,
                    "gate": training["gate"],
                    "training_loss": training["training_loss_mean"],
                    "training_episodes_requested": training["training_episodes_requested"],
                    "training_episodes_updated": training["training_episodes_updated"],
                    "zero_gradient_episodes": training["zero_gradient_episodes"],
                    "gradient_norm_mean": training["gradient_norm_mean"],
                }
            )
        no_prompt = evaluate_episodes(
            validation_episodes,
            info["x"].to(DEVICE),
            info["edge_index"].to(DEVICE),
            embeddings,
            None,
            backbone,
            "none",
        )
        seed_rows.append({"seed": seed, "condition": "NoPrompt", **no_prompt, "gate": None, "training_loss": None})
        del backbone
        print(f"completed seed={seed}; validation episodes shared across all conditions", flush=True)

    if snapshot_v1_artifacts(ROOT) != before:
        raise AssertionError("Existing v1 artifact hashes changed")

    aggregates: list[dict[str, Any]] = []
    for condition in ("V1 OriginalPrompt", "V2 correct", "V2 shuffled", "V2 random", "NoPrompt"):
        rows = [row for row in seed_rows if row["condition"] == condition]
        if not rows:
            continue
        accuracy = ci95([float(row["accuracy"]) for row in rows])
        macro_f1 = ci95([float(row["macro_f1"]) for row in rows])
        aggregates.append(
            {
                "condition": condition,
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "gate_mean": float(np.mean([row["gate"] for row in rows if row["gate"] is not None])) if any(row["gate"] is not None for row in rows) else None,
                "effective_prompt_norm_mean": float(np.mean([row["effective_prompt_norm"] for row in rows])),
                "training_loss_mean": float(np.mean([row["training_loss"] for row in rows if row["training_loss"] is not None])) if any(row["training_loss"] is not None for row in rows) else None,
            }
        )
    v2_root = {row["condition"]: row for row in aggregates}
    paired_differences: list[dict[str, Any]] = []
    for comparator in ("V1 OriginalPrompt", "NoPrompt", "V2 shuffled", "V2 random"):
        v2_name = "V2 correct"
        correct_by_seed = {row["seed"]: row for row in seed_rows if row["condition"] == v2_name}
        compare_by_seed = {row["seed"]: row for row in seed_rows if row["condition"] == comparator}
        for metric in ("accuracy", "macro_f1"):
            differences = [correct_by_seed[seed][metric] - compare_by_seed[seed][metric] for seed in SEEDS]
            paired_differences.append(
                {
                    "contrast": f"V2 correct - {comparator}",
                    "metric": metric,
                    "per_seed": differences,
                    "ci95": ci95(differences),
                }
            )

    result = {
        "runtime_seconds": time.perf_counter() - started,
        "seed_rows": seed_rows,
        "aggregates": aggregates,
        "paired_differences": paired_differences,
        "gcn_checkpoints": gcn_identifiers,
        "semantic_permutations": semantic_permutations,
        "embedding_norms": embedding_norms,
        "reference_norm": reference_norm,
        "test_classes_used": False,
    }
    write_report(result)
    print(json.dumps({"aggregates": aggregates, "paired_differences": paired_differences, "runtime_seconds": result["runtime_seconds"], "test_evaluation": "not run"}, indent=2), flush=True)


def write_report(result: dict[str, Any]) -> None:
    lines = [
        "# LLMGP-v2 Semantic Contribution Experiment",
        "",
        "## Protocol and safeguards",
        "- 3-shot; seeds 11, 23, 37; 100 training episodes per V2 arm and seed; 100 validation episodes per seed, reused identically by V1, all V2 arms, and NoPrompt.",
        "- V2 correct, shuffled, and random arms use identical episode indices, identical model initialization seeds per run, 100 training episodes, Adam, lr=0.001, and weight decay=0.0001. Validation labels were not used for training, checkpoint selection, or early stopping; final weights after the fixed budget were evaluated.",
        "- NoPrompt is a frozen-GCN reference with no trainable prompt component, so it has no training budget by definition.",
        "- The original V1 prompt checkpoint is loaded for each seed; frozen GCN checkpoints are reused. Existing checkpoint/result hashes remained unchanged.",
        "- Validation classes [3,4] were scored. Test classes [5,6] were not accessed for scoring, training, or selection.",
        "- Class-name mapping is count-verified against the authoritative LINQS `cora.content` source; no alphabetical ID assumption is used.",
        "- Runtime: {0:.2f} seconds on CPU.".format(result["runtime_seconds"]),
        "",
        "## Accuracy and Macro-F1",
        "Across-seed intervals use Student t with df=2 and are necessarily wide for only three seeds.",
        "| Condition | Accuracy mean +/- SD (95% CI) | Macro-F1 mean +/- SD (95% CI) | Mean gate | Effective prompt norm | Mean training loss |",
        "|---|---|---|---:|---:|---:|",
    ]
    for aggregate in result["aggregates"]:
        accuracy = aggregate["accuracy"]
        macro_f1 = aggregate["macro_f1"]
        accuracy_sd = float(np.std([row["accuracy"] for row in result["seed_rows"] if row["condition"] == aggregate["condition"]], ddof=1))
        f1_sd = float(np.std([row["macro_f1"] for row in result["seed_rows"] if row["condition"] == aggregate["condition"]], ddof=1))
        lines.append(
            f"| {aggregate['condition']} | {accuracy['mean']:.4f} +/- {accuracy_sd:.4f} [{accuracy['ci95_lower']:.4f}, {accuracy['ci95_upper']:.4f}] | {macro_f1['mean']:.4f} +/- {f1_sd:.4f} [{macro_f1['ci95_lower']:.4f}, {macro_f1['ci95_upper']:.4f}] | {aggregate['gate_mean'] if aggregate['gate_mean'] is not None else '--'} | {aggregate['effective_prompt_norm_mean']:.4f} | {aggregate['training_loss_mean'] if aggregate['training_loss_mean'] is not None else '--'} |"
        )
    lines.extend(["", "## Seed-wise results", "| Seed | Condition | Accuracy | Macro-F1 | Gate | Training loss | Zero-gradient episodes |", "|---:|---|---:|---:|---:|---:|---:|"])
    for row in result["seed_rows"]:
        lines.append(
            f"| {row['seed']} | {row['condition']} | {row['accuracy']:.4f} | {row['macro_f1']:.4f} | {row['gate'] if row['gate'] is not None else '--'} | {row['training_loss'] if row['training_loss'] is not None else '--'} | {row.get('zero_gradient_episodes', '--')} |"
        )
    lines.extend(["", "## Paired differences", "Differences are V2-correct minus comparator, paired within seed and identical validation episodes.", "| Comparator | Metric | Per-seed differences | Mean difference (95% CI) |", "|---|---|---|---|"])
    for row in result["paired_differences"]:
        ci = row["ci95"]
        lines.append(
            f"| {row['contrast']} | {row['metric']} | {[round(value,4) for value in row['per_seed']]} | {ci['mean']:.4f} [{ci['ci95_lower']:.4f}, {ci['ci95_upper']:.4f}] |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "Configured versus shuffled/random differences are the semantic contribution test. Small or inconsistent differences indicate no demonstrated benefit from correct assignment at this scale; they do not establish equivalence. Numeric-ID/topic-name alignment is source-verified by unique per-class count matching but does not prove that the manually authored descriptions capture every paper-topic nuance.",
            "",
            "## Exact command",
            "```powershell",
            "$py = \"c:/Users/User/Desktop/Papers to be submitted/LLMGPv0.2/implementation/.venv/Scripts/python.exe\"",
            "Set-Location \"C:/Users/User/Desktop/Papers to be submitted/LLMGPv0.2/implementation/LLMGP\"",
            "& $py experiments/run_v2_semantic_contribution.py",
            "```",
            "Validation only; no test evaluation and no full experiment.",
        ]
    )
    (ROOT / "reports" / "v2_semantic_contribution.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
