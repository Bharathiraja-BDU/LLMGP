from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
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
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
INITIAL_GATE = 0.1
TRAIN_CLASSES = tuple(CLASS_SPLIT["train"])
VALIDATION_CLASSES = tuple(CLASS_SPLIT["validation"])
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_frozen_inputs() -> tuple[dict[str, Any], np.ndarray, torch.Tensor, SciBERTEncoder]:
    info = verify_cora_dataset()
    labels = info["y"].cpu().numpy()
    encoder = SciBERTEncoder(model_name="allenai/scibert_scivocab_uncased", device=str(DEVICE))
    permitted_ids = (*TRAIN_CLASSES, *VALIDATION_CLASSES)
    encoded = encoder.encode_class_descriptions(
        [CLASS_DESCRIPTIONS[class_id] for class_id in permitted_ids],
        batch_size=len(permitted_ids),
    )
    if any(parameter.requires_grad for parameter in encoder.model.parameters()):
        raise AssertionError("SciBERT must be frozen")
    embeddings = torch.zeros((7, 768), dtype=encoded.dtype)
    for offset, class_id in enumerate(permitted_ids):
        embeddings[class_id] = encoded[offset]
    return {**info, "labels_array": labels}, labels, embeddings.to(DEVICE), encoder


def load_frozen_gcn(seed: int) -> tuple[GCNBackbone, dict[str, Any]]:
    path = ROOT / "results" / "checkpoints" / f"seed_{seed}_frozen_gcn.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    metadata = checkpoint["pretraining"]
    if int(metadata["seed"]) != seed:
        raise AssertionError(f"Frozen GCN seed mismatch: expected {seed}, found {metadata['seed']}")
    if metadata["objective"] != "masked_feature_reconstruction_mse":
        raise AssertionError("Expected existing self-supervised v1 GCN checkpoint")
    backbone = GCNBackbone(1433, 256, 256)
    backbone.load_state_dict(checkpoint["model_state_dict"])
    backbone.to(DEVICE).eval()
    for parameter in backbone.parameters():
        parameter.requires_grad = False
        parameter.grad = None
    return backbone, {
        "checkpoint": str(path.relative_to(ROOT)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "seed": seed,
        "objective": metadata["objective"],
    }


def make_semantic_embeddings(embeddings: torch.Tensor, condition: str, seed: int) -> tuple[torch.Tensor, list[int] | None]:
    output = embeddings.clone()
    if condition == "configured":
        return output, None
    if condition != "shuffled":
        raise ValueError(f"Unsupported semantic condition: {condition}")
    generator = torch.Generator(device=embeddings.device).manual_seed(PROTOCOL_SEED + seed * 91)
    permutation = torch.randperm(len(TRAIN_CLASSES), generator=generator).tolist()
    if permutation == list(range(len(TRAIN_CLASSES))):
        permutation = permutation[1:] + permutation[:1]
    source_ids = [TRAIN_CLASSES[index] for index in permutation]
    output[list(TRAIN_CLASSES)] = embeddings[source_ids]
    return output, source_ids


def episode_sets(labels: np.ndarray, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train_base = PROTOCOL_SEED * 10_000_000 + seed * 100_000 + SHOT * 1_000
    validation_base = train_base + 50_000
    train = [
        sample_episode(labels, "train", SHOT, train_base + episode_index)
        for episode_index in range(TRAIN_EPISODES)
    ]
    validation = [
        sample_episode(labels, "validation", SHOT, validation_base + episode_index)
        for episode_index in range(VALIDATION_EPISODES)
    ]
    if any(not set(ep["support_idx"].tolist()).isdisjoint(ep["query_idx"].tolist()) for ep in [*train, *validation]):
        raise AssertionError("Support/query overlap")
    if any(not set(ep["class_ids"].tolist()).issubset(TRAIN_CLASSES) for ep in train):
        raise AssertionError("Training episode contains a non-training class")
    if any(ep["class_ids"].tolist() != list(VALIDATION_CLASSES) for ep in validation):
        raise AssertionError("Validation episodes must use classes [3,4]")
    return train, validation


def train_v2(
    seed: int,
    semantic_condition: str,
    train_episodes: list[dict[str, Any]],
    info: dict[str, Any],
    class_embeddings: torch.Tensor,
    backbone: GCNBackbone,
    reference_norm: float,
) -> tuple[NormalizedGatedPrompt, dict[str, Any]]:
    torch.manual_seed(seed * 1_000_000 + SHOT * 10_000 + (0 if semantic_condition == "configured" else 1))
    model = NormalizedGatedPrompt(reference_node_norm=reference_norm, initial_gate=INITIAL_GATE).to(DEVICE)
    run_embeddings, semantic_assignment = make_semantic_embeddings(class_embeddings, semantic_condition, seed)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    x = info["x"].to(DEVICE)
    edge_index = info["edge_index"].to(DEVICE)
    losses: list[float] = []
    gradient_norms: list[float] = []
    skipped_zero_gradient_episodes = 0

    for episode in train_episodes:
        loss, _, _, _ = episode_forward(episode, x, edge_index, run_embeddings, model, backbone)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if any(parameter.grad is not None for parameter in backbone.parameters()):
            raise AssertionError("Frozen GCN received gradients")
        if not validate_trainable_gradients(model):
            skipped_zero_gradient_episodes += 1
            continue
        gradient_norms.append(
            float(torch.sqrt(sum(parameter.grad.detach().square().sum() for parameter in model.parameters())).item())
        )
        optimizer.step()
        losses.append(float(loss.detach().item()))
    if not losses:
        raise RuntimeError(f"No usable training gradients for seed={seed}, semantic={semantic_condition}")

    model.eval()
    return model, {
        "training_episodes_requested": len(train_episodes),
        "training_episodes_updated": len(losses),
        "zero_gradient_episodes_skipped": skipped_zero_gradient_episodes,
        "training_loss_mean": float(np.mean(losses)),
        "gradient_norm_mean": float(np.mean(gradient_norms)),
        "gate": float(model.gate.item()),
        "semantic_assignment": semantic_assignment,
    }


def confidence_summary(seed_values: list[float]) -> dict[str, float]:
    mean = float(np.mean(seed_values))
    sd = float(np.std(seed_values, ddof=1))
    margin = float(t.ppf(0.975, len(seed_values) - 1) * sd / np.sqrt(len(seed_values)))
    return {"mean": mean, "sd": sd, "ci95_lower": mean - margin, "ci95_upper": mean + margin}


def main() -> None:
    start = time.perf_counter()
    torch.set_num_threads(1)
    v1_before = snapshot_v1_artifacts(ROOT)
    info, labels, class_embeddings, encoder = load_frozen_inputs()
    train_mask = np.isin(labels, np.asarray(TRAIN_CLASSES))
    reference_norm = float(info["x"][torch.from_numpy(train_mask)].norm(dim=-1).mean().item())
    if not np.isfinite(reference_norm) or reference_norm <= 0:
        raise AssertionError("Invalid training-node feature norm")

    seed_results: list[dict[str, Any]] = []
    gcn_metadata: dict[str, Any] = {}
    v1_checkpoint_metadata: dict[str, Any] = {}

    for seed in SEEDS:
        backbone, gcn_detail = load_frozen_gcn(seed)
        gcn_metadata[seed] = gcn_detail
        train_episodes, validation_episodes = episode_sets(labels, seed)
        v1_path = ROOT / "results" / "checkpoints" / f"shot_{SHOT}_seed_{seed}_prompt.pt"
        v1_checkpoint = torch.load(v1_path, map_location="cpu", weights_only=False)
        metadata = v1_checkpoint["metadata"]
        if int(metadata["seed"]) != seed or int(metadata["shot"]) != SHOT:
            raise AssertionError("Saved v1 prompt checkpoint seed/shot mismatch")
        associated_gcn = str(metadata["backbone_checkpoint"]).replace("\\", "/")
        if not associated_gcn.endswith(f"seed_{seed}_frozen_gcn.pt"):
            raise AssertionError("Saved v1 prompt references an unexpected GCN")
        v1_checkpoint_metadata[seed] = {
            "checkpoint": str(v1_path.relative_to(ROOT)),
            "best_epoch": int(metadata["best_epoch"]),
            "sha256": hashlib.sha256(v1_path.read_bytes()).hexdigest(),
        }
        v1_model = PromptBuilder().to(DEVICE)
        v1_model.load_state_dict(v1_checkpoint["prompt_state_dict"])
        v1_model.eval()

        correct_model, correct_train = train_v2(
            seed, "configured", train_episodes, info, class_embeddings, backbone, reference_norm
        )
        shuffled_model, shuffled_train = train_v2(
            seed, "shuffled", train_episodes, info, class_embeddings, backbone, reference_norm
        )
        x = info["x"].to(DEVICE)
        edge_index = info["edge_index"].to(DEVICE)
        results = {
            "V1 saved OriginalPrompt": evaluate_episodes(
                validation_episodes, x, edge_index, class_embeddings, v1_model, backbone, "connected"
            ),
            "V2 configured descriptions": evaluate_episodes(
                validation_episodes, x, edge_index, class_embeddings, correct_model, backbone, "connected"
            ),
            "V2 shuffled train descriptions": evaluate_episodes(
                validation_episodes,
                x,
                edge_index,
                make_semantic_embeddings(class_embeddings, "shuffled", seed)[0],
                shuffled_model,
                backbone,
                "connected",
            ),
            "NoPrompt": evaluate_episodes(
                validation_episodes, x, edge_index, class_embeddings, None, backbone, "none"
            ),
        }
        for result_name, metrics in results.items():
            seed_results.append(
                {
                    "seed": seed,
                    "shot": SHOT,
                    "validation_episodes": VALIDATION_EPISODES,
                    "model": result_name,
                    "validation_accuracy": metrics["accuracy"],
                    "validation_macro_f1": metrics["macro_f1"],
                    "validation_loss": metrics["loss"],
                    "effective_prompt_norm": metrics["effective_prompt_norm"],
                    "gate": metrics["gate"],
                    "training_loss": (
                        correct_train["training_loss_mean"]
                        if result_name == "V2 configured descriptions"
                        else shuffled_train["training_loss_mean"]
                        if result_name == "V2 shuffled train descriptions"
                        else None
                    ),
                    "training_updates": (
                        correct_train["training_episodes_updated"]
                        if result_name == "V2 configured descriptions"
                        else shuffled_train["training_episodes_updated"]
                        if result_name == "V2 shuffled train descriptions"
                        else None
                    ),
                    "zero_gradient_skips": (
                        correct_train["zero_gradient_episodes_skipped"]
                        if result_name == "V2 configured descriptions"
                        else shuffled_train["zero_gradient_episodes_skipped"]
                        if result_name == "V2 shuffled train descriptions"
                        else None
                    ),
                }
            )
        del correct_model, shuffled_model, v1_model, backbone

    v1_after = snapshot_v1_artifacts(ROOT)
    if v1_before != v1_after:
        raise AssertionError("Existing v1 artifacts changed during extended validation")

    metric_names = ("validation_accuracy", "validation_macro_f1")
    aggregate: list[dict[str, Any]] = []
    model_names = [
        "V1 saved OriginalPrompt",
        "V2 configured descriptions",
        "V2 shuffled train descriptions",
        "NoPrompt",
    ]
    for model_name in model_names:
        rows = [row for row in seed_results if row["model"] == model_name]
        item: dict[str, Any] = {"model": model_name, "shot": SHOT, "seed_count": len(SEEDS)}
        for metric_name in metric_names:
            stats = confidence_summary([float(row[metric_name]) for row in rows])
            for stat_name, value in stats.items():
                item[f"{metric_name}_{stat_name}"] = value
        if model_name.startswith("V2"):
            item["gate_mean"] = float(np.mean([float(row["gate"]) for row in rows]))
            item["effective_prompt_norm_mean"] = float(np.mean([float(row["effective_prompt_norm"]) for row in rows]))
        else:
            item["gate_mean"] = None
            item["effective_prompt_norm_mean"] = float(np.mean([float(row["effective_prompt_norm"]) for row in rows]))
        aggregate.append(item)

    result = {
        "status": "completed_validation_only",
        "runtime_seconds": time.perf_counter() - start,
        "device": str(DEVICE),
        "shot": SHOT,
        "seeds": list(SEEDS),
        "train_episodes_per_v2_arm_per_seed": TRAIN_EPISODES,
        "validation_episodes_per_seed": VALIDATION_EPISODES,
        "validation_classes": list(VALIDATION_CLASSES),
        "test_classes_used": False,
        "reference_node_norm_train_only": reference_norm,
        "class_embeddings_encoded": list((*TRAIN_CLASSES, *VALIDATION_CLASSES)),
        "topic_alignment_verified": False,
        "sci_bert_identifier": "allenai/scibert_scivocab_uncased",
        "sci_bert_frozen": all(not parameter.requires_grad for parameter in encoder.model.parameters()),
        "gcn_checkpoints": gcn_metadata,
        "v1_prompt_checkpoints": v1_checkpoint_metadata,
        "seed_results": seed_results,
        "aggregate_results": aggregate,
        "v1_artifacts_unchanged": True,
    }
    report_path = ROOT / "reports" / "v2_extended_validation.md"
    write_report(result, report_path)
    print(json.dumps({"aggregate_results": aggregate, "runtime_seconds": result["runtime_seconds"], "test_evaluation": "not run"}, indent=2), flush=True)


def write_report(result: dict[str, Any], path: Path) -> None:
    lines = [
        "# LLMGP-v2 Extended Validation",
        "",
        "## Scope and safeguards",
        "- Validation only: 3-shot, seeds 11/23/37, 100 V2 training episodes for each semantic arm per seed, and 100 identical validation episodes per seed shared across V1, V2-correct, V2-shuffled, and NoPrompt.",
        "- Test classes [5,6] were not scored, used for model selection, or used to generate class-description embeddings.",
        "- Existing v1 checkpoints/results were hash-compared before/after and remained unchanged.",
        f"- Runtime: {result['runtime_seconds']:.2f} seconds on {result['device']}.",
        "",
        "## Reused frozen models and semantic caveat",
        "- Frozen GCN checkpoints are the existing seed-specific v1 self-supervised checkpoints; SHA-256 identifiers are listed below.",
        "- Frozen SciBERT identifier: `allenai/scibert_scivocab_uncased` (pretrained Hugging Face model/cache; this project has no separate SciBERT checkpoint file).",
        "- Only class IDs [0,1,2,3,4] were encoded. Local Cora Planetoid labels are numeric/one-hot and do not include aligned topic names; configured class-description alignment is therefore not independently verified.",
        "- The shuffled arm permutes descriptions only among training class slots [0,1,2]; validation class descriptions remain unchanged.",
        "",
        "## Validation results",
        "Values are means over 100 episode-level metrics per seed. Across-seed confidence intervals use a two-sided Student t interval with df=2; with three seeds these intervals are imprecise.",
        "| Model | Accuracy mean +/- SD (95% CI) | Macro-F1 mean +/- SD (95% CI) | Effective prompt norm | Gate |",
        "|---|---|---|---:|---:|",
    ]
    for row in result["aggregate_results"]:
        acc_seed = [float(item["validation_accuracy"]) for item in result["seed_results"] if item["model"] == row["model"]]
        f1_seed = [float(item["validation_macro_f1"]) for item in result["seed_results"] if item["model"] == row["model"]]
        acc_sd = float(np.std(acc_seed, ddof=1))
        f1_sd = float(np.std(f1_seed, ddof=1))
        lines.append(
            f"| {row['model']} | {row['validation_accuracy_mean']:.4f} +/- {acc_sd:.4f} [{row['validation_accuracy_ci95_lower']:.4f}, {row['validation_accuracy_ci95_upper']:.4f}] | {row['validation_macro_f1_mean']:.4f} +/- {f1_sd:.4f} [{row['validation_macro_f1_ci95_lower']:.4f}, {row['validation_macro_f1_ci95_upper']:.4f}] | {row['effective_prompt_norm_mean']:.4f} | {row['gate_mean'] if row['gate_mean'] is not None else '--'} |"
        )
    lines.extend(["", "### Seed-wise results", "| Seed | Model | Accuracy | Macro-F1 | Gate | Train loss (if trained) | Zero-gradient skips |", "|---:|---|---:|---:|---:|---:|---:|"])
    for row in result["seed_results"]:
        lines.append(
            f"| {row['seed']} | {row['model']} | {row['validation_accuracy']:.4f} | {row['validation_macro_f1']:.4f} | {row['gate'] if row['gate'] is not None else '--'} | {row['training_loss'] if row['training_loss'] is not None else '--'} | {row['zero_gradient_skips'] if row['zero_gradient_skips'] is not None else '--'} |"
        )
    lines.extend(["", "### Checkpoint identifiers", "| Seed | Frozen GCN checkpoint | SHA-256 | Saved V1 prompt checkpoint | V1 best epoch |", "|---:|---|---|---|---:|"])
    for seed in result["seeds"]:
        gcn = result["gcn_checkpoints"][seed]
        prompt = result["v1_prompt_checkpoints"][seed]
        lines.append(f"| {seed} | {gcn['checkpoint']} | {gcn['sha256']} | {prompt['checkpoint']} | {prompt['best_epoch']} |")
    lines.extend(
        [
            "",
            "## Consistency decision",
            "V2 is considered consistently better than a comparator only if it exceeds that comparator for both accuracy and Macro-F1 on every seed in this MVP. This is descriptive validation evidence only.",
        ]
    )
    for comparator in ("V1 saved OriginalPrompt", "NoPrompt"):
        v2_rows = [row for row in result["seed_results"] if row["model"] == "V2 configured descriptions"]
        baseline_rows = [row for row in result["seed_results"] if row["model"] == comparator]
        joined = {row["seed"]: row for row in baseline_rows}
        wins = [
            row["validation_accuracy"] > joined[row["seed"]]["validation_accuracy"]
            and row["validation_macro_f1"] > joined[row["seed"]]["validation_macro_f1"]
            for row in v2_rows
        ]
        lines.append(f"- V2 versus {comparator}: {sum(wins)}/{len(wins)} seeds beat it on both metrics; consistent improvement: **{all(wins)}**.")
    shuffled = {row["seed"]: row for row in result["seed_results"] if row["model"] == "V2 shuffled train descriptions"}
    configured = {row["seed"]: row for row in result["seed_results"] if row["model"] == "V2 configured descriptions"}
    semantic_wins = sum(
        configured[seed]["validation_macro_f1"] > shuffled[seed]["validation_macro_f1"]
        and configured[seed]["validation_accuracy"] > shuffled[seed]["validation_accuracy"]
        for seed in result["seeds"]
    )
    lines.append(f"- Configured versus shuffled descriptions: configured wins on both metrics for {semantic_wins}/{len(result['seeds'])} seeds. Topic-name alignment is unverified, so no claim of correctly aligned semantic benefit is made.")
    lines.extend(
        [
            "",
            "## Command and stop point",
            "```powershell",
            "$py = \"c:/Users/User/Desktop/Papers to be submitted/LLMGPv0.2/implementation/.venv/Scripts/python.exe\"",
            "Set-Location \"C:/Users/User/Desktop/Papers to be submitted/LLMGPv0.2/implementation/LLMGP\"",
            "& $py experiments/run_v2_extended_validation.py",
            "```",
            "Validation is complete. No test evaluation was run, and no additional seeds or shot settings were started.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
