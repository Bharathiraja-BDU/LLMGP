from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import verify_cora_dataset
from src.gnn_backbone import GCNBackbone
from src.llm_encoder import SciBERTEncoder
from src.revised_protocol import (
    CLASS_DESCRIPTIONS,
    CLASS_SPLIT,
    PROTOCOL_SEED,
    RUN_SEEDS,
    labels_visible_in_split,
    validate_protocol,
)
from src.v2.prompt_models import NormalizedGatedPrompt
from src.v2.validation import (
    MODEL_A,
    MODEL_B,
    MODEL_C,
    SEMANTIC_VARIANTS,
    build_prompt_model,
    class_embedding_variant,
    episode_forward,
    evaluate_episodes,
    make_fixed_episode_sets,
    sample_episode,
    snapshot_v1_artifacts,
    validate_trainable_gradients,
    validation_checkpoint_key,
    verify_frozen_backbone,
)


SHOTS = (1, 3, 5, 10)
EPOCHS = 50
EPISODES_PER_EPOCH = 100
PATIENCE = 10
VALIDATION_EPISODES = 100
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
INITIAL_GATE = 0.1
SEMANTIC_ARMS = ((MODEL_A, "configured"), (MODEL_B, "configured")) + tuple(
    (MODEL_C, variant) for variant in SEMANTIC_VARIANTS
)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
V2_ROOT = ROOT / "results" / "v2"
CHECKPOINTS = V2_ROOT / "checkpoints"
LOGS = V2_ROOT / "training_logs"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_frozen_gcn(seed: int) -> tuple[GCNBackbone, dict[str, Any]]:
    path = ROOT / "results" / "checkpoints" / f"seed_{seed}_frozen_gcn.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    metadata = checkpoint["pretraining"]
    if int(metadata["seed"]) != seed:
        raise AssertionError(f"V1 GCN seed mismatch: expected {seed}, got {metadata['seed']}")
    if metadata["objective"] != "masked_feature_reconstruction_mse":
        raise AssertionError("Expected existing v1 masked-feature self-supervised GCN checkpoint")
    backbone = GCNBackbone(in_channels=1433, hidden_channels=256, out_channels=256)
    backbone.load_state_dict(checkpoint["model_state_dict"])
    backbone.to(DEVICE)
    verify_frozen_backbone(backbone)
    return backbone, {
        "identifier": f"seed_{seed}_frozen_gcn.pt",
        "sha256": __import__("hashlib").sha256(path.read_bytes()).hexdigest(),
        "seed": seed,
        "objective": metadata["objective"],
        "pretraining_epochs": metadata["epochs"],
        "pretraining_final_loss": metadata["final_loss"],
    }


def prepare_permitted_embeddings(encoder: SciBERTEncoder) -> tuple[torch.Tensor, dict[str, Any]]:
    permitted_ids = (*CLASS_SPLIT["train"], *CLASS_SPLIT["validation"])
    descriptions = [CLASS_DESCRIPTIONS[class_id] for class_id in permitted_ids]
    encoded = encoder.encode_class_descriptions(descriptions, batch_size=len(descriptions))
    if encoded.shape != (len(permitted_ids), 768):
        raise AssertionError("Unexpected SciBERT embedding shape")
    if any(parameter.requires_grad for parameter in encoder.model.parameters()):
        raise AssertionError("SciBERT must remain frozen")
    class_embeddings = torch.zeros((7, 768), dtype=encoded.dtype)
    for offset, class_id in enumerate(permitted_ids):
        class_embeddings[class_id] = encoded[offset]
    return class_embeddings.to(DEVICE), {
        "embedded_class_ids": list(permitted_ids),
        "test_class_embeddings_constructed": False,
        "topic_id_alignment_verified": False,
    }


def training_episode_seed(seed: int, shot: int, epoch: int, episode_index: int) -> int:
    return PROTOCOL_SEED * 10_000_000 + seed * 1_000_000 + shot * 100_000 + epoch * EPISODES_PER_EPOCH + episode_index


def train_one(
    *,
    model_name: str,
    semantic_variant: str,
    seed: int,
    shot: int,
    reference_norm: float,
    data: Any,
    labels: np.ndarray,
    class_embeddings: torch.Tensor,
    backbone: GCNBackbone,
    validation_episodes: list[dict[str, Any]],
) -> tuple[torch.nn.Module, dict[str, Any], list[dict[str, Any]], Path, dict[str, Any]]:
    model_seed = seed * 1_000_000 + shot * 10_000 + (0 if semantic_variant == "configured" else (1 if semantic_variant == "shuffled" else 2))
    torch.manual_seed(model_seed)
    prompt_model = build_prompt_model(model_name, reference_norm).to(DEVICE)
    run_embeddings, permutation = class_embedding_variant(
        class_embeddings,
        semantic_variant,
        seed=PROTOCOL_SEED + seed * 100 + shot,
    )
    train_class_ids = set(CLASS_SPLIT["train"])
    if any(float(run_embeddings[class_id].norm().item()) == 0.0 for class_id in train_class_ids):
        raise AssertionError("Missing semantic embedding for a training class")

    optimizer = torch.optim.Adam(prompt_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    backbone.eval()
    verify_frozen_backbone(backbone)
    x = data.x.to(DEVICE)
    edge_index = data.edge_index.to(DEVICE)
    history: list[dict[str, Any]] = []
    best_key: tuple[float, float, float] | None = None
    best_state: dict[str, torch.Tensor] | None = None
    best_metrics: dict[str, Any] | None = None
    best_epoch = 0
    stale_epochs = 0

    for epoch in range(EPOCHS):
        prompt_model.train()
        epoch_losses: list[float] = []
        zero_signal_episodes = 0
        for episode_index in range(EPISODES_PER_EPOCH):
            episode = sample_episode(
                labels,
                "train",
                shot,
                training_episode_seed(seed, shot, epoch, episode_index),
            )
            loss, _, _, prompt_features = episode_forward(
                episode,
                x,
                edge_index,
                run_embeddings,
                prompt_model,
                backbone,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if any(parameter.grad is not None for parameter in backbone.parameters()):
                raise AssertionError("Frozen GCN received gradients during training")
            if not validate_trainable_gradients(prompt_model):
                zero_signal_episodes += 1
                continue
            optimizer.step()
            epoch_losses.append(float(loss.detach().item()))
        if not epoch_losses:
            raise RuntimeError("Every training episode had zero prompt gradient signal in this epoch")

        prompt_model.eval()
        val_metrics = evaluate_episodes(
            validation_episodes,
            x,
            edge_index,
            run_embeddings,
            prompt_model,
            backbone,
            prompt_mode="connected",
        )
        current_key = validation_checkpoint_key(val_metrics)
        gate_value = float(prompt_model.gate.item()) if isinstance(prompt_model, NormalizedGatedPrompt) else None
        effective_norm = val_metrics["effective_prompt_norm"]
        history_row = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(epoch_losses)),
            "validation_loss": val_metrics["loss"],
            "validation_accuracy": val_metrics["accuracy"],
            "validation_macro_f1": val_metrics["macro_f1"],
            "effective_prompt_norm": effective_norm,
            "gate": gate_value,
            "zero_signal_episodes": zero_signal_episodes,
        }
        history.append(history_row)
        if best_key is None or current_key > best_key:
            best_key = current_key
            best_epoch = epoch + 1
            best_metrics = val_metrics
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in prompt_model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= PATIENCE:
                break

    if best_state is None or best_metrics is None:
        raise RuntimeError("No validation checkpoint selected")
    prompt_model.load_state_dict(best_state)
    prompt_model.eval()
    checkpoint_path = CHECKPOINTS / f"{model_name.lower()}_{semantic_variant}_shot_{shot}_seed_{seed}.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best_state,
            "metadata": {
                "model": model_name,
                "semantic_variant": semantic_variant,
                "seed": seed,
                "shot": shot,
                "best_epoch": best_epoch,
                "best_validation_macro_f1": best_metrics["macro_f1"],
                "best_validation_accuracy": best_metrics["accuracy"],
                "reference_node_norm_train_only": reference_norm,
                "initial_gate": INITIAL_GATE if isinstance(prompt_model, NormalizedGatedPrompt) else None,
                "final_gate": float(prompt_model.gate.item()) if isinstance(prompt_model, NormalizedGatedPrompt) else None,
                "semantic_permutation": permutation,
                "v1_backbone_checkpoint": f"results/checkpoints/seed_{seed}_frozen_gcn.pt",
            },
        },
        checkpoint_path,
    )
    log_path = LOGS / f"{model_name.lower()}_{semantic_variant}_shot_{shot}_seed_{seed}.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)

    train_summary = {
        "model": model_name,
        "semantic_variant": semantic_variant,
        "seed": seed,
        "shot": shot,
        "reference_node_norm": reference_norm,
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "zero_signal_episodes": int(sum(row["zero_signal_episodes"] for row in history)),
        "train_loss_best_epoch": history[best_epoch - 1]["train_loss"],
        "validation_metrics": best_metrics,
        "gate": float(prompt_model.gate.item()) if isinstance(prompt_model, NormalizedGatedPrompt) else None,
        "effective_prompt_norm": best_metrics["effective_prompt_norm"],
        "checkpoint": str(checkpoint_path.relative_to(ROOT)),
        "training_log": str(log_path.relative_to(ROOT)),
    }
    return prompt_model, train_summary, history, checkpoint_path, {"permutation": permutation, "embedding_ids": [0, 1, 2, 3, 4]}


def confidence_interval(values: list[float]) -> tuple[float, float, float, float]:
    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1))
    margin = float(t.ppf(0.975, df=len(values) - 1) * sd / np.sqrt(len(values)))
    return mean, sd, mean - margin, mean + margin


def summarize_seed_metrics(seed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    aggregate: list[dict[str, Any]] = []
    keys = sorted({(row["model"], row["semantic_variant"], row["shot"]) for row in seed_rows})
    for model_name, semantic_variant, shot in keys:
        rows = [
            row for row in seed_rows
            if row["model"] == model_name and row["semantic_variant"] == semantic_variant and row["shot"] == shot
        ]
        accuracy_mean, accuracy_sd, accuracy_low, accuracy_high = confidence_interval(
            [float(row["validation_accuracy"]) for row in rows]
        )
        f1_mean, f1_sd, f1_low, f1_high = confidence_interval(
            [float(row["validation_macro_f1"]) for row in rows]
        )
        aggregate.append(
            {
                "model": model_name,
                "semantic_variant": semantic_variant,
                "shot": shot,
                "seed_count": len(rows),
                "validation_episodes_per_seed": VALIDATION_EPISODES,
                "accuracy_mean": accuracy_mean,
                "accuracy_sd": accuracy_sd,
                "accuracy_ci95_lower": accuracy_low,
                "accuracy_ci95_upper": accuracy_high,
                "macro_f1_mean": f1_mean,
                "macro_f1_sd": f1_sd,
                "macro_f1_ci95_lower": f1_low,
                "macro_f1_ci95_upper": f1_high,
                "best_epoch_mean": (
                    float(np.mean([row["best_epoch"] for row in rows if row["best_epoch"] is not None]))
                    if any(row["best_epoch"] is not None for row in rows)
                    else None
                ),
                "training_loss_mean": (
                    float(np.mean([row["train_loss_best_epoch"] for row in rows if row["train_loss_best_epoch"] is not None]))
                    if any(row["train_loss_best_epoch"] is not None for row in rows)
                    else None
                ),
                "zero_signal_episodes_mean": float(np.mean([row["zero_signal_episodes"] for row in rows])),
                "effective_prompt_norm_mean": float(np.mean([row["effective_prompt_norm"] for row in rows])),
                "gate_mean": float(np.mean([row["gate"] for row in rows if row["gate"] is not None])) if any(row["gate"] is not None for row in rows) else None,
            }
        )
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LLMGP-v2 validation only; never evaluates test classes.")
    parser.add_argument("--smoke-only", action="store_true", help="Run the critical v2 tests and one-epoch gradient smoke check.")
    args = parser.parse_args()
    started = time.perf_counter()
    torch.set_num_threads(1)
    V2_ROOT.mkdir(parents=True, exist_ok=True)

    before_manifest = snapshot_v1_artifacts(ROOT)
    before_path = V2_ROOT / "v1_artifact_manifest_before.json"
    before_path.write_text(json.dumps(before_manifest, indent=2), encoding="utf-8")
    subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "tests/v2/test_v2_validation.py"],
        cwd=ROOT,
        check=True,
    )

    info = verify_cora_dataset()
    labels = info["y"].cpu().numpy()
    validate_protocol(labels)
    visible_train_labels = labels_visible_in_split(labels, CLASS_SPLIT["train"])
    visible_validation_labels = labels_visible_in_split(labels, CLASS_SPLIT["validation"])
    assert set(np.unique(visible_train_labels[visible_train_labels >= 0])).issubset(CLASS_SPLIT["train"])
    assert set(np.unique(visible_validation_labels[visible_validation_labels >= 0])).issubset(CLASS_SPLIT["validation"])
    encoder = SciBERTEncoder(model_name="allenai/scibert_scivocab_uncased", device=str(DEVICE))
    class_embeddings, embedding_info = prepare_permitted_embeddings(encoder)
    if args.smoke_only:
        smoke_model = build_prompt_model(MODEL_C, reference_node_norm=float(info["x"][labels_visible_in_split(labels, CLASS_SPLIT["train"]) >= 0].norm(dim=-1).mean()))
        smoke_backbone, _ = load_frozen_gcn(RUN_SEEDS[0])
        sample = sample_episode(labels, "train", 1, 501)
        loss, _, _, _ = episode_forward(
            sample,
            info["x"].to(DEVICE),
            info["edge_index"].to(DEVICE),
            class_embeddings,
            smoke_model.to(DEVICE),
            smoke_backbone,
        )
        loss.backward()
        assert smoke_model.projection.weight.grad is not None and smoke_model.projection.weight.grad.norm().item() > 0
        assert smoke_model.gate_logit.grad is not None and smoke_model.gate_logit.grad.abs().item() > 0
        assert all(parameter.grad is None for parameter in smoke_backbone.parameters())
        after_manifest = snapshot_v1_artifacts(ROOT)
        if before_manifest != after_manifest:
            raise AssertionError("Existing v1 artifacts changed during smoke verification")
        print(json.dumps({"status": "smoke_passed", "test_evaluation": "not_run", "embedded_class_ids": embedding_info["embedded_class_ids"]}, indent=2))
        return

    reference_mask = np.isin(labels, CLASS_SPLIT["train"])
    reference_norm = float(info["x"][torch.from_numpy(reference_mask)].norm(dim=-1).mean().item())
    if not np.isfinite(reference_norm) or reference_norm <= 0:
        raise AssertionError("Invalid train-node reference feature norm")
    validation_sets = make_fixed_episode_sets(labels, SHOTS, VALIDATION_EPISODES, RUN_SEEDS)
    episode_manifest = {
        f"validation|shot={shot}|seed={seed}": [
            {
                key: value.tolist() if isinstance(value, np.ndarray) else value
                for key, value in episode.items()
            }
            for episode in validation_sets[("validation", shot, seed)]
        ]
        for shot in SHOTS
        for seed in RUN_SEEDS
    }
    (V2_ROOT / "validation_episode_manifest.json").write_text(json.dumps(episode_manifest), encoding="utf-8")

    seed_rows: list[dict[str, Any]] = []
    training_run_details: list[dict[str, Any]] = []
    gcn_identifiers: dict[int, dict[str, Any]] = {}
    for seed in RUN_SEEDS:
        backbone, gcn_info = load_frozen_gcn(seed)
        gcn_identifiers[seed] = gcn_info
        for shot in SHOTS:
            episodes = validation_sets[("validation", shot, seed)]
            for model_name, semantic_variant in SEMANTIC_ARMS:
                prompt_model, summary, _, _, semantic_info = train_one(
                    model_name=model_name,
                    semantic_variant=semantic_variant,
                    seed=seed,
                    shot=shot,
                    reference_norm=reference_norm,
                    data=info["data"],
                    labels=labels,
                    class_embeddings=class_embeddings,
                    backbone=backbone,
                    validation_episodes=episodes,
                )
                seed_rows.append(
                    {
                        "model": model_name,
                        "semantic_variant": semantic_variant,
                        "seed": seed,
                        "shot": shot,
                        "validation_accuracy": summary["validation_metrics"]["accuracy"],
                        "validation_macro_f1": summary["validation_metrics"]["macro_f1"],
                        "validation_loss": summary["validation_metrics"]["loss"],
                        "best_epoch": summary["best_epoch"],
                        "epochs_completed": summary["epochs_completed"],
                        "train_loss_best_epoch": summary["train_loss_best_epoch"],
                        "zero_signal_episodes": summary["zero_signal_episodes"],
                        "reference_node_norm_train_only": reference_norm,
                        "effective_prompt_norm": summary["effective_prompt_norm"],
                        "gate": summary["gate"],
                        "gcn_checkpoint": gcn_info["identifier"],
                        "gcn_sha256": gcn_info["sha256"],
                        "prompt_checkpoint": summary["checkpoint"],
                    }
                )
                training_run_details.append({**summary, **semantic_info, "gcn_checkpoint": gcn_info})
                print(
                    f"validated model={model_name} semantic={semantic_variant} seed={seed} shot={shot} "
                    f"epoch={summary['best_epoch']} f1={summary['validation_metrics']['macro_f1']:.4f}",
                    flush=True,
                )
                del prompt_model
        del backbone

    # Frozen-backbone reference controls use the same saved validation episodes.
    baseline_rows: list[dict[str, Any]] = []
    for seed in RUN_SEEDS:
        backbone, gcn_info = load_frozen_gcn(seed)
        for shot in SHOTS:
            episodes = validation_sets[("validation", shot, seed)]
            for variant, prompt_mode in (("NoPrompt", "none"), ("ZeroPrompt", "zero")):
                metrics = evaluate_episodes(
                    episodes,
                    info["x"].to(DEVICE),
                    info["edge_index"].to(DEVICE),
                    class_embeddings,
                    prompt_model=None,
                    backbone=backbone,
                    prompt_mode=prompt_mode,
                )
                baseline_rows.append(
                    {
                        "model": variant,
                        "semantic_variant": "none",
                        "seed": seed,
                        "shot": shot,
                        "validation_accuracy": metrics["accuracy"],
                        "validation_macro_f1": metrics["macro_f1"],
                        "validation_loss": metrics["loss"],
                        "best_epoch": None,
                        "epochs_completed": 0,
                        "train_loss_best_epoch": None,
                        "zero_signal_episodes": 0,
                        "reference_node_norm_train_only": reference_norm,
                        "effective_prompt_norm": 0.0,
                        "gate": None,
                        "gcn_checkpoint": gcn_info["identifier"],
                        "gcn_sha256": gcn_info["sha256"],
                        "prompt_checkpoint": None,
                    }
                )
        del backbone

    all_seed_rows = seed_rows + baseline_rows
    aggregate_rows = summarize_seed_metrics(all_seed_rows)
    write_csv(V2_ROOT / "seed_wise_validation.csv", all_seed_rows)
    write_csv(V2_ROOT / "validation_results.csv", aggregate_rows)
    gate_rows = [
        {
            "model": row["model"],
            "semantic_variant": row["semantic_variant"],
            "seed": row["seed"],
            "shot": row["shot"],
            "gate": row["gate"],
            "effective_prompt_norm": row["effective_prompt_norm"],
            "best_epoch": row["best_epoch"],
        }
        for row in seed_rows
        if row["model"] == MODEL_C
    ]
    write_csv(V2_ROOT / "gate_statistics.csv", gate_rows)
    semantic_rows = [
        row for row in aggregate_rows if row["model"] == MODEL_C
    ]
    write_csv(V2_ROOT / "semantic_ablation.csv", semantic_rows)

    # Select a single candidate by aggregate validation Macro-F1, then accuracy.
    selection_groups = sorted({(row["model"], row["semantic_variant"]) for row in aggregate_rows})
    candidate_scores: list[dict[str, Any]] = []
    for model_name, semantic_variant in selection_groups:
        rows = [row for row in aggregate_rows if row["model"] == model_name and row["semantic_variant"] == semantic_variant]
        candidate_scores.append(
            {
                "model": model_name,
                "semantic_variant": semantic_variant,
                "mean_validation_macro_f1_over_shots": float(np.mean([row["macro_f1_mean"] for row in rows])),
                "mean_validation_accuracy_over_shots": float(np.mean([row["accuracy_mean"] for row in rows])),
            }
        )
    selected = max(candidate_scores, key=lambda row: (row["mean_validation_macro_f1_over_shots"], row["mean_validation_accuracy_over_shots"]))
    after_manifest = snapshot_v1_artifacts(ROOT)
    if before_manifest != after_manifest:
        differences = sorted(set(before_manifest.items()) ^ set(after_manifest.items()))
        raise AssertionError(f"V1 artifacts changed during v2 validation: {differences[:10]}")
    (V2_ROOT / "v1_artifact_manifest_after.json").write_text(json.dumps(after_manifest, indent=2), encoding="utf-8")

    report = {
        "status": "validation_complete_test_not_run",
        "runtime_seconds": time.perf_counter() - started,
        "device": str(DEVICE),
        "configuration": {
            "shots": SHOTS,
            "seeds": RUN_SEEDS,
            "epochs": EPOCHS,
            "episodes_per_epoch": EPISODES_PER_EPOCH,
            "patience": PATIENCE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "validation_episodes": VALIDATION_EPISODES,
            "initial_gate": INITIAL_GATE,
            "reference_norm": reference_norm,
        },
        "split": {key: list(value) for key, value in CLASS_SPLIT.items()},
        "embedding_info": embedding_info,
        "gcn_checkpoints": gcn_identifiers,
        "seed_rows": all_seed_rows,
        "aggregate_rows": aggregate_rows,
        "gate_rows": gate_rows,
        "semantic_rows": semantic_rows,
        "candidate_scores": candidate_scores,
        "selected": selected,
        "v1_artifacts_unchanged": True,
    }
    (V2_ROOT / "validation_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_validation_report(report)
    print(json.dumps({"selected": selected, "candidate_scores": candidate_scores, "runtime_seconds": report["runtime_seconds"], "test_evaluation": "not_run"}, indent=2), flush=True)


def write_validation_report(result: dict[str, Any]) -> None:
    aggregate = result["aggregate_rows"]
    lines = [
        "# LLMGP-v2 Validation Report",
        "",
        "## Status and semantic alignment limitation",
        "Validation-only v2 experiments completed. No test-class embeddings, labels, or episodes were constructed or evaluated. The original Planetoid raw files contain numeric/one-hot labels but no human-readable class-name column, so the mapping from Cora numeric IDs to topic names and the correctness of the configured semantic descriptions cannot be independently verified from this repository. Consequently, semantic ablations are reported as configured/shuffled/random embedding conditions; correct semantic alignment is not claimed.",
        "",
        "## Protocol and implementation",
        f"- Split: train {result['split']['train']}, validation {result['split']['validation']}, test {result['split']['test']} (test unused). Protocol seed {PROTOCOL_SEED}; run seeds {list(RUN_SEEDS)}.",
        f"- Two-way K-shot, query 15/class; {VALIDATION_EPISODES} fixed validation episodes per seed/shot, shared across all models/semantic arms.",
        f"- Training: up to {EPOCHS} epochs x {EPISODES_PER_EPOCH} episodes, Adam lr={LEARNING_RATE}, wd={WEIGHT_DECAY}, patience={PATIENCE}; early stopping/checkpoint selection uses validation Macro-F1, then accuracy, then lower loss.",
        "- Some episodes can have exactly zero prompt gradient because prompt/support first-layer pre-activations are all nonpositive and ReLU blocks the prompt signal. Such no-signal episodes are skipped without optimizer updates and counted per run. Missing or nonfinite gradients remain a hard failure.",
        f"- Model B/C reference norm is the mean L2 norm of Cora node features from training classes [0,1,2] only: {result['configuration']['reference_norm']:.6f}.",
        "- Model A: original linear projection; Model B: L2-normalized projection rescaled to the train-only reference norm; Model C: same normalization multiplied by a learned sigmoid gate initialized to 0.1.",
        "- SciBERT and GCN are frozen. GCNs are reused from v1 and each seed/checkpoint SHA-256 is recorded in the CSV; no GCN was retrained.",
        "",
        "## Validation results by shot",
        "| Model | Semantic condition | Shot | Accuracy mean +/- SD (95% CI) | Macro-F1 mean +/- SD (95% CI) | Best epoch mean | Prompt norm | Gate mean |",
        "|---|---|---:|---|---|---:|---:|---:|",
    ]
    for row in aggregate:
        lines.append(
            f"| {row['model']} | {row['semantic_variant']} | {row['shot']} | {row['accuracy_mean']:.4f} +/- {row['accuracy_sd']:.4f} [{row['accuracy_ci95_lower']:.4f}, {row['accuracy_ci95_upper']:.4f}] | {row['macro_f1_mean']:.4f} +/- {row['macro_f1_sd']:.4f} [{row['macro_f1_ci95_lower']:.4f}, {row['macro_f1_ci95_upper']:.4f}] | {row['best_epoch_mean'] if row['best_epoch_mean'] is not None else '--'} | {row['effective_prompt_norm_mean']:.4f} | {row['gate_mean'] if row['gate_mean'] is not None else '--'} |"
        )
    lines.extend(
        [
            "",
            "## Candidate selection",
            "Candidate ranking is the unweighted mean of per-shot, five-seed validation Macro-F1; validation accuracy is the secondary tie-breaker. Test performance was not accessed.",
            f"- Selected validation candidate: **{result['selected']['model']} / {result['selected']['semantic_variant']}**.",
        ]
    )
    for candidate in result["candidate_scores"]:
        lines.append(
            f"- {candidate['model']} / {candidate['semantic_variant']}: mean validation Macro-F1 {candidate['mean_validation_macro_f1_over_shots']:.4f}; mean accuracy {candidate['mean_validation_accuracy_over_shots']:.4f}."
        )
    lines.extend(
        [
            "",
            "## Answers",
            "- Normalization improvement: compare Model B against Model A and ZeroPrompt/NoPrompt in the validation table; treat gains as validation findings, not test claims.",
            "- Gating improvement beyond normalization: compare Model C/configured with Model B at each shot; the gate distribution and effective prompt norms are recorded in `results/v2/gate_statistics.csv`.",
            "- V2 versus NoPrompt: NoPrompt is included as a frozen-GCN validation control; ZeroPrompt is separately included to isolate retained connectivity with zero prompt inputs.",
            "- Correct semantic benefit: **not verifiable** because the local numeric Cora label files do not establish numeric-ID/topic-name alignment. Configured vs shuffled/random comparisons are conditional on the unverified mapping.",
            "- Gate meaningfully above zero: see per-run gate distribution in `results/v2/gate_statistics.csv`; interpret it together with effective prompt norms, not gate alone.",
            "- Test evaluation: **not run**. The selected candidate may proceed to a separately authorized exploratory test evaluation, but it cannot be called untouched confirmation because previous v1 work already inspected these test classes.",
            "- No-signal episodic gradients: per-run skip totals are included in `results/v2/seed_wise_validation.csv`; one reproduced case had all prompt/support first-layer ReLU activations dead. This diagnostic is not a test-set criterion.",
            "",
            "## Artifacts and commands",
            "- Exact validation command: `& $py experiments/run_llmgp_v2_validation.py` from `LLMGP/`.",
            "- Critical tests: `& $py -m pytest -q tests/v2/test_v2_validation.py`.",
            "- Smoke-only command: `& $py experiments/run_llmgp_v2_validation.py --smoke-only`.",
            "- Outputs: `results/v2/validation_results.csv`, `results/v2/seed_wise_validation.csv`, `results/v2/gate_statistics.csv`, `results/v2/semantic_ablation.csv`, model checkpoints, training logs, fixed validation episode manifest, and before/after v1 artifact manifests.",
            f"- Runtime: {result['runtime_seconds']:.1f} seconds on {result['device']}; test evaluation was not run.",
            "- Existing v1 artifacts SHA-256 comparison: unchanged.",
            "",
            "## Interpretation and next step",
            "Measured facts are the validation metrics, gate values, and embedding norms in the saved CSVs. Any explanation about semantic value is conditional on the unresolved topic mapping. Recommend verifying the class-ID/topic-name mapping against the source `cora.content` file before using the semantic ablation to claim benefit from correctly aligned SciBERT semantics. Do not tune or select using the already examined test set.",
        ]
    )
    (ROOT / "reports" / "llmgp_v2_validation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
