from __future__ import annotations

import argparse
import csv
import json
import sys
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import verify_cora_dataset
from src.llm_encoder import SciBERTEncoder
from src.pretrain_gnn import pretrain_backbone
from src.prompt_builder import PromptBuilder
from src.revised_experiment import (
    QUERY_SHOTS,
    TEMPERATURE,
    confidence_interval,
    evaluate_prompt_model,
    episode_loss_and_predictions,
    load_frozen_backbone,
    sample_split_episode,
    train_prompt_model,
)
from src.revised_protocol import (
    CLASS_DESCRIPTIONS,
    CLASS_ID_TO_TOPIC,
    CLASS_SPLIT,
    PROTOCOL_NAME,
    PROTOCOL_SEED,
    RUN_SEEDS,
    validate_protocol,
)


CONFIG = {
    "shots": (1, 3, 5, 10),
    "max_epochs": 50,
    "episodes_per_epoch": 100,
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "temperature": TEMPERATURE,
    "patience": 10,
    "test_episodes": 600,
    "query_per_class": QUERY_SHOTS,
    "validation_episodes": 30,
    "pretraining_epochs": 100,
    "pretraining_mask_probability": 0.15,
    "hidden_dim": 256,
}
MANUSCRIPT_ACCURACY = {1: 0.614, 3: 0.712, 5: 0.759, 10: 0.813}


def prepare_runtime() -> tuple[dict[str, Any], SciBERTEncoder, torch.Tensor, torch.device]:
    torch.set_num_threads(1)
    info = verify_cora_dataset()
    labels = info["y"].cpu().numpy()
    validate_protocol(labels)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = SciBERTEncoder(model_name="allenai/scibert_scivocab_uncased", device=str(device))
    class_descriptions = [CLASS_DESCRIPTIONS[class_id] for class_id in range(7)]
    class_embeddings = encoder.encode_class_descriptions(class_descriptions, batch_size=7)
    if class_embeddings.shape != (7, 768):
        raise AssertionError(f"Expected seven 768-dimensional SciBERT vectors, got {tuple(class_embeddings.shape)}")
    if any(parameter.requires_grad for parameter in encoder.model.parameters()):
        raise AssertionError("SciBERT must remain frozen")
    return {**info, "labels_array": labels}, encoder, class_embeddings.to(device), device


def check_episode_leakage(labels: np.ndarray) -> None:
    train_episode = sample_split_episode(labels, "train", support_shots=3, seed=PROTOCOL_SEED)
    val_episode = sample_split_episode(labels, "validation", support_shots=3, seed=PROTOCOL_SEED + 1)
    test_episode = sample_split_episode(labels, "test", support_shots=3, seed=PROTOCOL_SEED + 2)
    for split_name, episode in (("train", train_episode), ("validation", val_episode), ("test", test_episode)):
        allowed = set(CLASS_SPLIT[split_name])
        if not set(episode["class_ids"].tolist()).issubset(allowed):
            raise AssertionError(f"{split_name} episode contains a class outside its split")
        if not set(episode["support_idx"].tolist()).isdisjoint(episode["query_idx"].tolist()):
            raise AssertionError(f"{split_name} episode has support/query overlap")


def run_smoke_test() -> dict[str, Any]:
    started = time.perf_counter()
    info, encoder, class_embeddings, device = prepare_runtime()
    labels = info["labels_array"]
    check_episode_leakage(labels)

    smoke_root = ROOT / "results" / "revised_smoke"
    backbone_checkpoint = smoke_root / "seed_2026_frozen_gcn.pt"
    backbone, pretraining_loss = pretrain_backbone(
        info["data"],
        epochs=2,
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"],
        hidden_dim=CONFIG["hidden_dim"],
        seed=PROTOCOL_SEED,
        mask_probability=CONFIG["pretraining_mask_probability"],
        checkpoint_path=backbone_checkpoint,
        device=str(device),
    )
    del backbone
    backbone = load_frozen_backbone(backbone_checkpoint, hidden_dim=CONFIG["hidden_dim"]).to(device)
    if any(parameter.requires_grad for parameter in backbone.parameters()):
        raise AssertionError("Loaded GCN is not frozen")

    checkpoint_path = smoke_root / "shot_1_seed_11_prompt.pt"
    prompt_builder, train_summary, _ = train_prompt_model(
        data=info["data"],
        labels=labels,
        class_embeddings=class_embeddings,
        backbone=backbone,
        shot=1,
        seed=RUN_SEEDS[0],
        checkpoint_path=checkpoint_path,
        log_path=smoke_root / "shot_1_seed_11_training.csv",
        max_epochs=2,
        episodes_per_epoch=2,
        validation_episodes=3,
        patience=1,
    )
    test_summary = evaluate_prompt_model(
        data=info["data"],
        labels=labels,
        class_embeddings=class_embeddings,
        backbone=backbone,
        prompt_builder=prompt_builder,
        shot=1,
        seed=RUN_SEEDS[0],
        episodes=3,
        predictions_path=smoke_root / "smoke_predictions.csv",
        episode_metrics_path=smoke_root / "smoke_episode_metrics.csv",
    )

    episode = sample_split_episode(labels, "test", 1, seed=987_654)
    data_x = info["x"].to(device)
    edge_index = info["edge_index"].to(device)
    with torch.no_grad():
        before_logits = episode_loss_and_predictions(
            episode, class_embeddings, prompt_builder, backbone, data_x, edge_index, device
        )[1].clone()
    saved = torch.load(checkpoint_path, map_location=device, weights_only=False)
    reloaded = PromptBuilder().to(device)
    reloaded.load_state_dict(saved["prompt_state_dict"])
    reloaded.eval()
    with torch.no_grad():
        after_logits = episode_loss_and_predictions(
            episode, class_embeddings, reloaded, backbone, data_x, edge_index, device
        )[1]
    torch.testing.assert_close(before_logits, after_logits, rtol=0, atol=0)
    if any(parameter.grad is not None for parameter in backbone.parameters()):
        raise AssertionError("Frozen GCN has gradients after smoke test")
    if any(parameter.requires_grad for parameter in encoder.model.parameters()):
        raise AssertionError("SciBERT was unexpectedly unfrozen")

    result = {
        "status": "passed",
        "protocol": PROTOCOL_NAME,
        "split": {key: list(value) for key, value in CLASS_SPLIT.items()},
        "protocol_seed": PROTOCOL_SEED,
        "run_seed": RUN_SEEDS[0],
        "shot": 1,
        "pretraining_loss": pretraining_loss,
        "best_epoch": train_summary["best_epoch"],
        "best_validation_loss": train_summary["best_validation_loss"],
        "smoke_test_metrics": test_summary,
        "projection_checkpoint_reload_logits_identical": True,
        "frozen_gcn": True,
        "frozen_scibert": True,
        "runtime_seconds_including_model_load": time.perf_counter() - started,
        "device": str(device),
    }
    smoke_root.mkdir(parents=True, exist_ok=True)
    (smoke_root / "smoke_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def run_full_experiments() -> dict[str, Any]:
    full_started = time.perf_counter()
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_revised_protocol.py",
            "tests/test_self_supervised_pretraining.py",
            "tests/test_revised_training.py",
        ],
        cwd=ROOT,
        check=True,
    )
    smoke_gate = run_smoke_test()
    if smoke_gate["status"] != "passed":
        raise RuntimeError("Revised-protocol smoke gate failed; full experiments were not started")
    info, encoder, class_embeddings, device = prepare_runtime()
    labels = info["labels_array"]
    check_episode_leakage(labels)

    metrics_dir = ROOT / "results" / "metrics"
    checkpoints_dir = ROOT / "results" / "checkpoints"
    logs_dir = ROOT / "results" / "logs" / "revised"
    figures_dir = ROOT / "results" / "figures"
    for directory in (metrics_dir, checkpoints_dir, logs_dir, figures_dir):
        directory.mkdir(parents=True, exist_ok=True)

    seed_rows: list[dict[str, Any]] = []
    pretraining_rows: list[dict[str, Any]] = []
    episode_results = metrics_dir / "episode_metrics.csv"
    if episode_results.exists():
        episode_results.unlink()

    for seed in RUN_SEEDS:
        print(f"Pretraining frozen GCN for seed {seed}", flush=True)
        gcn_path = checkpoints_dir / f"seed_{seed}_frozen_gcn.pt"
        _, pretrain_loss = pretrain_backbone(
            info["data"],
            epochs=CONFIG["pretraining_epochs"],
            lr=CONFIG["learning_rate"],
            weight_decay=CONFIG["weight_decay"],
            hidden_dim=CONFIG["hidden_dim"],
            seed=seed,
            mask_probability=CONFIG["pretraining_mask_probability"],
            checkpoint_path=gcn_path,
            device=str(device),
        )
        (logs_dir / f"seed_{seed}_pretraining.json").write_text(
            json.dumps(
                {
                    "seed": seed,
                    "objective": "masked_feature_reconstruction_mse",
                    "epochs": CONFIG["pretraining_epochs"],
                    "mask_probability": CONFIG["pretraining_mask_probability"],
                    "learning_rate": CONFIG["learning_rate"],
                    "weight_decay": CONFIG["weight_decay"],
                    "hidden_dim": CONFIG["hidden_dim"],
                    "final_loss": pretrain_loss,
                    "loss_history": torch.load(gcn_path, map_location="cpu", weights_only=False)["pretraining"]["loss_history"],
                    "graph_visibility": "all node features and graph edges; no node labels",
                    "checkpoint": str(gcn_path.relative_to(ROOT)),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        pretraining_rows.append({"seed": seed, "pretraining_loss": pretrain_loss, "checkpoint": str(gcn_path)})
        del _
        backbone = load_frozen_backbone(gcn_path, hidden_dim=CONFIG["hidden_dim"]).to(device)

        for shot in CONFIG["shots"]:
            print(f"Training {shot}-shot prompt model, seed {seed}", flush=True)
            prompt_path = checkpoints_dir / f"shot_{shot}_seed_{seed}_prompt.pt"
            training_log = logs_dir / f"shot_{shot}_seed_{seed}_training.csv"
            prompt_builder, training_summary, _ = train_prompt_model(
                data=info["data"],
                labels=labels,
                class_embeddings=class_embeddings,
                backbone=backbone,
                shot=shot,
                seed=seed,
                checkpoint_path=prompt_path,
                log_path=training_log,
                max_epochs=CONFIG["max_epochs"],
                episodes_per_epoch=CONFIG["episodes_per_epoch"],
                validation_episodes=CONFIG["validation_episodes"],
                patience=CONFIG["patience"],
                learning_rate=CONFIG["learning_rate"],
                weight_decay=CONFIG["weight_decay"],
            )
            prediction_path = metrics_dir / f"raw_predictions_seed_{seed}_shot_{shot}.csv"
            if prediction_path.exists():
                prediction_path.unlink()
            eval_summary = evaluate_prompt_model(
                data=info["data"],
                labels=labels,
                class_embeddings=class_embeddings,
                backbone=backbone,
                prompt_builder=prompt_builder,
                shot=shot,
                seed=seed,
                episodes=CONFIG["test_episodes"],
                predictions_path=prediction_path,
                episode_metrics_path=episode_results,
            )
            seed_rows.append(
                {
                    "seed": seed,
                    "shot": shot,
                    "test_episodes": CONFIG["test_episodes"],
                    "accuracy_mean": eval_summary["accuracy_mean"],
                    "accuracy_episode_std": eval_summary["accuracy_std_episode"],
                    "macro_f1_mean": eval_summary["macro_f1_mean"],
                    "macro_f1_episode_std": eval_summary["macro_f1_std_episode"],
                    "best_epoch": training_summary["best_epoch"],
                    "best_validation_loss": training_summary["best_validation_loss"],
                    "best_validation_accuracy": training_summary["best_validation_accuracy"],
                    "training_seconds": training_summary["training_seconds"],
                    "pretraining_loss": pretrain_loss,
                    "prompt_checkpoint": str(prompt_path.relative_to(ROOT)),
                    "gcn_checkpoint": str(gcn_path.relative_to(ROOT)),
                }
            )
        del backbone

    seed_fields = list(seed_rows[0])
    with (metrics_dir / "seed_wise_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=seed_fields)
        writer.writeheader()
        writer.writerows(seed_rows)

    final_rows: list[dict[str, Any]] = []
    for shot in CONFIG["shots"]:
        shot_rows = [row for row in seed_rows if row["shot"] == shot]
        accuracy_values = [row["accuracy_mean"] for row in shot_rows]
        f1_values = [row["macro_f1_mean"] for row in shot_rows]
        accuracy_mean, accuracy_std, accuracy_low, accuracy_high = confidence_interval(accuracy_values)
        f1_mean, f1_std, f1_low, f1_high = confidence_interval(f1_values)
        reference = MANUSCRIPT_ACCURACY[shot]
        final_rows.append(
            {
                "shot": shot,
                "seeds": len(shot_rows),
                "test_episodes_per_seed": CONFIG["test_episodes"],
                "accuracy_mean_across_seeds": accuracy_mean,
                "accuracy_std_across_seeds": accuracy_std,
                "accuracy_ci95_lower": accuracy_low,
                "accuracy_ci95_upper": accuracy_high,
                "macro_f1_mean_across_seeds": f1_mean,
                "macro_f1_std_across_seeds": f1_std,
                "macro_f1_ci95_lower": f1_low,
                "macro_f1_ci95_upper": f1_high,
                "manuscript_original_protocol_accuracy_reference": reference,
                "absolute_difference_vs_reference": abs(accuracy_mean - reference),
                "comparison_status": "different_protocol_reference_only_not_reproduction",
            }
        )

    final_fields = list(final_rows[0])
    with (metrics_dir / "final_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=final_fields)
        writer.writeheader()
        writer.writerows(final_rows)

    make_figures(final_rows, figures_dir)
    runtime_seconds = time.perf_counter() - full_started
    report = {
        "status": "completed",
        "protocol": PROTOCOL_NAME,
        "protocol_seed": PROTOCOL_SEED,
        "run_seeds": list(RUN_SEEDS),
        "class_split": {key: list(value) for key, value in CLASS_SPLIT.items()},
        "class_id_to_topic": CLASS_ID_TO_TOPIC,
        "class_descriptions": CLASS_DESCRIPTIONS,
        "dataset_nodes": info["num_nodes"],
        "dataset_features": info["num_features"],
        "configuration": CONFIG,
        "pretraining_runs": pretraining_rows,
        "seed_wise_results": seed_rows,
        "final_results": final_rows,
        "runtime_seconds_including_model_load": runtime_seconds,
        "device": str(device),
    }
    write_final_report(report)
    print(json.dumps({"final_results": final_rows, "runtime_seconds": runtime_seconds}, indent=2), flush=True)
    return report


def make_figures(final_rows: list[dict[str, Any]], figures_dir: Path) -> None:
    shots = [int(row["shot"]) for row in final_rows]
    means = [float(row["accuracy_mean_across_seeds"]) for row in final_rows]
    lows = [float(row["accuracy_ci95_lower"]) for row in final_rows]
    highs = [float(row["accuracy_ci95_upper"]) for row in final_rows]
    references = [float(row["manuscript_original_protocol_accuracy_reference"]) for row in final_rows]
    errors = [np.asarray(means) - np.asarray(lows), np.asarray(highs) - np.asarray(means)]

    figure, axis = plt.subplots(figsize=(8, 5))
    positions = np.arange(len(shots))
    axis.bar(positions - 0.18, means, width=0.36, label="Revised 3/2/2 protocol", color="#157a6e")
    axis.bar(positions + 0.18, references, width=0.36, label="Manuscript 4/1/2 reference", color="#d9822b")
    axis.errorbar(positions - 0.18, means, yerr=errors, fmt="none", ecolor="#162b32", capsize=4)
    axis.set_xticks(positions, [f"{shot}-shot" for shot in shots])
    axis.set_ylabel("Accuracy")
    axis.set_ylim(0, 1)
    axis.set_title("Different protocols; reference values are not reproduction targets")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(figures_dir / "accuracy_comparison.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(shots, means, marker="o", linewidth=2, color="#157a6e", label="Revised 3/2/2 protocol")
    axis.fill_between(shots, lows, highs, color="#157a6e", alpha=0.18, label="95% t CI across seeds")
    axis.plot(shots, references, marker="s", linestyle="--", color="#d9822b", label="Manuscript 4/1/2 reference")
    axis.set_xticks(shots)
    axis.set_xlabel("Support examples per class")
    axis.set_ylabel("Accuracy")
    axis.set_ylim(0, 1)
    axis.set_title("Accuracy by shot setting")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(figures_dir / "accuracy_trend.png", dpi=180)
    plt.close(figure)


def write_final_report(report: dict[str, Any]) -> None:
    configuration = report["configuration"]
    lines = [
        "# Revised Protocol Experiment Report",
        "",
        "## Status",
        "Full revised-protocol runs completed successfully. These experiments use a methodological change from the manuscript's original 4/1/2 class split: a fixed 3/2/2 split and self-supervised masked-feature GCN pretraining. Results are not claimed to reproduce the manuscript.",
        "",
        "## Protocol and class mapping",
        f"- Protocol: `{report['protocol']}`.",
        f"- Fixed protocol seed: {report['protocol_seed']}; independent run seeds: {', '.join(map(str, report['run_seeds']))}.",
        f"- Train class IDs: {report['class_split']['train']}.",
        f"- Validation class IDs: {report['class_split']['validation']}.",
        f"- Test class IDs: {report['class_split']['test']}.",
        "- Split assignment was fixed before training and was not selected or adjusted using test accuracy.",
        "- Cora class-ID mapping was checked against the canonical per-class node counts in the loaded Planetoid labels.",
    ]
    for class_id, topic in report["class_id_to_topic"].items():
        lines.append(f"- Class {class_id}: {topic}; prompt description: {report['class_descriptions'][class_id]}")
    lines.extend(
        [
            "",
            "## Self-supervised GCN pretraining",
            "- Objective: denoising masked-feature reconstruction. For each epoch, independently mask 15% of observed nonzero bag-of-words entries, run the two-layer GCN on corrupted features and the graph, map the 256-dimensional node states back to 1,433 features, and minimize MSE only on masked entries.",
            "- Graph visibility: all Cora nodes, bag-of-words features, and graph edges are visible to this transductive self-supervised objective. No node labels are read by the pretraining function, used in the objective, or used for checkpoint selection.",
            f"- Optimization: Adam, learning rate {configuration['learning_rate']}, weight decay {configuration['weight_decay']}, {configuration['pretraining_epochs']} epochs, hidden size {configuration['hidden_dim']}, mask probability {configuration['pretraining_mask_probability']}.",
            "- One frozen GCN checkpoint was trained and saved for each of the five run seeds; each seed's checkpoint is shared across that seed's four independently trained prompt models.",
            "- SciBERT remains frozen. During prompt training only the 768-to-1,433 projection layer is optimized; the GCN is frozen while gradients propagate through it to the prompt nodes.",
            "",
            "## Experimental configuration",
            f"- Dataset: Cora Planetoid, {report['dataset_nodes']} nodes, {report['dataset_features']} bag-of-words features, 7 classes.",
            "- Task: 2-way episodes in train, validation, and test splits; 15 query nodes per class.",
            f"- Shots: {', '.join(map(str, configuration['shots']))}; maximum epochs {configuration['max_epochs']}; {configuration['episodes_per_epoch']} training episodes per epoch; early stopping patience {configuration['patience']}.",
            f"- Prompt optimizer: Adam, learning rate {configuration['learning_rate']}, weight decay {configuration['weight_decay']}; cosine temperature {configuration['temperature']}.",
            f"- Validation: {configuration['validation_episodes']} fixed-seed 2-way episodes per epoch on classes {report['class_split']['validation']}; checkpoint selected by mean validation cross-entropy only.",
            f"- Test: {configuration['test_episodes']} episodes per shot and seed on classes {report['class_split']['test']}; test episodes were evaluated under `torch.no_grad()` with parameter-state equality checks.",
            "- 95% confidence intervals: two-sided Student t intervals over the five independent seed-level mean metrics (df=4, t critical value from SciPy); standard deviation uses the sample estimator (ddof=1).",
            "",
            "## Results",
            "| Shot | Revised accuracy (mean +/- SD) | 95% CI | Macro-F1 (mean +/- SD) | Manuscript reference (original 4/1/2) | Absolute difference | Comparison status |",
            "|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in report["final_results"]:
        lines.append(
            f"| {row['shot']} | {row['accuracy_mean_across_seeds']:.4f} +/- {row['accuracy_std_across_seeds']:.4f} | [{row['accuracy_ci95_lower']:.4f}, {row['accuracy_ci95_upper']:.4f}] | {row['macro_f1_mean_across_seeds']:.4f} +/- {row['macro_f1_std_across_seeds']:.4f} | {row['manuscript_original_protocol_accuracy_reference']:.4f} | {row['absolute_difference_vs_reference']:.4f} | Different protocol; not reproduction |"
        )
    lines.extend(
        [
            "",
            "## Artifacts and commands",
            "- GCN checkpoints: `results/checkpoints/seed_<seed>_frozen_gcn.pt`.",
            "- Prompt checkpoints: `results/checkpoints/shot_<shot>_seed_<seed>_prompt.pt`.",
            "- Pretraining logs: `results/logs/revised/seed_<seed>_pretraining.json`.",
            "- Prompt training logs: `results/logs/revised/shot_<shot>_seed_<seed>_training.csv`.",
            "- Raw predictions: `results/metrics/raw_predictions_seed_<seed>_shot_<shot>.csv`.",
            "- Episode metrics: `results/metrics/episode_metrics.csv`.",
            "- Seed-wise metrics: `results/metrics/seed_wise_results.csv`.",
            "- Aggregate metrics and confidence intervals: `results/metrics/final_results.csv`.",
            "- Figures: `results/figures/accuracy_comparison.png` and `results/figures/accuracy_trend.png`.",
            "- Smoke command: `& $py experiments/run_revised_experiments.py --smoke-only`.",
            "- Full command: `& $py experiments/run_revised_experiments.py --full`.",
            f"- Total full-run elapsed time, including model load: {report['runtime_seconds_including_model_load']:.2f} seconds on {report['device']}.",
            "",
            "## Interpretation and limitations",
            "- The 3/2/2 class split resolves the manuscript's 4/1/2 versus 2-way-validation inconsistency for this approved revised study. It is an explicit protocol change.",
            "- Self-supervised pretraining is also a methodological change; all graph features and edges are transductively visible, while labels are excluded from pretraining.",
            "- The manuscript's accuracy values are shown only as reference points from its different original protocol. Absolute differences do not measure reproduction error.",
            "- Class descriptions are explicit manually authored short descriptions of the mapped Cora topic labels; they are not asserted to be the manuscript's exact wording.",
            "- Standard Planetoid Cora provides bag-of-words features but not aligned paper titles/abstracts in this repository; per-node text encoding is not part of these runs.",
            "- The new metrics are verified only for the specified transductive revised protocol and should not be compared as a direct replication of the manuscript.",
        ]
    )
    (ROOT / "reports" / "revised_protocol_experiment_report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the leakage-safe revised LLMGP protocol.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke-only", action="store_true", help="Run short pretraining, prompt, and inference smoke checks.")
    mode.add_argument("--full", action="store_true", help="Run all shots and five seeds after smoke/tests pass.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke_only:
        run_smoke_test()
    else:
        run_full_experiments()


if __name__ == "__main__":
    main()