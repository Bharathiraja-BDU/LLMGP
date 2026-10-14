from __future__ import annotations

import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import t
from sklearn.metrics import accuracy_score, f1_score
from torch_geometric.nn.conv.gcn_conv import gcn_norm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.classifier import ProtoClassifier
from src.data_loader import verify_cora_dataset
from src.graph_augmenter import GraphAugmenter
from src.gnn_backbone import GCNBackbone
from src.llm_encoder import SciBERTEncoder
from src.prompt_builder import PromptBuilder
from src.revised_experiment import QUERY_SHOTS, TEMPERATURE, sample_split_episode
from src.revised_protocol import (
    CLASS_DESCRIPTIONS,
    CLASS_SPLIT,
    RUN_SEEDS,
    labels_visible_in_split,
    validate_protocol,
)


SCALES = (0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0)
SHOTS = (1, 3, 5, 10)
VALIDATION_EPISODES = 200
TEST_EPISODES = 600
DIAGNOSTIC_ROOT = ROOT / "diagnostics"
CHECKPOINT_ROOT = ROOT / "results" / "checkpoints"
METRICS_ROOT = ROOT / "results" / "metrics"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_run_models(seed: int, shot: int) -> tuple[GCNBackbone, PromptBuilder, dict[str, Any]]:
    gcn_path = CHECKPOINT_ROOT / f"seed_{seed}_frozen_gcn.pt"
    prompt_path = CHECKPOINT_ROOT / f"shot_{shot}_seed_{seed}_prompt.pt"
    gcn_checkpoint = torch.load(gcn_path, map_location="cpu", weights_only=False)
    prompt_checkpoint = torch.load(prompt_path, map_location="cpu", weights_only=False)
    gcn_metadata = gcn_checkpoint["pretraining"]
    prompt_metadata = prompt_checkpoint["metadata"]
    if int(gcn_metadata["seed"]) != seed or int(prompt_metadata["seed"]) != seed:
        raise AssertionError("Checkpoint seed mismatch")
    if int(prompt_metadata["shot"]) != shot:
        raise AssertionError("Prompt checkpoint shot mismatch")
    if prompt_metadata["backbone_checkpoint"].replace("\\", "/").endswith(f"seed_{seed}_frozen_gcn.pt") is False:
        raise AssertionError("Prompt checkpoint points at a different GCN seed")

    backbone = GCNBackbone(in_channels=1433, hidden_channels=256, out_channels=256)
    backbone.load_state_dict(gcn_checkpoint["model_state_dict"])
    backbone.to(DEVICE).eval()
    for parameter in backbone.parameters():
        parameter.requires_grad = False
        parameter.grad = None

    prompt_builder = PromptBuilder().to(DEVICE)
    prompt_builder.load_state_dict(prompt_checkpoint["prompt_state_dict"])
    prompt_builder.eval()
    return backbone, prompt_builder, {
        "gcn_checkpoint": str(gcn_path.relative_to(ROOT)),
        "prompt_checkpoint": str(prompt_path.relative_to(ROOT)),
        "gcn_seed": int(gcn_metadata["seed"]),
        "prompt_seed": int(prompt_metadata["seed"]),
        "shot": int(prompt_metadata["shot"]),
    }


def normalization_statistics(edge_index: torch.Tensor, num_nodes: int, original_node_count: int) -> dict[str, float]:
    raw_degrees = torch.bincount(edge_index[1], minlength=num_nodes)
    normalized_edges, normalized_weights = gcn_norm(
        edge_index,
        num_nodes=num_nodes,
        add_self_loops=True,
        dtype=torch.float32,
    )
    if normalized_weights is None:
        raise AssertionError("Expected scalar normalized edge weights")
    source, target = normalized_edges
    original_edge_mask = (source < original_node_count) & (target < original_node_count)
    prompt_edge_mask = (source >= original_node_count) | (target >= original_node_count)
    prompt_nodes = raw_degrees[original_node_count:]
    return {
        "normalized_edge_weight_mean": float(normalized_weights.mean().item()),
        "normalized_edge_weight_std": float(normalized_weights.std(unbiased=False).item()),
        "normalized_edge_weight_min": float(normalized_weights.min().item()),
        "normalized_edge_weight_max": float(normalized_weights.max().item()),
        "original_subgraph_normalized_weight_mean": float(normalized_weights[original_edge_mask].mean().item()),
        "prompt_incident_normalized_weight_mean": (
            float(normalized_weights[prompt_edge_mask].mean().item()) if prompt_edge_mask.any() else 0.0
        ),
        "prompt_node_degree_min": float(prompt_nodes.min().item()) if prompt_nodes.numel() else 0.0,
        "prompt_node_degree_mean": float(prompt_nodes.float().mean().item()) if prompt_nodes.numel() else 0.0,
        "prompt_node_degree_max": float(prompt_nodes.max().item()) if prompt_nodes.numel() else 0.0,
        "normalized_edge_count": int(normalized_edges.shape[1]),
    }


def infer_one(
    *,
    episode: dict[str, Any],
    x: torch.Tensor,
    edge_index: torch.Tensor,
    class_embeddings: torch.Tensor,
    backbone: GCNBackbone,
    prompt_builder: PromptBuilder,
    scale: float | None,
    baseline_embeddings: torch.Tensor,
) -> dict[str, Any]:
    support_idx = torch.as_tensor(episode["support_idx"], dtype=torch.long, device=DEVICE)
    query_idx = torch.as_tensor(episode["query_idx"], dtype=torch.long, device=DEVICE)
    support_labels = torch.as_tensor(episode["support_labels"], dtype=torch.long, device=DEVICE)
    query_labels = torch.as_tensor(episode["query_labels"], dtype=torch.long, device=DEVICE)
    class_ids = torch.as_tensor(episode["class_ids"], dtype=torch.long, device=DEVICE)

    if scale is None:
        query_embeddings = baseline_embeddings[query_idx]
        support_embeddings = baseline_embeddings[support_idx]
        used_edges = edge_index
        prompt_features = torch.empty((0, x.shape[1]), dtype=x.dtype, device=DEVICE)
        norm_stats = normalization_statistics(edge_index, x.shape[0], x.shape[0])
        variant = "NoPrompt"
        actual_scale: float | None = None
    else:
        projected_prompts = prompt_builder(class_embeddings[class_ids])
        prompt_features = projected_prompts * scale
        augmented = GraphAugmenter(num_classes=2).augment(
            x,
            edge_index,
            support_idx,
            support_labels,
            prompt_embeddings=prompt_features,
        )
        node_embeddings = backbone(augmented.x, augmented.edge_index)
        original_embeddings = node_embeddings[augmented.original_mask]
        query_embeddings = original_embeddings[query_idx]
        support_embeddings = original_embeddings[support_idx]
        used_edges = augmented.edge_index
        norm_stats = normalization_statistics(used_edges, augmented.x.shape[0], x.shape[0])
        actual_scale = float(scale)
        variant = "ZeroPrompt" if scale == 0.0 else ("OriginalPrompt" if scale == 1.0 else "ScaledPrompt")

    prototypes = ProtoClassifier.prototypes(support_embeddings, support_labels, 2)
    logits = ProtoClassifier.logits(query_embeddings, prototypes, temperature=TEMPERATURE)
    predictions = logits.argmax(dim=-1)
    query_truth = query_labels.cpu().numpy()
    query_predictions = predictions.cpu().numpy()
    baseline_query = baseline_embeddings[query_idx]
    prompt_norm = (
        float(torch.linalg.vector_norm(prompt_features, dim=-1).mean().item())
        if prompt_features.shape[0]
        else 0.0
    )
    query_delta = query_embeddings - baseline_query
    query_l2_delta = float(torch.linalg.vector_norm(query_delta, dim=-1).mean().item())
    query_relative_delta = query_l2_delta / max(
        float(torch.linalg.vector_norm(baseline_query, dim=-1).mean().item()), 1e-12
    )
    query_cosine = float(
        torch.nn.functional.cosine_similarity(query_embeddings, baseline_query, dim=-1).mean().item()
    )

    return {
        "variant": variant,
        "scale_factor": actual_scale,
        "accuracy": float(accuracy_score(query_truth, query_predictions)),
        "macro_f1": float(f1_score(query_truth, query_predictions, labels=[0, 1], average="macro", zero_division=0)),
        "prompt_feature_norm_mean": prompt_norm,
        "prompt_to_original_feature_norm_ratio": prompt_norm / max(float(x.norm(dim=-1).mean().item()), 1e-12),
        "query_embedding_l2_change_vs_noprompt": query_l2_delta,
        "query_embedding_relative_l2_change_vs_noprompt": query_relative_delta,
        "query_embedding_cosine_vs_noprompt": query_cosine,
        "augmented_edge_entries": int(used_edges.shape[1]),
        **norm_stats,
    }


def episode_seed(split: str, seed: int, shot: int, episode_index: int) -> int:
    split_offset = 8_000_000 if split == "validation" else 9_000_000
    return seed * 10_000_000 + split_offset + shot * 100_000 + episode_index


def saved_test_episodes_match(seed: int, shot: int, episodes: list[dict[str, Any]]) -> None:
    saved_rows = read_csv(METRICS_ROOT / f"raw_predictions_seed_{seed}_shot_{shot}.csv")
    by_episode: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in saved_rows:
        by_episode[int(row["episode"])].append(row)
    if len(by_episode) != TEST_EPISODES:
        raise AssertionError("Saved test prediction count is not 600 episodes")
    for index, episode in enumerate(episodes):
        rows = by_episode[index]
        if [int(row["query_node_id"]) for row in rows] != [int(value) for value in episode["query_idx"]]:
            raise AssertionError(f"Test query sequence mismatch at episode {index}, seed={seed}, shot={shot}")
        expected_labels = [int(episode["class_ids"][int(local)]) for local in episode["query_labels"]]
        if [int(row["true_class_id"]) for row in rows] != expected_labels:
            raise AssertionError(f"Test labels mismatch at episode {index}, seed={seed}, shot={shot}")


def summarize_seed_rows(raw_rows: list[dict[str, Any]], split_name: str) -> list[dict[str, Any]]:
    groups: dict[tuple[float, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        scale_key = -1.0 if row["scale_factor"] is None else float(row["scale_factor"])
        groups[(scale_key, int(row["shot"]), int(row["seed"]))].append(row)
    seed_summaries: list[dict[str, Any]] = []
    for (scale_key, shot, seed), rows in groups.items():
        summary = dict(rows[0])
        summary["scale_factor"] = None if scale_key < 0 else scale_key
        summary["episode_count"] = len(rows)
        for metric in (
            "accuracy",
            "macro_f1",
            "prompt_feature_norm_mean",
            "prompt_to_original_feature_norm_ratio",
            "query_embedding_l2_change_vs_noprompt",
            "query_embedding_relative_l2_change_vs_noprompt",
            "query_embedding_cosine_vs_noprompt",
            "normalized_edge_weight_mean",
            "normalized_edge_weight_std",
            "normalized_edge_weight_min",
            "normalized_edge_weight_max",
            "original_subgraph_normalized_weight_mean",
            "prompt_incident_normalized_weight_mean",
            "prompt_node_degree_min",
            "prompt_node_degree_mean",
            "prompt_node_degree_max",
            "augmented_edge_entries",
            "normalized_edge_count",
        ):
            summary[metric] = float(np.mean([float(row[metric]) for row in rows]))
        summary["split"] = split_name
        seed_summaries.append(summary)
    return seed_summaries


def add_seed_confidence_intervals(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, float | None, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["split"]), row["scale_factor"], int(row["shot"]))].append(row)
    enriched: list[dict[str, Any]] = []
    for row in rows:
        key = (str(row["split"]), row["scale_factor"], int(row["shot"]))
        peers = groups[key]
        for metric in ("accuracy", "macro_f1"):
            values = [float(peer[metric]) for peer in peers]
            mean = float(np.mean(values))
            std = float(np.std(values, ddof=1))
            margin = float(t.ppf(0.975, df=len(values) - 1) * std / np.sqrt(len(values)))
            row[f"{metric}_mean_across_seeds"] = mean
            row[f"{metric}_std_across_seeds"] = std
            row[f"{metric}_ci95_lower"] = mean - margin
            row[f"{metric}_ci95_upper"] = mean + margin
        enriched.append(row)
    return enriched


def run_sweep(
    split: str,
    info: dict[str, Any],
    labels: np.ndarray,
    class_embeddings: torch.Tensor,
    episode_count: int,
    verify_saved_test: bool,
) -> list[dict[str, Any]]:
    x = info["x"].to(DEVICE)
    edge_index = info["edge_index"].to(DEVICE)
    raw_rows: list[dict[str, Any]] = []

    for seed in RUN_SEEDS:
        for shot in SHOTS:
            backbone, prompt_builder, checkpoint_info = load_run_models(seed, shot)
            episodes = [
                sample_split_episode(labels, split, shot, episode_seed(split, seed, shot, index))
                for index in range(episode_count)
            ]
            if verify_saved_test:
                saved_test_episodes_match(seed, shot, episodes)

            with torch.no_grad():
                base_output = backbone(x, edge_index)
                baseline_embeddings = base_output.detach()
                prompt_classes = torch.as_tensor(CLASS_SPLIT[split], dtype=torch.long, device=DEVICE)
                for episode_index, episode in enumerate(episodes):
                    for variant_scale in (None, *SCALES):
                        metrics = infer_one(
                            episode=episode,
                            x=x,
                            edge_index=edge_index,
                            class_embeddings=class_embeddings,
                            backbone=backbone,
                            prompt_builder=prompt_builder,
                            scale=variant_scale,
                            baseline_embeddings=baseline_embeddings,
                        )
                        raw_rows.append(
                            {
                                "split": split,
                                "seed": seed,
                                "shot": shot,
                                "episode": episode_index,
                                "variant": metrics["variant"],
                                "scale_factor": metrics["scale_factor"],
                                "accuracy": metrics["accuracy"],
                                "macro_f1": metrics["macro_f1"],
                                "prompt_feature_norm_mean": metrics["prompt_feature_norm_mean"],
                                "prompt_to_original_feature_norm_ratio": metrics["prompt_to_original_feature_norm_ratio"],
                                "query_embedding_l2_change_vs_noprompt": metrics["query_embedding_l2_change_vs_noprompt"],
                                "query_embedding_relative_l2_change_vs_noprompt": metrics["query_embedding_relative_l2_change_vs_noprompt"],
                                "query_embedding_cosine_vs_noprompt": metrics["query_embedding_cosine_vs_noprompt"],
                                "augmented_edge_entries": metrics["augmented_edge_entries"],
                                "normalized_edge_count": metrics["normalized_edge_count"],
                                "normalized_edge_weight_mean": metrics["normalized_edge_weight_mean"],
                                "normalized_edge_weight_std": metrics["normalized_edge_weight_std"],
                                "normalized_edge_weight_min": metrics["normalized_edge_weight_min"],
                                "normalized_edge_weight_max": metrics["normalized_edge_weight_max"],
                                "original_subgraph_normalized_weight_mean": metrics["original_subgraph_normalized_weight_mean"],
                                "prompt_incident_normalized_weight_mean": metrics["prompt_incident_normalized_weight_mean"],
                                "prompt_node_degree_min": metrics["prompt_node_degree_min"],
                                "prompt_node_degree_mean": metrics["prompt_node_degree_mean"],
                                "prompt_node_degree_max": metrics["prompt_node_degree_max"],
                                "episode_seed": episode_seed(split, seed, shot, episode_index),
                                "gcn_checkpoint": checkpoint_info["gcn_checkpoint"],
                                "prompt_checkpoint": checkpoint_info["prompt_checkpoint"],
                            }
                        )
            del backbone, prompt_builder
            print(f"completed split={split}, seed={seed}, shot={shot}", flush=True)

    return summarize_seed_rows(raw_rows, split)


def selected_scale_from_validation(validation_rows: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate: list[dict[str, Any]] = []
    for scale in SCALES:
        rows = [row for row in validation_rows if row["scale_factor"] == scale]
        aggregate.append(
            {
                "scale_factor": scale,
                "mean_accuracy_over_seed_shot": float(np.mean([row["accuracy"] for row in rows])),
                "mean_macro_f1_over_seed_shot": float(np.mean([row["macro_f1"] for row in rows])),
            }
        )
    selected = max(aggregate, key=lambda row: row["mean_macro_f1_over_seed_shot"])
    return {
        "selection_metric": "mean validation Macro-F1 averaged equally over five seeds and four shot settings",
        "candidate_scales": list(SCALES),
        "validation_aggregate": aggregate,
        "selected_scale": selected["scale_factor"],
        "selected_validation_mean_macro_f1": selected["mean_macro_f1_over_seed_shot"],
        "selected_validation_mean_accuracy": selected["mean_accuracy_over_seed_shot"],
    }


def connectivity_rows(scale_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in scale_rows:
        scale = row["scale_factor"]
        if scale is None:
            model_name = "NoPrompt"
        elif scale == 0.0:
            model_name = "ZeroPrompt"
        elif scale == 1.0:
            model_name = "OriginalPrompt"
        else:
            model_name = "ScaledPrompt"
        output.append({"connectivity_variant": model_name, **row})
    return output


def write_report(
    validation_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    selection: dict[str, Any],
    runtime_seconds: float,
) -> None:
    test_by_run = {
        (int(row["seed"]), int(row["shot"]), row["scale_factor"]): row
        for row in test_rows
    }
    contrast_definitions = (
        ("ZeroPrompt - NoPrompt", 0.0, None),
        ("Scale 0.01 - ZeroPrompt", 0.01, 0.0),
        ("OriginalPrompt - ZeroPrompt", 1.0, 0.0),
    )
    paired_contrasts: list[dict[str, Any]] = []
    for shot in SHOTS:
        for name, first_scale, second_scale in contrast_definitions:
            differences = [
                float(test_by_run[(seed, shot, first_scale)]["accuracy"])
                - float(test_by_run[(seed, shot, second_scale)]["accuracy"])
                for seed in RUN_SEEDS
            ]
            mean_difference = float(np.mean(differences))
            margin = float(t.ppf(0.975, 4) * np.std(differences, ddof=1) / np.sqrt(5))
            paired_contrasts.append(
                {
                    "shot": shot,
                    "contrast": name,
                    "seed_differences": differences,
                    "mean_difference": mean_difference,
                    "ci95_lower": mean_difference - margin,
                    "ci95_upper": mean_difference + margin,
                }
            )
    overall_paired_contrasts: list[dict[str, Any]] = []
    for name, first_scale, second_scale in contrast_definitions:
        per_seed = [
            float(
                np.mean(
                    [
                        float(test_by_run[(seed, shot, first_scale)]["accuracy"])
                        - float(test_by_run[(seed, shot, second_scale)]["accuracy"])
                        for shot in SHOTS
                    ]
                )
            )
            for seed in RUN_SEEDS
        ]
        mean_difference = float(np.mean(per_seed))
        margin = float(t.ppf(0.975, 4) * np.std(per_seed, ddof=1) / np.sqrt(5))
        overall_paired_contrasts.append(
            {
                "contrast": name,
                "per_seed_mean_across_shots": per_seed,
                "mean_difference": mean_difference,
                "ci95_lower": mean_difference - margin,
                "ci95_upper": mean_difference + margin,
            }
        )

    lines = [
        "# Prompt Scale and Connectivity Analysis",
        "",
        "## Scope and test-status warning",
        "- This is a read-only controlled inference analysis. No models were retrained, no checkpoints were overwritten, and no optimizer steps were taken.",
        "- The test classes/episodes have already informed the previous diagnosis. Therefore, every test result here, including the selected-scale test value, is exploratory and post hoc, not an untouched independent confirmation.",
        "- Scaling was selected using validation classes [3, 4] only, before the test sweep. The single global scale maximizes the unweighted mean validation Macro-F1 across all five seeds and four shot settings; ties select the lower scale because candidates are ordered ascending.",
        "- Runtime includes model loading and both validation and test sweeps.",
        "",
        "## Executed command",
        "```powershell",
        "$py = \"c:/Users/User/Desktop/Papers to be submitted/LLMGPv0.2/implementation/.venv/Scripts/python.exe\"",
        "Set-Location \"C:/Users/User/Desktop/Papers to be submitted/LLMGPv0.2/implementation/LLMGP\"",
        "& $py experiments/run_prompt_scale_analysis.py",
        "```",
        f"Runtime: {runtime_seconds:.1f} seconds on {DEVICE}.",
        "",
        "## Control definitions",
        "- **NoPrompt:** original graph only; prompt nodes and prompt edges are absent.",
        "- **ZeroPrompt (scale 0.0):** the same two prompt nodes and support connections remain, but their input features are exactly zero. This is not equivalent to NoPrompt because topology, node degrees, self-loop normalization, and message passing change.",
        "- **OriginalPrompt (scale 1.0):** saved learned projection output and original prompt connectivity.",
        "- **ScaledPrompt:** the saved learned prompt features multiplied by the listed factor; all variants share the same support/query nodes and frozen checkpoints within each episode.",
        "- Validation: 200 reproducible paired episodes per seed/shot on classes [3,4]. Test: the original 600 paired episodes per seed/shot on classes [5,6], reconstructed and checked against saved query IDs/truth labels.",
        "- Accuracy and Macro-F1 confidence intervals are two-sided Student-t intervals over the five seed-level episode means (df=4; sample SD, ddof=1). Validation intervals describe selection data; test intervals are exploratory only.",
        "",
        "## Validation-only scale selection",
        "| Scale | Mean validation accuracy | Mean validation Macro-F1 |",
        "|---:|---:|---:|",
    ]
    for row in selection["validation_aggregate"]:
        lines.append(
            f"| {row['scale_factor']:.2f} | {row['mean_accuracy_over_seed_shot']:.4f} | {row['mean_macro_f1_over_seed_shot']:.4f} |"
        )
    lines.extend(
        [
            f"\nSelected global scale: **{selection['selected_scale']:.2f}**, based only on validation mean Macro-F1 {selection['selected_validation_mean_macro_f1']:.4f} (mean accuracy {selection['selected_validation_mean_accuracy']:.4f}).",
            "",
            "## Test scale sweep (exploratory)",
            "Each entry is mean across five seed means; 95% CI is across seeds, not episodes.",
            "| Shot | Scale | Accuracy (95% CI) | Macro-F1 (95% CI) | Prompt norm / node norm | Query delta L2 | Query cosine vs NoPrompt | Prompt degree | Norm edge-weight mean |",
            "|---:|---:|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for shot in SHOTS:
        for scale in (None, *SCALES):
            rows = [row for row in test_rows if int(row["shot"]) == shot and row["scale_factor"] == scale]
            if not rows:
                continue
            mean_accuracy = float(np.mean([row["accuracy"] for row in rows]))
            std_accuracy = float(np.std([row["accuracy"] for row in rows], ddof=1))
            accuracy_margin = float(t.ppf(0.975, 4) * std_accuracy / np.sqrt(5))
            mean_f1 = float(np.mean([row["macro_f1"] for row in rows]))
            std_f1 = float(np.std([row["macro_f1"] for row in rows], ddof=1))
            f1_margin = float(t.ppf(0.975, 4) * std_f1 / np.sqrt(5))
            variant = "NoPrompt" if scale is None else ("ZeroPrompt" if scale == 0.0 else ("OriginalPrompt" if scale == 1 else "ScaledPrompt"))
            lines.append(
                f"| {shot} | {variant} ({'--' if scale is None else f'{scale:.2f}'}) | {mean_accuracy:.4f} [{mean_accuracy-accuracy_margin:.4f}, {mean_accuracy+accuracy_margin:.4f}] | {mean_f1:.4f} [{mean_f1-f1_margin:.4f}, {mean_f1+f1_margin:.4f}] | {np.mean([r['prompt_to_original_feature_norm_ratio'] for r in rows]):.3f} | {np.mean([r['query_embedding_l2_change_vs_noprompt'] for r in rows]):.4f} | {np.mean([r['query_embedding_cosine_vs_noprompt'] for r in rows]):.5f} | {np.mean([r['prompt_node_degree_mean'] for r in rows]):.2f} | {np.mean([r['normalized_edge_weight_mean'] for r in rows]):.6f} |"
            )

    lines.extend(
        [
            "",
            "## Paired connectivity comparison",
            "| Shot | NoPrompt accuracy | ZeroPrompt accuracy | OriginalPrompt accuracy | Validation-selected scale test accuracy |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    chosen_scale = float(selection["selected_scale"])
    for shot in SHOTS:
        values: dict[str, float] = {}
        for scale, name in ((None, "NoPrompt"), (0.0, "ZeroPrompt"), (1.0, "OriginalPrompt"), (chosen_scale, "Selected")):
            rows = [row for row in test_rows if int(row["shot"]) == shot and row["scale_factor"] == scale]
            values[name] = float(np.mean([row["accuracy"] for row in rows]))
        lines.append(
            f"| {shot} | {values['NoPrompt']:.4f} | {values['ZeroPrompt']:.4f} | {values['OriginalPrompt']:.4f} | {values['Selected']:.4f} |"
        )

    lines.extend(
        [
            "",
            "### Paired accuracy contrasts (percentage points)",
            "Intervals are paired Student-t intervals across five seeds, using the same episode set within each comparison.",
            "| Shot | Contrast | Mean difference (pp) | 95% CI (pp) |",
            "|---:|---|---:|---:|",
        ]
    )
    for row in paired_contrasts:
        lines.append(
            f"| {row['shot']} | {row['contrast']} | {row['mean_difference']*100:.2f} | [{row['ci95_lower']*100:.2f}, {row['ci95_upper']*100:.2f}] |"
        )
    lines.extend(
        [
            "",
            "Across-shot paired contrasts average each seed's four shot-specific differences, then form a t interval across the five seed means:",
            "| Contrast | Mean difference (pp) | 95% CI (pp) |",
            "|---|---:|---:|",
        ]
    )
    for row in overall_paired_contrasts:
        lines.append(
            f"| {row['contrast']} | {row['mean_difference']*100:.2f} | [{row['ci95_lower']*100:.2f}, {row['ci95_upper']*100:.2f}] |"
        )
    lines.extend(
        [
            "",
            "## Findings",
            "- A difference between NoPrompt and ZeroPrompt isolates the topology/normalization effect at zero feature magnitude. Scale changes relative to ZeroPrompt then measure the additional effect of nonzero prompt features under the same connectivity.",
            "- Review the table above alongside `diagnostics/prompt_scale_results.csv`, which retains seed-level paired measurements, and `diagnostics/connectivity_comparison.csv`, which labels the connectivity variant explicitly.",
            "- **Hypothesis status: supported for prompt-feature magnitude, not for connectivity as the main driver.** Validation Macro-F1 declines monotonically from scale 0.0 to 1.0, and exploratory test accuracy drops strongly from ZeroPrompt to OriginalPrompt, especially at 5/10-shot. ZeroPrompt remains close to NoPrompt, with a much smaller paired difference. This is a controlled fixed-checkpoint association, not proof that scale alone causes the original cross-shot trend.",
            "- The validation-selected value 0.0 is **ZeroPrompt**, not NoPrompt: prompt nodes and edges remain, but prompt features are zero. The test set was already examined, so its selected-scale score is exploratory and not independent confirmation.",
            "- Feature scaling does not alter edge weights within a fixed connectivity graph; normalization statistics are constant across nonzero scales. They change between NoPrompt and prompt-connected graphs and as support count changes.",
            "",
            "## Recommended next experiment",
            "Run a newly preregistered study with a fresh held-out class split or independent dataset, selecting scale using train/validation only. Include trainable or validation-selected prompt-scale controls and separately trained NoPrompt/RandomPrompt baselines. Preserve nested support sets across K and report graph normalization and prompt-feature scale. Independently verify the Cora numeric-ID/topic-description alignment. Do not tune against the currently examined test classes.",
            "",
            "## Reproducibility outputs",
            "- `diagnostics/prompt_scale_results.csv`: split-, seed-, shot-, variant-, and scale-level metrics, confidence intervals, embedding shifts, and normalization values.",
            "- `diagnostics/connectivity_comparison.csv`: NoPrompt, ZeroPrompt, OriginalPrompt, and ScaledPrompt rows on the same paired episodes.",
            "- Original result CSVs/checkpoints and `reports/performance_diagnosis.md` were not overwritten.",
        ]
    )
    (ROOT / "reports" / "prompt_scale_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    started = time.perf_counter()
    torch.set_num_threads(1)
    info = verify_cora_dataset()
    labels = info["y"].cpu().numpy()
    validate_protocol(labels)
    encoder = SciBERTEncoder(model_name="allenai/scibert_scivocab_uncased", device=str(DEVICE))
    class_embeddings = encoder.encode_class_descriptions(
        [CLASS_DESCRIPTIONS[class_id] for class_id in range(7)],
        batch_size=7,
    ).to(DEVICE)
    if any(parameter.requires_grad for parameter in encoder.model.parameters()):
        raise AssertionError("SciBERT must remain frozen")

    DIAGNOSTIC_ROOT.mkdir(parents=True, exist_ok=True)
    validation_rows = run_sweep(
        "validation",
        info,
        labels,
        class_embeddings,
        VALIDATION_EPISODES,
        verify_saved_test=False,
    )
    selection = selected_scale_from_validation(validation_rows)
    (DIAGNOSTIC_ROOT / "validation_scale_selection.json").write_text(
        json.dumps(selection, indent=2),
        encoding="utf-8",
    )
    print(f"validation selected scale={selection['selected_scale']:.2f}", flush=True)

    test_rows = run_sweep(
        "test",
        info,
        labels,
        class_embeddings,
        TEST_EPISODES,
        verify_saved_test=True,
    )
    combined_rows = add_seed_confidence_intervals(validation_rows + test_rows)
    write_csv(DIAGNOSTIC_ROOT / "prompt_scale_results.csv", combined_rows)
    write_csv(DIAGNOSTIC_ROOT / "connectivity_comparison.csv", connectivity_rows(combined_rows))
    runtime = time.perf_counter() - started
    write_report(validation_rows, test_rows, selection, runtime)
    print(
        json.dumps(
            {
                "selected_validation_scale": selection["selected_scale"],
                "validation_rows": len(validation_rows),
                "test_rows": len(test_rows),
                "runtime_seconds": runtime,
                "outputs": [
                    "diagnostics/prompt_scale_results.csv",
                    "diagnostics/connectivity_comparison.csv",
                    "reports/prompt_scale_analysis.md",
                ],
                "test_status": "exploratory_post_hoc",
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
