from __future__ import annotations

import copy
import csv
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import t
from sklearn.metrics import f1_score
from torch.optim import Adam

from .classifier import ProtoClassifier
from .episode_sampler import EpisodeSampler
from .graph_augmenter import GraphAugmenter
from .gnn_backbone import GCNBackbone
from .prompt_builder import PromptBuilder
from .revised_protocol import (
    CLASS_DESCRIPTIONS,
    CLASS_ID_TO_TOPIC,
    CLASS_SPLIT,
    PROTOCOL_NAME,
    PROTOCOL_SEED,
    labels_visible_in_split,
    remap_episode_labels,
)


QUERY_SHOTS = 15
TEMPERATURE = 0.1


def sample_split_episode(
    labels: np.ndarray,
    split_name: str,
    support_shots: int,
    seed: int,
    query_shots: int = QUERY_SHOTS,
) -> dict[str, Any]:
    class_pool = CLASS_SPLIT[split_name]
    rng = np.random.default_rng(seed)
    if len(class_pool) == 2:
        class_ids = list(class_pool)
    else:
        class_ids = [int(value) for value in rng.choice(class_pool, size=2, replace=False)]
    visible_labels = labels_visible_in_split(labels, class_pool)
    sampler = EpisodeSampler(num_classes=2, support_shots=support_shots, query_shots=query_shots)
    episode = sampler.sample_episode(
        labels=visible_labels,
        class_ids=class_ids,
        seed=seed ^ 0x5DEECE66D,
    )
    episode = remap_episode_labels(episode)
    if not set(episode["support_idx"].tolist()).isdisjoint(episode["query_idx"].tolist()):
        raise AssertionError("Support/query node overlap detected")
    return episode


def episode_loss_and_predictions(
    episode: dict[str, Any],
    class_embeddings: torch.Tensor,
    prompt_builder: PromptBuilder,
    backbone: GCNBackbone,
    data_x: torch.Tensor,
    edge_index: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    class_ids = torch.as_tensor(episode["class_ids"], dtype=torch.long, device=device)
    support_idx = torch.as_tensor(episode["support_idx"], dtype=torch.long, device=device)
    query_idx = torch.as_tensor(episode["query_idx"], dtype=torch.long, device=device)
    support_labels = torch.as_tensor(episode["support_labels"], dtype=torch.long, device=device)
    query_labels = torch.as_tensor(episode["query_labels"], dtype=torch.long, device=device)

    prompt_nodes = prompt_builder(class_embeddings[class_ids])
    augmented = GraphAugmenter(num_classes=2).augment(
        data_x,
        edge_index,
        support_idx,
        support_labels,
        prompt_embeddings=prompt_nodes,
    )
    node_embeddings = backbone(augmented.x, augmented.edge_index)
    original_embeddings = node_embeddings[augmented.original_mask]
    support_embeddings = original_embeddings[support_idx]
    query_embeddings = original_embeddings[query_idx]
    prototypes = ProtoClassifier.prototypes(support_embeddings, support_labels, 2)
    logits = ProtoClassifier.logits(query_embeddings, prototypes, temperature=TEMPERATURE)
    loss = F.cross_entropy(logits, query_labels)
    return loss, logits, query_labels


def train_prompt_model(
    *,
    data: Any,
    labels: np.ndarray,
    class_embeddings: torch.Tensor,
    backbone: GCNBackbone,
    shot: int,
    seed: int,
    checkpoint_path: Path,
    log_path: Path,
    max_epochs: int = 50,
    episodes_per_epoch: int = 100,
    validation_episodes: int = 30,
    patience: int = 10,
    learning_rate: float = 0.001,
    weight_decay: float = 0.0001,
) -> tuple[PromptBuilder, dict[str, Any], float]:
    device = next(backbone.parameters()).device
    torch.manual_seed(seed + shot * 100_003)
    prompt_builder = PromptBuilder().to(device)
    optimizer = Adam(prompt_builder.parameters(), lr=learning_rate, weight_decay=weight_decay)
    backbone.eval()
    for parameter in backbone.parameters():
        parameter.requires_grad = False
        parameter.grad = None

    data_x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    class_embeddings = class_embeddings.to(device)
    train_history: list[dict[str, Any]] = []
    best_val_loss = float("inf")
    best_val_accuracy = 0.0
    best_epoch = 0
    epochs_without_improvement = 0
    best_state: dict[str, torch.Tensor] | None = None
    started = time.perf_counter()

    for epoch in range(max_epochs):
        prompt_builder.train()
        train_losses: list[float] = []
        for episode_index in range(episodes_per_epoch):
            episode_seed = seed * 10_000_000 + shot * 100_000 + epoch * episodes_per_epoch + episode_index
            episode = sample_split_episode(labels, "train", shot, episode_seed)
            loss, _, _ = episode_loss_and_predictions(
                episode, class_embeddings, prompt_builder, backbone, data_x, edge_index, device
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if any(parameter.grad is not None for parameter in backbone.parameters()):
                raise AssertionError("Frozen GNN received gradients during prompt training")
            if not all(parameter.grad is not None for parameter in prompt_builder.parameters()):
                raise AssertionError("Prompt projection did not receive gradients")
            optimizer.step()
            train_losses.append(float(loss.detach().item()))

        prompt_builder.eval()
        validation_losses: list[float] = []
        validation_accuracies: list[float] = []
        with torch.no_grad():
            for episode_index in range(validation_episodes):
                episode_seed = seed * 10_000_000 + 8_000_000 + episode_index
                episode = sample_split_episode(labels, "validation", shot, episode_seed)
                loss, logits, query_labels = episode_loss_and_predictions(
                    episode, class_embeddings, prompt_builder, backbone, data_x, edge_index, device
                )
                validation_losses.append(float(loss.item()))
                validation_accuracies.append(float((logits.argmax(dim=-1) == query_labels).float().mean().item()))

        mean_train_loss = float(np.mean(train_losses))
        mean_val_loss = float(np.mean(validation_losses))
        mean_val_accuracy = float(np.mean(validation_accuracies))
        train_history.append(
            {
                "epoch": epoch + 1,
                "train_loss": mean_train_loss,
                "validation_loss": mean_val_loss,
                "validation_accuracy": mean_val_accuracy,
            }
        )

        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            best_val_accuracy = mean_val_accuracy
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            best_state = {key: value.detach().cpu().clone() for key, value in prompt_builder.state_dict().items()}
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "prompt_state_dict": best_state,
                    "metadata": {
                        "protocol": PROTOCOL_NAME,
                        "class_split": {key: list(value) for key, value in CLASS_SPLIT.items()},
                        "seed": seed,
                        "shot": shot,
                        "best_epoch": best_epoch,
                        "best_validation_loss": best_val_loss,
                        "best_validation_accuracy": best_val_accuracy,
                        "backbone_checkpoint": str(checkpoint_path.parent / f"seed_{seed}_frozen_gcn.pt"),
                        "optimizer": "Adam",
                        "learning_rate": learning_rate,
                        "weight_decay": weight_decay,
                        "temperature": TEMPERATURE,
                        "episodes_per_epoch": episodes_per_epoch,
                        "validation_episodes": validation_episodes,
                        "max_epochs": max_epochs,
                        "patience": patience,
                    },
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    if best_state is None:
        raise RuntimeError("No validation checkpoint was selected")
    prompt_builder.load_state_dict(best_state)
    prompt_builder.eval()
    elapsed = time.perf_counter() - started

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(train_history[0]))
        writer.writeheader()
        writer.writerows(train_history)

    training_summary = {
        "best_epoch": best_epoch,
        "best_validation_loss": best_val_loss,
        "best_validation_accuracy": best_val_accuracy,
        "epochs_completed": len(train_history),
        "training_seconds": elapsed,
        "history": train_history,
    }
    return prompt_builder, training_summary, elapsed


def evaluate_prompt_model(
    *,
    data: Any,
    labels: np.ndarray,
    class_embeddings: torch.Tensor,
    backbone: GCNBackbone,
    prompt_builder: PromptBuilder,
    shot: int,
    seed: int,
    episodes: int,
    predictions_path: Path,
    episode_metrics_path: Path,
) -> dict[str, Any]:
    device = next(backbone.parameters()).device
    data_x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    class_embeddings = class_embeddings.to(device)
    backbone.eval()
    prompt_builder.eval()
    before = {key: value.detach().clone() for key, value in prompt_builder.state_dict().items()}
    episode_results: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []

    with torch.no_grad():
        for episode_index in range(episodes):
            episode_seed = seed * 10_000_000 + 9_000_000 + shot * 100_000 + episode_index
            episode = sample_split_episode(labels, "test", shot, episode_seed)
            loss, logits, query_labels = episode_loss_and_predictions(
                episode, class_embeddings, prompt_builder, backbone, data_x, edge_index, device
            )
            probabilities = logits.softmax(dim=-1)
            predictions = logits.argmax(dim=-1)
            local_classes = [int(value) for value in episode["class_ids"]]
            true_values = query_labels.cpu().numpy()
            predicted_values = predictions.cpu().numpy()
            accuracy = float(np.mean(true_values == predicted_values))
            macro_f1 = float(f1_score(true_values, predicted_values, labels=[0, 1], average="macro", zero_division=0))
            episode_results.append(
                {
                    "seed": seed,
                    "shot": shot,
                    "episode": episode_index,
                    "class_ids": ",".join(map(str, local_classes)),
                    "loss": float(loss.item()),
                    "accuracy": accuracy,
                    "macro_f1": macro_f1,
                    "support_query_overlap": False,
                }
            )
            for query_offset, (query_index, true_local, predicted_local) in enumerate(
                zip(episode["query_idx"], true_values, predicted_values)
            ):
                prediction_rows.append(
                    {
                        "seed": seed,
                        "shot": shot,
                        "episode": episode_index,
                        "query_node_id": int(query_index),
                        "true_class_id": local_classes[int(true_local)],
                        "predicted_class_id": local_classes[int(predicted_local)],
                        "probability_first_episode_class": float(probabilities[query_offset, 0].item()),
                        "probability_second_episode_class": float(probabilities[query_offset, 1].item()),
                    }
                )

    for key, value in prompt_builder.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    if any(parameter.grad is not None for parameter in backbone.parameters()):
        raise AssertionError("Frozen GNN accumulated gradients during evaluation")

    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    append_csv_rows(predictions_path, prediction_rows)
    append_csv_rows(episode_metrics_path, episode_results)
    return {
        "episodes": episodes,
        "accuracy_mean": float(np.mean([row["accuracy"] for row in episode_results])),
        "accuracy_std_episode": float(np.std([row["accuracy"] for row in episode_results], ddof=1)),
        "macro_f1_mean": float(np.mean([row["macro_f1"] for row in episode_results])),
        "macro_f1_std_episode": float(np.std([row["macro_f1"] for row in episode_results], ddof=1)),
    }


def append_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def confidence_interval(values: list[float], confidence: float = 0.95) -> tuple[float, float, float, float]:
    if len(values) < 2:
        raise ValueError("At least two independent seed values are required for a confidence interval")
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1))
    critical_value = float(t.ppf((1.0 + confidence) / 2.0, df=len(values) - 1))
    margin = critical_value * std / math.sqrt(len(values))
    return mean, std, mean - margin, mean + margin


def load_frozen_backbone(checkpoint_path: Path, in_channels: int = 1433, hidden_dim: int = 256) -> GCNBackbone:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    backbone = GCNBackbone(in_channels=in_channels, hidden_channels=hidden_dim, out_channels=hidden_dim)
    backbone.load_state_dict(checkpoint["model_state_dict"])
    backbone.eval()
    for parameter in backbone.parameters():
        parameter.requires_grad = False
    return backbone