from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

from ..classifier import ProtoClassifier
from ..graph_augmenter import GraphAugmenter
from ..revised_protocol import CLASS_SPLIT, PROTOCOL_SEED, labels_visible_in_split, remap_episode_labels
from ..episode_sampler import EpisodeSampler
from .prompt_models import NormalizedGatedPrompt, NormalizedPrompt, OriginalPrompt


MODEL_A = "OriginalPrompt"
MODEL_B = "NormalizedPrompt"
MODEL_C = "NormalizedGatedPrompt"
SEMANTIC_VARIANTS = ("configured", "shuffled", "random")
QUERY_SHOTS = 15
TEMPERATURE = 0.1


def build_prompt_model(model_name: str, reference_node_norm: float) -> torch.nn.Module:
    if model_name == MODEL_A:
        return OriginalPrompt()
    if model_name == MODEL_B:
        return NormalizedPrompt(reference_node_norm=reference_node_norm)
    if model_name == MODEL_C:
        return NormalizedGatedPrompt(reference_node_norm=reference_node_norm, initial_gate=0.1)
    raise ValueError(f"Unknown v2 prompt model: {model_name}")


def class_embedding_variant(
    class_embeddings: torch.Tensor,
    variant: str,
    seed: int,
) -> tuple[torch.Tensor, list[int] | None]:
    if variant == "configured":
        return class_embeddings, None
    if variant == "shuffled":
        available_ids = torch.nonzero(class_embeddings.norm(dim=-1) > 0, as_tuple=False).flatten()
        local_permutation = torch.randperm(
            available_ids.numel(),
            generator=torch.Generator(device=class_embeddings.device).manual_seed(seed),
        )
        permutation = available_ids[local_permutation].tolist()
        if permutation == list(range(len(permutation))):
            permutation = permutation[1:] + permutation[:1]
        shuffled = class_embeddings.clone()
        shuffled[available_ids] = class_embeddings[permutation]
        return shuffled, permutation
    if variant == "random":
        available_ids = torch.nonzero(class_embeddings.norm(dim=-1) > 0, as_tuple=False).flatten()
        generator = torch.Generator(device=class_embeddings.device).manual_seed(seed + 77_777)
        random_embeddings = class_embeddings.clone()
        generated = torch.randn(
            (available_ids.numel(), class_embeddings.shape[1]),
            generator=generator,
            device=class_embeddings.device,
            dtype=class_embeddings.dtype,
        )
        generated = F.normalize(generated, dim=-1) * class_embeddings[available_ids].norm(dim=-1, keepdim=True)
        random_embeddings[available_ids] = generated
        return random_embeddings, None
    raise ValueError(f"Unknown semantic variant: {variant}")


def sample_episode(
    labels: np.ndarray,
    split_name: str,
    shot: int,
    seed: int,
    query_shots: int = QUERY_SHOTS,
) -> dict[str, Any]:
    split_classes = CLASS_SPLIT[split_name]
    if len(split_classes) == 2:
        class_ids = list(split_classes)
    else:
        class_rng = np.random.default_rng(seed ^ 0xA5A5A5A5)
        class_ids = [int(value) for value in class_rng.choice(split_classes, size=2, replace=False)]
    visible_labels = labels_visible_in_split(labels, split_classes)
    sampler = EpisodeSampler(num_classes=2, support_shots=shot, query_shots=query_shots)
    episode = sampler.sample_episode(labels=visible_labels, class_ids=class_ids, seed=seed)
    episode = remap_episode_labels(episode)
    support = set(episode["support_idx"].tolist())
    query = set(episode["query_idx"].tolist())
    if not support.isdisjoint(query):
        raise AssertionError("Support and query nodes overlap")
    return episode


def make_fixed_episode_sets(
    labels: np.ndarray,
    shots: tuple[int, ...],
    validation_episodes: int,
    seeds: tuple[int, ...],
) -> dict[tuple[str, int, int], list[dict[str, Any]]]:
    """Pre-generate exactly shared episodes for every model/semantic arm."""
    episodes: dict[tuple[str, int, int], list[dict[str, Any]]] = {}
    for split_name, count in (("train", 0), ("validation", validation_episodes)):
        if split_name == "train":
            continue
        for shot in shots:
            for seed in seeds:
                episode_seed = PROTOCOL_SEED * 10_000_000 + seed * 10_000 + shot * 100_000
                episodes[(split_name, shot, seed)] = [
                    sample_episode(labels, split_name, shot, episode_seed + index)
                    for index in range(count)
                ]
    return episodes


def episode_forward(
    episode: dict[str, Any],
    x: torch.Tensor,
    edge_index: torch.Tensor,
    class_embeddings: torch.Tensor,
    prompt_model: torch.nn.Module,
    backbone: torch.nn.Module,
    prompt_mode: str = "connected",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    device = x.device
    support_idx = torch.as_tensor(episode["support_idx"], dtype=torch.long, device=device)
    query_idx = torch.as_tensor(episode["query_idx"], dtype=torch.long, device=device)
    support_labels = torch.as_tensor(episode["support_labels"], dtype=torch.long, device=device)
    query_labels = torch.as_tensor(episode["query_labels"], dtype=torch.long, device=device)
    class_ids = torch.as_tensor(episode["class_ids"], dtype=torch.long, device=device)
    prompt_features = prompt_model(class_embeddings[class_ids])

    if prompt_mode == "none":
        graph_x = x
        graph_edges = edge_index
        original_mask = None
        prompt_features = prompt_features.new_empty((0, x.shape[1]))
    elif prompt_mode == "zero":
        prompt_features = torch.zeros_like(prompt_features)
        augmented = GraphAugmenter(num_classes=2).augment(
            x,
            edge_index,
            support_idx,
            support_labels,
            prompt_embeddings=prompt_features,
        )
        graph_x, graph_edges, original_mask = augmented.x, augmented.edge_index, augmented.original_mask
    elif prompt_mode == "connected":
        augmented = GraphAugmenter(num_classes=2).augment(
            x,
            edge_index,
            support_idx,
            support_labels,
            prompt_embeddings=prompt_features,
        )
        graph_x, graph_edges, original_mask = augmented.x, augmented.edge_index, augmented.original_mask
    else:
        raise ValueError(f"Unknown prompt mode: {prompt_mode}")

    node_embeddings = backbone(graph_x, graph_edges)
    if original_mask is not None:
        node_embeddings = node_embeddings[original_mask]
    support_embeddings = node_embeddings[support_idx]
    query_embeddings = node_embeddings[query_idx]
    prototypes = ProtoClassifier.prototypes(support_embeddings, support_labels, 2)
    logits = ProtoClassifier.logits(query_embeddings, prototypes, temperature=TEMPERATURE)
    loss = F.cross_entropy(logits, query_labels)
    return loss, logits, query_labels, prompt_features


def evaluate_episodes(
    episodes: list[dict[str, Any]],
    x: torch.Tensor,
    edge_index: torch.Tensor,
    class_embeddings: torch.Tensor,
    prompt_model: torch.nn.Module | None,
    backbone: torch.nn.Module,
    prompt_mode: str,
) -> dict[str, Any]:
    losses: list[float] = []
    accuracies: list[float] = []
    macro_f1s: list[float] = []
    prompt_norms: list[float] = []
    gate_values: list[float] = []
    model = prompt_model or OriginalPrompt().to(x.device)
    model.eval()
    backbone.eval()
    with torch.no_grad():
        for episode in episodes:
            loss, logits, labels, prompt_features = episode_forward(
                episode,
                x,
                edge_index,
                class_embeddings,
                model,
                backbone,
                prompt_mode=prompt_mode,
            )
            predictions = logits.argmax(dim=-1)
            labels_np = labels.cpu().numpy()
            predictions_np = predictions.cpu().numpy()
            losses.append(float(loss.item()))
            accuracies.append(float((predictions == labels).float().mean().item()))
            macro_f1s.append(float(f1_score(labels_np, predictions_np, labels=[0, 1], average="macro", zero_division=0)))
            prompt_norms.append(
                float(prompt_features.norm(dim=-1).mean().item()) if prompt_features.numel() else 0.0
            )
            if isinstance(model, NormalizedGatedPrompt):
                gate_values.append(float(model.gate.item()))
    return {
        "loss": float(np.mean(losses)),
        "accuracy": float(np.mean(accuracies)),
        "macro_f1": float(np.mean(macro_f1s)),
        "effective_prompt_norm": float(np.mean(prompt_norms)),
        "gate": float(np.mean(gate_values)) if gate_values else None,
        "episodes": len(episodes),
    }


def snapshot_v1_artifacts(project_root: Path) -> dict[str, str]:
    patterns = (
        "src/*.py",
        "experiments/*.py",
        "tests/*.py",
        "data/**/*",
        "configs/*",
        "results/checkpoints/**/*",
        "results/metrics/**/*",
        "results/diagnostics/**/*",
        "results/logs/**/*",
        "results/figures/**/*",
        "diagnostics/**/*",
        "reports/*",
    )
    excluded = {
        "experiments/run_llmgp_v2_validation.py",
        "experiments/run_v2_mvp.py",
        "experiments/run_v2_extended_validation.py",
        "experiments/run_v2_semantic_alignment_audit.py",
        "experiments/run_v2_semantic_contribution.py",
        "reports/llmgp_v2_validation_report.md",
        "reports/v2_mvp_report.md",
        "reports/v2_extended_validation.md",
        "reports/semantic_alignment_audit.md",
        "reports/v2_semantic_contribution.md",
    }
    files: set[Path] = set()
    for pattern in patterns:
        files.update(path for path in project_root.glob(pattern) if path.is_file())
    manifest: dict[str, str] = {}
    for path in sorted(files):
        relative = path.relative_to(project_root).as_posix()
        if relative.startswith("src/v2/") or relative.startswith("tests/v2/"):
            continue
        if relative.startswith("results/v2/") or relative.startswith("experiments/run_llmgp_v2_validation.py"):
            continue
        if relative in excluded:
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest[relative] = digest

    manuscript = project_root.parent / "LLMGP_v0.2.docx"
    if manuscript.exists():
        manifest["../LLMGP_v0.2.docx"] = hashlib.sha256(manuscript.read_bytes()).hexdigest()
    return manifest


def verify_frozen_backbone(backbone: torch.nn.Module) -> None:
    backbone.eval()
    for parameter in backbone.parameters():
        parameter.requires_grad = False
        parameter.grad = None


def validation_checkpoint_key(metrics: dict[str, Any]) -> tuple[float, float, float]:
    return (float(metrics["macro_f1"]), float(metrics["accuracy"]), -float(metrics["loss"]))


def validate_trainable_gradients(module: torch.nn.Module) -> bool:
    """Check gradient integrity; return False when an episode has no prompt signal."""
    gradients: list[torch.Tensor] = []
    for parameter in module.parameters():
        if parameter.grad is None:
            raise AssertionError("Missing gradient for a trainable v2 parameter")
        if not torch.isfinite(parameter.grad).all():
            raise AssertionError("Non-finite gradient for a trainable v2 parameter")
        gradients.append(parameter.grad)
    return any(float(gradient.norm().item()) > 0.0 for gradient in gradients)
