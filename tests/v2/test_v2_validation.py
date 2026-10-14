import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from src.classifier import ProtoClassifier
from src.graph_augmenter import GraphAugmenter
from src.gnn_backbone import GCNBackbone
from src.revised_protocol import CLASS_SPLIT, labels_visible_in_split
from src.v2.prompt_models import NormalizedGatedPrompt, NormalizedPrompt, OriginalPrompt
from src.v2.validation import (
    class_embedding_variant,
    episode_forward,
    evaluate_episodes,
    sample_episode,
    snapshot_v1_artifacts,
    validate_trainable_gradients,
)


def test_prompt_model_output_shapes_and_norm_control():
    torch.manual_seed(5)
    embeddings = torch.randn(3, 768)
    reference_norm = 6.25

    original = OriginalPrompt()(embeddings)
    normalized = NormalizedPrompt(reference_node_norm=reference_norm)(embeddings)
    gated_model = NormalizedGatedPrompt(reference_node_norm=reference_norm, initial_gate=0.08)
    gated = gated_model(embeddings)

    assert original.shape == normalized.shape == gated.shape == (3, 1433)
    assert torch.isfinite(original).all()
    assert torch.isfinite(normalized).all()
    assert torch.isfinite(gated).all()
    torch.testing.assert_close(normalized.norm(dim=-1), torch.full((3,), reference_norm), atol=1e-5, rtol=0)
    torch.testing.assert_close(gated.norm(dim=-1), torch.full((3,), reference_norm * 0.08), atol=1e-5, rtol=0)
    assert 0 < gated_model.gate.item() < 1


def test_projection_and_gate_gradients_flow_through_frozen_gcn():
    torch.manual_seed(17)
    node_count = 40
    x = torch.rand(node_count, 1433)
    forward_edges = torch.stack((torch.arange(node_count - 1), torch.arange(1, node_count)))
    edge_index = torch.cat((forward_edges, forward_edges.flip(0)), dim=1)
    backbone = GCNBackbone(1433, hidden_channels=16, out_channels=16)
    for parameter in backbone.parameters():
        parameter.requires_grad = False
    prompt_model = NormalizedGatedPrompt(reference_node_norm=4.0, initial_gate=0.1)
    class_embeddings = torch.randn(7, 768)
    episode = {
        "class_ids": np.asarray([0, 1]),
        "support_idx": np.asarray([0, 20]),
        "query_idx": np.asarray(list(range(1, 16)) + list(range(21, 36))),
        "support_labels": np.asarray([0, 1]),
        "query_labels": np.asarray([0] * 15 + [1] * 15),
    }

    loss, logits, _, effective_prompts = episode_forward(
        episode,
        x,
        edge_index,
        class_embeddings,
        prompt_model,
        backbone,
    )
    loss.backward()

    assert logits.shape == (30, 2)
    assert torch.isfinite(effective_prompts).all()
    assert prompt_model.projection.weight.grad is not None
    assert torch.isfinite(prompt_model.projection.weight.grad).all()
    assert prompt_model.projection.weight.grad.norm().item() > 0
    assert prompt_model.gate_logit.grad is not None
    assert torch.isfinite(prompt_model.gate_logit.grad).all()
    assert prompt_model.gate_logit.grad.abs().item() > 0
    assert all(parameter.grad is None for parameter in backbone.parameters())


def test_split_sampling_and_heldout_label_isolation():
    labels = np.repeat(np.arange(7), 50)
    changed_test_labels = labels.copy()
    changed_test_labels[np.isin(labels, CLASS_SPLIT["test"])] = 99

    train_episode = sample_episode(labels, "train", shot=3, seed=91)
    train_episode_changed = sample_episode(changed_test_labels, "train", shot=3, seed=91)
    validation_episode = sample_episode(labels, "validation", shot=3, seed=92)

    assert set(train_episode["class_ids"].tolist()).issubset(CLASS_SPLIT["train"])
    assert train_episode["class_ids"].tolist() == train_episode_changed["class_ids"].tolist()
    assert train_episode["support_idx"].tolist() == train_episode_changed["support_idx"].tolist()
    assert train_episode["query_idx"].tolist() == train_episode_changed["query_idx"].tolist()
    assert validation_episode["class_ids"].tolist() == list(CLASS_SPLIT["validation"])
    assert set(train_episode["support_idx"].tolist()).isdisjoint(train_episode["query_idx"].tolist())
    assert set(validation_episode["support_idx"].tolist()).isdisjoint(validation_episode["query_idx"].tolist())

    visible = labels_visible_in_split(labels, CLASS_SPLIT["validation"])
    assert set(np.unique(visible[visible >= 0]).tolist()) == set(CLASS_SPLIT["validation"])


def test_semantic_ablation_never_constructs_test_class_vectors():
    torch.manual_seed(22)
    permitted = torch.randn(5, 768)
    class_embeddings = torch.zeros(7, 768)
    class_embeddings[:5] = permitted
    for variant in ("shuffled", "random"):
        changed, assignment = class_embedding_variant(class_embeddings, variant, seed=44)
        assert changed[5:].count_nonzero().item() == 0
        assert torch.isfinite(changed).all()
        assert changed[:5].norm(dim=-1).min().item() > 0
        if assignment is not None:
            assert sorted(assignment) == [0, 1, 2, 3, 4]


def test_reference_control_prompt_norms_match_applied_features():
    class_embeddings = torch.randn(7, 768)
    prompt_model = OriginalPrompt()
    backbone = GCNBackbone(1433, hidden_channels=8, out_channels=8)
    for parameter in backbone.parameters():
        parameter.requires_grad = False
    node_count = 16
    x = torch.rand(node_count, 1433)
    forward_edges = torch.stack((torch.arange(node_count - 1), torch.arange(1, node_count)))
    edge_index = torch.cat((forward_edges, forward_edges.flip(0)), dim=1)
    episode = {
        "class_ids": np.asarray([0, 1]),
        "support_idx": np.asarray([0, 8]),
        "query_idx": np.asarray(list(range(1, 8)) + list(range(9, 16))),
        "support_labels": np.asarray([0, 1]),
        "query_labels": np.asarray([0] * 7 + [1] * 7),
    }
    none = episode_forward(episode, x, edge_index, class_embeddings, prompt_model, backbone, "none")
    zero = episode_forward(episode, x, edge_index, class_embeddings, prompt_model, backbone, "zero")
    assert none[3].numel() == 0
    assert zero[3].count_nonzero().item() == 0


def test_no_prompt_validation_norm_is_zero_not_nan():
    torch.manual_seed(3)
    labels = np.repeat(np.arange(7), 100)
    episode = sample_episode(labels, "validation", shot=1, seed=3)
    node_count = len(labels)
    x = torch.rand(node_count, 12)
    edges = torch.tensor(
        [
            [i for i in range(node_count - 1)] + [i + 1 for i in range(node_count - 1)],
            [i + 1 for i in range(node_count - 1)] + [i for i in range(node_count - 1)],
        ]
    )
    backbone = GCNBackbone(12, hidden_channels=8, out_channels=8)
    for parameter in backbone.parameters():
        parameter.requires_grad = False
    metrics = evaluate_episodes(
        [episode],
        x,
        edges,
        torch.rand(7, 768),
        prompt_model=None,
        backbone=backbone,
        prompt_mode="none",
    )
    assert metrics["effective_prompt_norm"] == 0.0


def test_zero_prompt_gradient_signal_is_detected_without_weight_update():
    layer = torch.nn.Linear(2, 1)
    layer.weight.grad = torch.zeros_like(layer.weight)
    layer.bias.grad = torch.zeros_like(layer.bias)
    assert not validate_trainable_gradients(layer)
    layer.bias.grad = torch.ones_like(layer.bias)
    assert validate_trainable_gradients(layer)


def test_v1_artifact_snapshot_unchanged_when_available():
    project_root = Path(__file__).resolve().parents[2]
    snapshot_path = project_root / "results" / "v2" / "v1_artifact_manifest_before.json"
    if not snapshot_path.exists():
        pytest.skip("V1 snapshot is written by the validation runner before its test gate")
    before = json.loads(snapshot_path.read_text(encoding="utf-8"))
    after = snapshot_v1_artifacts(project_root)
    assert before == after
