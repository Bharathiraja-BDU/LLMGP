import numpy as np
import torch

from src.gnn_backbone import GCNBackbone
from src.prompt_builder import PromptBuilder
from src.revised_experiment import episode_loss_and_predictions, sample_split_episode


def test_training_episode_only_updates_prompt_projection():
    torch.manual_seed(7)
    node_count = 40
    feature_count = 16
    x = torch.rand(node_count, feature_count)
    forward_edges = torch.stack((torch.arange(node_count - 1), torch.arange(1, node_count)))
    edge_index = torch.cat((forward_edges, forward_edges.flip(0)), dim=1)
    backbone = GCNBackbone(feature_count, hidden_channels=8, out_channels=8)
    for parameter in backbone.parameters():
        parameter.requires_grad = False
    prompt_builder = PromptBuilder(in_dim=768, out_dim=feature_count)
    class_embeddings = torch.rand(7, 768)
    episode = {
        "class_ids": np.asarray([0, 1]),
        "support_idx": np.asarray([0, 20]),
        "query_idx": np.asarray(list(range(1, 16)) + list(range(21, 36))),
        "support_labels": np.asarray([0, 1]),
        "query_labels": np.asarray([0] * 15 + [1] * 15),
    }

    loss, logits, _ = episode_loss_and_predictions(
        episode,
        class_embeddings,
        prompt_builder,
        backbone,
        x,
        edge_index,
        torch.device("cpu"),
    )
    loss.backward()

    assert logits.shape == (30, 2)
    assert all(parameter.grad is None for parameter in backbone.parameters())
    assert prompt_builder.layer.weight.grad is not None
    assert prompt_builder.layer.weight.grad.abs().sum().item() > 0


def test_train_episode_is_invariant_to_test_label_changes():
    labels = np.repeat(np.arange(7), 20)
    changed_test_labels = labels.copy()
    changed_test_labels[np.isin(labels, [5, 6])] = 99

    episode_a = sample_split_episode(labels, "train", support_shots=3, seed=113)
    episode_b = sample_split_episode(changed_test_labels, "train", support_shots=3, seed=113)

    for key in ("class_ids", "support_idx", "query_idx", "support_labels", "query_labels"):
        np.testing.assert_array_equal(episode_a[key], episode_b[key])
    assert set(episode_a["class_ids"].tolist()).issubset({0, 1, 2})