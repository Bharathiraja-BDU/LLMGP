import torch

from src.graph_augmenter import GraphAugmenter


def test_graph_augmentation_shapes():
    x = torch.randn(10, 1433)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    support_idx = torch.tensor([0, 2])
    labels = torch.tensor([0, 1])
    augmenter = GraphAugmenter(num_classes=2)
    aug = augmenter.augment(x, edge_index, support_idx, labels)
    assert aug.x.shape[0] == 12
    assert aug.edge_index.shape[0] == 2
    assert aug.original_mask.shape[0] == 12
    assert aug.prompt_mask.shape[0] == 12
