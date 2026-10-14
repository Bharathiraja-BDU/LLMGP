from types import SimpleNamespace

import torch

from src.pretrain_gnn import pretrain_backbone


def test_pretraining_is_invariant_to_node_labels():
    torch.manual_seed(13)
    x = torch.rand(24, 8)
    edge_index = torch.tensor(
        [[index for index in range(23)], [index + 1 for index in range(23)]],
        dtype=torch.long,
    )
    labels_a = torch.arange(24) % 3
    labels_b = torch.randperm(24) % 7
    data_a = SimpleNamespace(x=x, edge_index=edge_index, y=labels_a)
    data_b = SimpleNamespace(x=x, edge_index=edge_index, y=labels_b)

    model_a, loss_a = pretrain_backbone(
        data_a,
        epochs=3,
        hidden_dim=6,
        seed=29,
        device="cpu",
    )
    model_b, loss_b = pretrain_backbone(
        data_b,
        epochs=3,
        hidden_dim=6,
        seed=29,
        device="cpu",
    )

    assert loss_a == loss_b
    for parameter_a, parameter_b in zip(model_a.parameters(), model_b.parameters()):
        torch.testing.assert_close(parameter_a, parameter_b, rtol=0, atol=0)
    assert all(not parameter.requires_grad for parameter in model_a.parameters())