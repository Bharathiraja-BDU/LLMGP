from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.data import Data

from .gnn_backbone import GCNBackbone


def pretrain_backbone(
    data: Data,
    epochs: int = 100,
    lr: float = 0.001,
    weight_decay: float = 0.0001,
    hidden_dim: int = 256,
    seed: int = 2026,
    mask_probability: float = 0.15,
    checkpoint_path: str | Path | None = None,
    device: str | None = None,
) -> tuple[GCNBackbone, float]:
    """Pretrain the GCN by reconstructing randomly masked nonzero node features.

    The objective reads only ``data.x`` and ``data.edge_index``. Node labels are
    neither consumed nor used for model selection. All nodes and graph edges are
    visible in this transductive self-supervised pretraining stage.
    """
    if epochs < 1:
        raise ValueError("epochs must be positive")
    if not 0.0 < mask_probability < 1.0:
        raise ValueError("mask_probability must be between zero and one")

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    selected_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    x = data.x.to(selected_device)
    edge_index = data.edge_index.to(selected_device)
    model = GCNBackbone(in_channels=x.shape[1], hidden_channels=hidden_dim, out_channels=hidden_dim).to(selected_device)
    decoder = nn.Linear(hidden_dim, x.shape[1]).to(selected_device)
    optimizer = Adam(
        [*model.parameters(), *decoder.parameters()],
        lr=lr,
        weight_decay=weight_decay,
    )
    generator = torch.Generator(device=selected_device).manual_seed(seed)
    final_loss = float("nan")
    loss_history: list[float] = []

    for _ in range(epochs):
        model.train()
        decoder.train()
        mask = (x > 0) & (torch.rand(x.shape, device=selected_device, generator=generator) < mask_probability)
        if not mask.any():
            raise RuntimeError("Feature masking selected no observed entries")
        corrupted_x = x.masked_fill(mask, 0.0)
        reconstructed = decoder(model(corrupted_x, edge_index))
        loss = F.mse_loss(reconstructed[mask], x[mask])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().item())
        loss_history.append(final_loss)

    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False

    metadata: dict[str, Any] = {
        "objective": "masked_feature_reconstruction_mse",
        "mask_probability": mask_probability,
        "epochs": epochs,
        "learning_rate": lr,
        "weight_decay": weight_decay,
        "hidden_dim": hidden_dim,
        "seed": seed,
        "graph_visibility": "all node features and graph edges; no node labels",
        "final_loss": final_loss,
        "loss_history": loss_history,
    }
    if checkpoint_path is not None:
        checkpoint = Path(checkpoint_path)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "decoder_state_dict": decoder.state_dict(),
                "pretraining": metadata,
            },
            checkpoint,
        )

    return model, final_loss
