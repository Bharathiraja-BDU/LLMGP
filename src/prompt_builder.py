from __future__ import annotations

import torch
from torch import nn


class PromptBuilder(nn.Module):
    """Project 768-dim class semantics into the 1,433-dim Cora input space."""

    def __init__(self, in_dim: int = 768, out_dim: int = 1433):
        super().__init__()
        self.layer = nn.Linear(in_dim, out_dim, bias=True)
        nn.init.xavier_uniform_(self.layer.weight)
        self.in_dim = in_dim
        self.out_dim = out_dim

    def forward(self, class_embeddings: torch.Tensor) -> torch.Tensor:
        return self.layer(class_embeddings)
