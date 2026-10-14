from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class SemanticPrototypeFusion(nn.Module):
    """Fuse support prototypes with projected frozen class descriptions.

    Inputs have matching class order. Frozen encoders remain external; only
    the semantic projection and a shared sigmoid gate are trainable here.
    """

    def __init__(
        self, semantic_dim: int = 768, embedding_dim: int = 256,
        initial_lambda: float = 0.1,
    ) -> None:
        super().__init__()
        if not 0 < initial_lambda < 1:
            raise ValueError("initial_lambda must be strictly between zero and one")
        self.semantic_projection = nn.Linear(semantic_dim, embedding_dim)
        nn.init.xavier_uniform_(self.semantic_projection.weight)
        nn.init.zeros_(self.semantic_projection.bias)
        self.fusion_logit = nn.Parameter(torch.tensor(
            math.log(initial_lambda / (1 - initial_lambda)), dtype=torch.float32,
        ))

    @property
    def fusion_lambda(self) -> torch.Tensor:
        return self.fusion_logit.sigmoid()

    def forward(
        self, support_prototypes: torch.Tensor, semantic_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        if support_prototypes.ndim != 2 or semantic_embeddings.ndim != 2:
            raise ValueError("Prototypes and semantic embeddings must be matrices")
        if support_prototypes.shape[0] != semantic_embeddings.shape[0]:
            raise ValueError("Support and semantic prototypes must have matching class rows")
        support = F.normalize(support_prototypes, dim=-1)
        semantic = F.normalize(self.semantic_projection(semantic_embeddings), dim=-1)
        weight = self.fusion_lambda
        return F.normalize((1 - weight) * support + weight * semantic, dim=-1)
