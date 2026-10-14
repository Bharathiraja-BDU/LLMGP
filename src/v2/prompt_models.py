from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class _PromptProjection(nn.Module):
    def __init__(self, input_dim: int = 768, output_dim: int = 1433) -> None:
        super().__init__()
        self.projection = nn.Linear(input_dim, output_dim, bias=True)
        nn.init.xavier_uniform_(self.projection.weight)


class OriginalPrompt(_PromptProjection):
    """V1-style projected prompt features and prompt-node connectivity."""

    def forward(self, class_embeddings: torch.Tensor) -> torch.Tensor:
        return self.projection(class_embeddings)


class NormalizedPrompt(_PromptProjection):
    """L2-normalized prompt features rescaled to the train-node reference norm."""

    def __init__(
        self,
        reference_node_norm: float,
        input_dim: int = 768,
        output_dim: int = 1433,
        eps: float = 1e-12,
    ) -> None:
        super().__init__(input_dim=input_dim, output_dim=output_dim)
        if not math.isfinite(reference_node_norm) or reference_node_norm <= 0:
            raise ValueError("reference_node_norm must be finite and positive")
        self.reference_node_norm = float(reference_node_norm)
        self.eps = eps

    def normalize_projection(self, projected: torch.Tensor) -> torch.Tensor:
        return F.normalize(projected, p=2, dim=-1, eps=self.eps) * self.reference_node_norm

    def forward(self, class_embeddings: torch.Tensor) -> torch.Tensor:
        return self.normalize_projection(self.projection(class_embeddings))


class NormalizedGatedPrompt(NormalizedPrompt):
    """Normalized semantic prompts with a shared learnable sigmoid contribution gate."""

    def __init__(
        self,
        reference_node_norm: float,
        initial_gate: float = 0.1,
        input_dim: int = 768,
        output_dim: int = 1433,
        eps: float = 1e-12,
    ) -> None:
        if not 0.0 < initial_gate < 1.0:
            raise ValueError("initial_gate must lie strictly between zero and one")
        super().__init__(
            reference_node_norm=reference_node_norm,
            input_dim=input_dim,
            output_dim=output_dim,
            eps=eps,
        )
        gate_logit = math.log(initial_gate / (1.0 - initial_gate))
        self.gate_logit = nn.Parameter(torch.tensor(gate_logit, dtype=torch.float32))

    @property
    def gate(self) -> torch.Tensor:
        return torch.sigmoid(self.gate_logit)

    def forward(self, class_embeddings: torch.Tensor) -> torch.Tensor:
        normalized = self.normalize_projection(self.projection(class_embeddings))
        return normalized * self.gate
