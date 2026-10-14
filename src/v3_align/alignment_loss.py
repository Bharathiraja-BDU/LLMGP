from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def class_discriminative_alignment(
    projected_semantics: torch.Tensor,
    support_prototypes: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Symmetric contrastive CE with matching rows positive, other rows negative.

    Support prototypes are frozen targets. No fusion gate is used by this loss;
    the joint episodic classification loss trains the existing gate.
    """
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Alignment temperature must be finite and positive")
    if projected_semantics.ndim != 2 or projected_semantics.shape != support_prototypes.shape:
        raise ValueError("Semantic and support prototypes must have the same matrix shape")
    if projected_semantics.shape[0] < 2:
        raise ValueError("Contrastive alignment requires at least two classes")
    semantic = F.normalize(projected_semantics, dim=-1)
    support = F.normalize(support_prototypes.detach(), dim=-1)
    similarity = semantic @ support.T / temperature
    targets = torch.arange(similarity.shape[0], device=similarity.device)
    return 0.5 * (F.cross_entropy(similarity, targets) + F.cross_entropy(similarity.T, targets))
