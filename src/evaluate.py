from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

from .classifier import ProtoClassifier


def evaluate_episode(
    query_embeddings: torch.Tensor,
    support_embeddings: torch.Tensor,
    query_labels: torch.Tensor,
    support_labels: torch.Tensor,
    num_classes: int,
    temperature: float = 0.1,
):
    prototypes = ProtoClassifier.prototypes(support_embeddings, support_labels, num_classes)
    probabilities = ProtoClassifier.probabilities(query_embeddings, prototypes, temperature=temperature)
    predictions = probabilities.argmax(dim=-1)
    acc = accuracy_score(query_labels.cpu().numpy(), predictions.cpu().numpy())
    macro_f1 = f1_score(query_labels.cpu().numpy(), predictions.cpu().numpy(), average="macro")
    return {"accuracy": float(acc), "macro_f1": float(macro_f1), "probabilities": probabilities, "predictions": predictions}
