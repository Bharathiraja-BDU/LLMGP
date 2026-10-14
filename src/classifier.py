from __future__ import annotations

import torch
import torch.nn.functional as F


class ProtoClassifier:
    """Prototype-based classification using cosine similarity and temperature."""

    @staticmethod
    def prototypes(support_embeddings: torch.Tensor, labels: torch.Tensor, num_classes: int) -> torch.Tensor:
        prototypes = []
        for class_id in range(num_classes):
            class_mask = labels == class_id
            if not torch.any(class_mask):
                raise ValueError(f"No support nodes found for class {class_id}")
            class_embeddings = support_embeddings[class_mask]
            prototypes.append(class_embeddings.mean(dim=0, keepdim=True))
        return torch.cat(prototypes, dim=0)

    @staticmethod
    def logits(query_embeddings: torch.Tensor, prototypes: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
        query_norm = F.normalize(query_embeddings, dim=-1)
        prototype_norm = F.normalize(prototypes, dim=-1)
        similarity = torch.matmul(query_norm, prototype_norm.T) / temperature
        return similarity

    @staticmethod
    def probabilities(query_embeddings: torch.Tensor, prototypes: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
        logits = ProtoClassifier.logits(query_embeddings, prototypes, temperature=temperature)
        return torch.softmax(logits, dim=-1)

    @staticmethod
    def loss(query_embeddings: torch.Tensor, prototypes: torch.Tensor, query_labels: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
        logits = ProtoClassifier.logits(query_embeddings, prototypes, temperature=temperature)
        return F.cross_entropy(logits, query_labels)
