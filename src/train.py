from __future__ import annotations

from typing import Dict, Iterable

import torch
from torch import nn
from torch.optim import Adam

from .classifier import ProtoClassifier
from .graph_augmenter import GraphAugmenter
from .gnn_backbone import GCNBackbone
from .prompt_builder import PromptBuilder


def train_prompt_projection(
    class_embeddings: torch.Tensor,
    data_x: torch.Tensor,
    edge_index: torch.Tensor,
    support_idx: torch.Tensor,
    support_labels: torch.Tensor,
    query_idx: torch.Tensor,
    query_labels: torch.Tensor,
    backbone: GCNBackbone,
    class_count: int,
    temperature: float = 0.1,
    learning_rate: float = 0.001,
    weight_decay: float = 0.0001,
    epochs: int = 5,
) -> Dict[str, float]:
    prompt_builder = PromptBuilder()
    optimizer = Adam(prompt_builder.parameters(), lr=learning_rate, weight_decay=weight_decay)
    graph_augmenter = GraphAugmenter(num_classes=class_count)
    backbone.eval()
    for param in backbone.parameters():
        param.requires_grad = False

    running_loss = []
    for _ in range(epochs):
        optimizer.zero_grad()
        prompt_nodes = prompt_builder(class_embeddings)
        augmented = graph_augmenter.augment(data_x, edge_index, support_idx, support_labels, prompt_embeddings=prompt_nodes)
        node_embeddings = backbone(augmented.x, augmented.edge_index)
        valid_embeddings = node_embeddings[augmented.original_mask]
        prototypes = ProtoClassifier.prototypes(valid_embeddings[support_idx], support_labels, class_count)
        query_embeddings = valid_embeddings[query_idx]
        loss = ProtoClassifier.loss(query_embeddings, prototypes, query_labels, temperature=temperature)
        loss.backward()
        optimizer.step()
        running_loss.append(float(loss.item()))

    return {"loss": float(sum(running_loss) / len(running_loss)) if running_loss else 0.0}
