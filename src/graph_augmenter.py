from __future__ import annotations

import torch
from torch_geometric.data import Data


class GraphAugmenter:
    """Append one prompt node per class and connect it to the matching support nodes."""

    def __init__(self, num_classes: int = 7):
        self.num_classes = num_classes

    def augment(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        support_idx: torch.Tensor,
        labels: torch.Tensor,
        prompt_embeddings: torch.Tensor | None = None,
    ) -> Data:
        if x.dim() != 2:
            raise ValueError(f"x must be 2D [N, F], received shape {tuple(x.shape)}")
        if edge_index.dim() != 2 or edge_index.shape[0] != 2:
            raise ValueError(f"edge_index must be [2, E], received shape {tuple(edge_index.shape)}")

        support_idx = torch.as_tensor(support_idx, dtype=torch.long, device=x.device)
        labels = torch.as_tensor(labels, dtype=torch.long, device=x.device)

        if prompt_embeddings is None:
            prompt_embeddings = torch.zeros((self.num_classes, x.shape[1]), device=x.device, dtype=x.dtype)
        elif prompt_embeddings.shape[0] != self.num_classes:
            raise ValueError(
                f"Expected {self.num_classes} prompt embeddings but received {prompt_embeddings.shape[0]}"
            )

        prompt_offset = x.shape[0]
        augmented_x = torch.cat([x, prompt_embeddings], dim=0)

        source_edges: list[int] = []
        target_edges: list[int] = []
        for node_idx, label in zip(support_idx.tolist(), labels.tolist()):
            if label < 0 or label >= self.num_classes:
                raise ValueError(f"Class label {label} is outside the valid range [0,{self.num_classes-1}]")
            prompt_node_idx = prompt_offset + int(label)
            source_edges.extend([int(node_idx), int(prompt_node_idx)])
            target_edges.extend([int(prompt_node_idx), int(node_idx)])

        if source_edges:
            prompt_edge_index = torch.tensor([source_edges, target_edges], dtype=torch.long, device=x.device)
            augmented_edge_index = torch.cat([edge_index.to(x.device), prompt_edge_index], dim=1)
        else:
            augmented_edge_index = edge_index.to(x.device).clone()

        original_mask = torch.zeros(augmented_x.shape[0], dtype=torch.bool, device=x.device)
        original_mask[: x.shape[0]] = True
        prompt_mask = torch.zeros(augmented_x.shape[0], dtype=torch.bool, device=x.device)
        prompt_mask[prompt_offset:] = True

        return Data(
            x=augmented_x,
            edge_index=augmented_edge_index,
            original_mask=original_mask,
            prompt_mask=prompt_mask,
        )
