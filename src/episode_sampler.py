from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


class EpisodeSampler:
    """Constrained episode sampling for N-way K-shot tasks with 15 query nodes per class."""

    def __init__(self, num_classes: int, support_shots: int, query_shots: int = 15, rng: Optional[np.random.Generator] = None):
        self.num_classes = num_classes
        self.support_shots = support_shots
        self.query_shots = query_shots
        self.rng = rng or np.random.default_rng(0)

    def sample_episode(
        self,
        labels: Optional[np.ndarray] = None,
        class_ids: Optional[List[int]] = None,
        seed: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = self.rng

        if labels is None:
            if class_ids is None:
                class_ids = [0, 1]
            synthetic = []
            for class_id in class_ids:
                synthetic.extend([int(class_id)] * (self.support_shots + self.query_shots))
            label_values = np.asarray(synthetic, dtype=np.int64)
        else:
            label_values = np.asarray(labels)
            if class_ids is None:
                available_classes = np.unique(label_values)
                class_ids = rng.choice(available_classes, size=2, replace=False).tolist()

        if class_ids is None:
            available_classes = np.unique(label_values)
            class_ids = rng.choice(available_classes, size=2, replace=False).tolist()

        support_idx: List[int] = []
        query_idx: List[int] = []
        support_labels: List[int] = []
        query_labels: List[int] = []

        for class_id in class_ids:
            class_nodes = np.where(label_values == class_id)[0]
            if class_nodes.shape[0] < self.support_shots + self.query_shots:
                raise ValueError(f"Class {class_id} has insufficient nodes: {class_nodes.shape[0]}")
            perm = rng.permutation(class_nodes)
            support_nodes = perm[: self.support_shots]
            query_nodes = perm[self.support_shots : self.support_shots + self.query_shots]
            support_idx.extend(support_nodes.tolist())
            query_idx.extend(query_nodes.tolist())
            support_labels.extend([int(class_id)] * self.support_shots)
            query_labels.extend([int(class_id)] * self.query_shots)

        return {
            "support_idx": np.asarray(support_idx, dtype=np.int64),
            "query_idx": np.asarray(query_idx, dtype=np.int64),
            "support_labels": np.asarray(support_labels, dtype=np.int64),
            "query_labels": np.asarray(query_labels, dtype=np.int64),
            "class_ids": np.asarray(class_ids, dtype=np.int64),
        }
