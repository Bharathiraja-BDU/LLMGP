from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid


def verify_cora_dataset(root: str | None = None) -> Dict[str, Any]:
    """Load the standard Planetoid Cora graph and validate the expected metadata.

    Notes:
        - Planetoid Cora has 2,708 nodes, 1,433 features, and 7 classes.
        - `edge_index` stores directed adjacency entries, and for an undirected graph each
          undirected edge appears in both directions. This is the counting convention used by
          PyG's Planetoid loader and should be reported in the reproducibility notes.
    """
    data_root = Path(root) if root is not None else Path(__file__).resolve().parents[1] / "data"
    dataset = Planetoid(root=str(data_root), name="Cora")
    data: Data = dataset[0]

    label_distribution = {
        int(class_id): int((data.y == class_id).sum().item())
        for class_id in range(int(dataset.num_classes))
    }

    metadata = {
        "dataset": dataset,
        "data": data,
        "x": data.x,
        "y": data.y,
        "edge_index": data.edge_index,
        "num_nodes": int(data.num_nodes),
        "num_features": int(data.x.shape[1]),
        "num_classes": int(dataset.num_classes),
        "edges": int(data.edge_index.shape[1]),
        "label_distribution": {"classes": label_distribution, "count": len(label_distribution)},
    }

    if metadata["num_nodes"] != 2708:
        raise ValueError(f"Expected 2708 nodes but found {metadata['num_nodes']}")
    if metadata["num_features"] != 1433:
        raise ValueError(f"Expected 1433 features but found {metadata['num_features']}")
    if metadata["num_classes"] != 7:
        raise ValueError(f"Expected 7 classes but found {metadata['num_classes']}")

    return metadata
