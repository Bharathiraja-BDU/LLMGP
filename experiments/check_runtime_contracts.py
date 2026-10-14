import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import verify_cora_dataset
from src.graph_augmenter import GraphAugmenter
from src.gnn_backbone import GCNBackbone
from src.llm_encoder import SciBERTEncoder
from src.prompt_builder import PromptBuilder
from src.utils import get_class_split


def main() -> None:
    info = verify_cora_dataset()
    assert info["num_nodes"] == 2708
    assert info["num_features"] == 1433
    assert info["num_classes"] == 7
    print("DATASET_OK", info["num_nodes"], info["num_features"], info["num_classes"], info["edges"])

    split_a = get_class_split("A")
    split_b = get_class_split("B")
    assert split_a["train"] == [0, 1, 2, 3]
    assert split_a["val"] == [4]
    assert split_a["test"] == [5, 6]
    assert split_b["train"] == [0, 1, 2]
    assert split_b["val"] == [3, 4]
    assert split_b["test"] == [5, 6]
    print("SPLITS_OK", split_a, split_b)

    builder = PromptBuilder()
    params = sum(p.numel() for p in builder.parameters())
    assert params == 1_101_977
    assert builder.layer.in_features == 768
    assert builder.layer.out_features == 1433
    out = builder(torch.randn(2, 768))
    assert out.shape == (2, 1433)
    print("PROMPT_OK", params, tuple(out.shape))

    graph = GraphAugmenter(num_classes=2)
    x = torch.randn(4, 1433)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    support_idx = torch.tensor([0, 2])
    labels = torch.tensor([0, 1])
    aug = graph.augment(x, edge_index, support_idx, labels, torch.randn(2, 1433))
    assert aug.x.shape[0] == 6
    assert aug.original_mask.sum().item() == 4
    assert aug.prompt_mask.sum().item() == 2
    assert aug.edge_index.shape[1] >= edge_index.shape[1]
    print("GRAPH_OK", tuple(aug.x.shape), aug.edge_index.shape[1], int(aug.original_mask.sum()), int(aug.prompt_mask.sum()))

    gnn = GCNBackbone()
    for parameter in gnn.parameters():
        parameter.requires_grad = False
    assert all(not p.requires_grad for p in gnn.parameters())
    print("FROZEN_GNN_OK", all(not p.requires_grad for p in gnn.parameters()))

    enc = SciBERTEncoder(model_name="allenai/scibert_scivocab_uncased")
    embs = enc.encode_texts(["An example title.", "Another paper abstract."], batch_size=2)
    assert embs.shape[1] == 768
    print("SCIBERT_OK", tuple(embs.shape))


if __name__ == "__main__":
    main()
