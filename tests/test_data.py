import numpy as np

from src.data_loader import verify_cora_dataset


def test_cora_metadata():
    info = verify_cora_dataset()
    assert info["num_nodes"] == 2708
    assert info["num_features"] == 1433
    assert info["num_classes"] == 7
    assert info["label_distribution"]["count"] == 7


def test_dataset_shape():
    info = verify_cora_dataset()
    assert info["x"].shape[0] == info["num_nodes"]
    assert info["x"].shape[1] == info["num_features"]
    assert info["y"].shape[0] == info["num_nodes"]
