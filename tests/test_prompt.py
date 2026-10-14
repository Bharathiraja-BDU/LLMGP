import torch

from src.prompt_builder import PromptBuilder


def test_prompt_shape_and_params():
    builder = PromptBuilder()
    emb = torch.randn(7, 768)
    out = builder(emb)
    assert out.shape == (7, 1433)
    assert builder.layer.in_features == 768
    assert builder.layer.out_features == 1433
    param_count = sum(p.numel() for p in builder.parameters())
    assert param_count == 1_101_977
