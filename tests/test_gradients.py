import torch

from src.prompt_builder import PromptBuilder


def test_prompt_gradients_flow():
    builder = PromptBuilder()
    emb = torch.randn(2, 768, requires_grad=False)
    out = builder(emb)
    loss = out.pow(2).sum()
    loss.backward()
    assert emb.grad is None
    assert builder.layer.weight.grad is not None
    assert builder.layer.bias.grad is not None
    assert builder.layer.weight.grad.abs().sum() > 0
    assert builder.layer.bias.grad.abs().sum() > 0
