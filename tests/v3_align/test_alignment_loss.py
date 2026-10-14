import pytest
import torch

from src.v3.semantic_prototype_fusion import SemanticPrototypeFusion
from src.v3_align.alignment_loss import class_discriminative_alignment


def test_correct_pairs_have_lower_loss_than_wrong_pairs():
    support = torch.eye(3)
    correct = class_discriminative_alignment(support, support)
    wrong = class_discriminative_alignment(support.roll(1, dims=0), support)
    assert correct < 0.001 and wrong > 9
    torch.testing.assert_close(class_discriminative_alignment(support * 3, support * 7), correct)
    torch.testing.assert_close(class_discriminative_alignment(support.flip(0), support.flip(0)), correct)


def test_alignment_trains_projection_without_encoder_or_gate_gradients():
    torch.manual_seed(31)
    model = SemanticPrototypeFusion(4, 3)
    support = torch.randn(3, 3, requires_grad=True)
    descriptions = torch.randn(3, 4)
    loss = class_discriminative_alignment(model.semantic_projection(descriptions), support)
    loss.backward()
    assert support.grad is None
    assert model.fusion_logit.grad is None
    for parameter in model.semantic_projection.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.norm() > 0


def test_invalid_temperature_or_missing_negative_is_rejected():
    with pytest.raises(ValueError):
        class_discriminative_alignment(torch.eye(2), torch.eye(2), temperature=0)
    with pytest.raises(ValueError):
        class_discriminative_alignment(torch.ones(1, 3), torch.ones(1, 3))
