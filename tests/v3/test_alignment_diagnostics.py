import math

import pytest
import torch

from experiments.run_v3_mvp_validation import correct_class_margin, evaluate
from src.v3.semantic_prototype_fusion import SemanticPrototypeFusion


def test_correct_class_margin_uses_highest_wrong_logit():
    logits = torch.tensor([[5.0, 2.0, 4.0], [7.0, 3.0, 2.0]])
    labels = torch.tensor([0, 1])
    torch.testing.assert_close(correct_class_margin(logits, labels), torch.tensor([1.0, -4.0]))


def test_misaligned_semantics_reduce_margins_and_flip_predictions():
    model = SemanticPrototypeFusion(2, 2, initial_lambda=0.9).eval()
    with torch.no_grad():
        model.semantic_projection.weight.copy_(torch.eye(2))
        model.semantic_projection.bias.zero_()
    semantics = torch.zeros(7, 2)
    semantics[3], semantics[4] = torch.tensor([0.0, 1.0]), torch.tensor([1.0, 0.0])
    episode = (torch.eye(2), torch.eye(2), torch.tensor([0, 1]), torch.tensor([3, 4]))
    metrics = evaluate(model, [episode], semantics)
    assert metrics["semantic_support_cosine_mean"] == pytest.approx(0)
    assert metrics["semantic_wrong_support_cosine_mean"] == pytest.approx(1)
    assert metrics["margin_before_mean"] == pytest.approx(10)
    assert metrics["margin_after_mean"] == pytest.approx(-8 / math.sqrt(0.82))
    assert metrics["accuracy"] == 0
    assert metrics["correct_to_wrong"] == 2 and metrics["wrong_to_correct"] == 0
    assert metrics["margin_decreased_fraction"] == 1


def test_similar_semantics_are_detected_without_assuming_error():
    model = SemanticPrototypeFusion(2, 2).eval()
    with torch.no_grad():
        model.semantic_projection.weight.zero_()
        model.semantic_projection.bias.fill_(1)
    episode = (torch.eye(2), torch.eye(2), torch.tensor([0, 1]), torch.tensor([3, 4]))
    metrics = evaluate(model, [episode], torch.randn(7, 2))
    assert metrics["semantic_pairwise_cosine_mean"] == pytest.approx(1)
    assert metrics["alignment_advantage_mean"] == pytest.approx(0)
    assert metrics["accuracy"] == 1
    assert metrics["prediction_flips"] == 0
