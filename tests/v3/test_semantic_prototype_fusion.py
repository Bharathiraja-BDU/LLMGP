import torch
import torch.nn.functional as F

from src.v3.semantic_prototype_fusion import SemanticPrototypeFusion


def test_fusion_formula_and_normalization():
    torch.manual_seed(17)
    model = SemanticPrototypeFusion(4, 3)
    support, descriptions = torch.randn(2, 3), torch.randn(2, 4)
    assert torch.isclose(model.fusion_lambda, torch.tensor(0.1))
    semantic = F.normalize(model.semantic_projection(descriptions), dim=-1)
    expected = F.normalize(0.9 * F.normalize(support, dim=-1) + 0.1 * semantic, dim=-1)
    torch.testing.assert_close(model(support, descriptions), expected)
    torch.testing.assert_close(model(support, descriptions).norm(dim=-1), torch.ones(2))
    torch.testing.assert_close(model(support * 7, descriptions), expected)


def test_endpoints_recover_support_and_semantics():
    model = SemanticPrototypeFusion(3, 3)
    support, descriptions = torch.randn(2, 3), torch.randn(2, 3)
    with torch.no_grad():
        model.fusion_logit.fill_(-float("inf"))
    torch.testing.assert_close(model(support, descriptions), F.normalize(support, dim=-1))
    with torch.no_grad():
        model.fusion_logit.fill_(float("inf"))
    torch.testing.assert_close(model(support, descriptions), F.normalize(model.semantic_projection(descriptions), dim=-1))


def test_training_gradients_without_encoder_gradients():
    torch.manual_seed(19)
    model = SemanticPrototypeFusion(4, 3)
    encoder = torch.nn.Linear(5, 3).requires_grad_(False)
    support = encoder(torch.randn(2, 5)).detach()
    output = model(support, torch.randn(2, 4))
    F.cross_entropy(output @ torch.randn(3, 2), torch.tensor([0, 1])).backward()
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all() and parameter.grad.norm() > 0
    assert all(parameter.grad is None for parameter in encoder.parameters())
