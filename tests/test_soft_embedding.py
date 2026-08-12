import pytest
import torch

from dialect_asr.modules import SoftDialectEmbedding


def test_soft_embedding_returns_expected_shapes_and_probabilities() -> None:
    module = SoftDialectEmbedding(
        num_regions=3,
        embedding_dim=4,
        temperature=1.0,
    )
    dialect_logits = torch.tensor([[2.0, 1.0, 0.0], [0.0, 1.0, 2.0]])
    # Region logits: [B=2, R=3].

    dialect_embedding, posterior = module(dialect_logits)
    # [B=2, R=3] -> embedding [B=2, E=4], posterior [B=2, R=3].

    assert dialect_embedding.shape == (2, 4)  # [B=2, E=4].
    assert posterior.shape == (2, 3)  # [B=2, R=3].
    assert torch.allclose(posterior.sum(dim=-1), torch.ones(2))
    # [B=2, R=3] -> sum over R -> [B=2].
    expected_embedding = posterior @ module.embedding.weight
    # [B=2, R=3] @ [R=3, E=4] -> [B=2, E=4].
    assert torch.allclose(dialect_embedding, expected_embedding)


def test_lower_temperature_produces_sharper_posterior() -> None:
    logits = torch.tensor([[2.0, 1.0, 0.0]])  # [B=1, R=3].
    regular = SoftDialectEmbedding(temperature=1.0)
    sharp = SoftDialectEmbedding(temperature=0.5)

    _, regular_posterior = regular(logits)  # [1, 3] -> [1, 3].
    _, sharp_posterior = sharp(logits)  # [1, 3] -> [1, 3].

    assert sharp_posterior.max() > regular_posterior.max()


def test_soft_embedding_backpropagates_to_logits_and_embedding_weight() -> None:
    module = SoftDialectEmbedding(num_regions=3, embedding_dim=4)
    dialect_logits = torch.randn(2, 3, requires_grad=True)  # [B=2, R=3].

    dialect_embedding, posterior = module(dialect_logits)
    # [B=2, R=3] -> embedding [B=2, E=4], posterior [B=2, R=3].
    loss = dialect_embedding.square().sum() + posterior.square().sum()
    # [B=2, E=4] and [B=2, R=3] -> scalar loss [].
    loss.backward()

    assert dialect_logits.grad is not None
    assert dialect_logits.grad.shape == dialect_logits.shape  # [B=2, R=3].
    assert module.embedding.weight.grad is not None
    assert module.embedding.weight.grad.shape == (3, 4)  # [R=3, E=4].


def test_embedding_weight_uses_small_normal_initialization() -> None:
    torch.manual_seed(42)
    module = SoftDialectEmbedding(num_regions=3, embedding_dim=4096)
    weights = module.embedding.weight.detach()  # [R=3, E=4096].

    assert weights.mean().item() == pytest.approx(0.0, abs=1.0e-3)
    assert weights.std().item() == pytest.approx(0.02, abs=1.0e-3)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"num_regions": 1}, "num_regions"),
        ({"embedding_dim": 0}, "embedding_dim"),
        ({"temperature": 0.0}, "temperature"),
        ({"temperature": -1.0}, "temperature"),
    ],
)
def test_soft_embedding_rejects_invalid_configuration(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        SoftDialectEmbedding(**kwargs)


def test_soft_embedding_rejects_incompatible_logits_shape() -> None:
    module = SoftDialectEmbedding(num_regions=3)

    with pytest.raises(ValueError, match="dialect_logits"):
        module(torch.randn(3))  # Invalid [R=3]; batch dimension is missing.
    with pytest.raises(ValueError, match="region logits"):
        module(torch.randn(2, 4))  # Invalid [B=2, R=4]; expected R=3.
