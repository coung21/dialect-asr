import pytest
import torch
from torch import nn

from dialect_asr.modules import DGGFM


def test_dggfm_returns_expected_output_and_statistics_shapes() -> None:
    module = DGGFM(hidden_size=8, dialect_dim=4, gate_hidden_dim=6)
    hidden_states = torch.randn(2, 5, 8)  # [B=2, T_frame=5, H=8].
    dialect_embedding = torch.randn(2, 4)  # [B=2, E=4].
    feature_attention_mask = torch.ones(2, 5)  # [B=2, T_frame=5].

    output, statistics = module(
        hidden_states,              # [B=2, T_frame=5, H=8].
        dialect_embedding,          # [B=2, E=4].
        feature_attention_mask,     # [B=2, T_frame=5].
    )

    assert output.shape == (2, 5, 8)  # [B=2, T_frame=5, H=8].
    assert statistics["gate"].shape == (2, 1, 8)  # [B=2, 1, H=8].
    assert statistics["gate_mean"].ndim == 0  # Scalar [].
    assert statistics["gate_std"].ndim == 0  # Scalar [].
    assert not statistics["gate_mean"].requires_grad
    assert not statistics["gate_std"].requires_grad


def test_dggfm_starts_as_exact_identity_with_mostly_closed_gate() -> None:
    module = DGGFM(hidden_size=8, dialect_dim=4, gate_hidden_dim=6)
    hidden_states = torch.randn(2, 5, 8)  # [B=2, T_frame=5, H=8].
    dialect_embedding = torch.randn(2, 4)  # [B=2, E=4].

    output, statistics = module(hidden_states, dialect_embedding)
    # [B=2, T_frame=5, H=8] and [B=2, E=4] -> output [B=2, T_frame=5, H=8].

    expected_gate = torch.sigmoid(torch.tensor(-2.0))  # Scalar [] -> scalar [].
    assert torch.equal(output, hidden_states)
    assert torch.allclose(statistics["gate"], torch.full((2, 1, 8), expected_gate))
    # Constant scalar gate [] -> expanded expected gate [B=2, 1, H=8].


def test_dggfm_uses_requested_network_dimensions_and_initialization() -> None:
    module = DGGFM(hidden_size=8, dialect_dim=4, gate_hidden_dim=6)

    assert module.to_gamma.weight.shape == (8, 4)  # [H=8, E=4].
    assert module.to_beta.weight.shape == (8, 4)  # [H=8, E=4].
    assert torch.count_nonzero(module.to_gamma.weight) == 0
    assert torch.count_nonzero(module.to_beta.weight) == 0
    assert isinstance(module.gate_network[0], nn.Linear)
    assert module.gate_network[0].weight.shape == (6, 12)  # [G=6, E+H=12].
    assert isinstance(module.gate_network[-1], nn.Linear)
    assert module.gate_network[-1].weight.shape == (8, 6)  # [H=8, G=6].
    assert torch.count_nonzero(module.gate_network[-1].weight) == 0
    assert torch.all(module.gate_network[-1].bias == -2.0)


def test_dggfm_backpropagates_through_modulation_after_nonzero_initialization() -> None:
    module = DGGFM(hidden_size=4, dialect_dim=3, gate_hidden_dim=5)
    nn.init.normal_(module.to_gamma.weight, std=0.02)
    nn.init.normal_(module.to_beta.weight, std=0.02)
    hidden_states = torch.randn(2, 6, 4, requires_grad=True)
    # Encoder features: [B=2, T_frame=6, H=4].
    dialect_embedding = torch.randn(2, 3, requires_grad=True)
    # Dialect embedding: [B=2, E=3].

    output, _ = module(hidden_states, dialect_embedding)
    # [B=2, T_frame=6, H=4] -> [B=2, T_frame=6, H=4].
    loss = output.square().mean()  # [B=2, T_frame=6, H=4] -> scalar loss [].
    loss.backward()

    assert hidden_states.grad is not None
    assert hidden_states.grad.shape == hidden_states.shape  # [2, 6, 4].
    assert dialect_embedding.grad is not None
    assert dialect_embedding.grad.shape == dialect_embedding.shape  # [2, 3].


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"hidden_size": 0}, "hidden_size"),
        ({"hidden_size": 8, "dialect_dim": 0}, "dialect_dim"),
        ({"hidden_size": 8, "gate_hidden_dim": 0}, "gate_hidden_dim"),
    ],
)
def test_dggfm_rejects_invalid_configuration(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        DGGFM(**kwargs)


def test_dggfm_rejects_incompatible_input_shapes() -> None:
    module = DGGFM(hidden_size=8, dialect_dim=4)

    with pytest.raises(ValueError, match="hidden_states"):
        module(torch.randn(2, 8), torch.randn(2, 4))
        # Invalid hidden states [B=2, H=8]; T_frame is missing.
    with pytest.raises(ValueError, match="Hidden size"):
        module(torch.randn(2, 5, 7), torch.randn(2, 4))
        # Invalid hidden states [B=2, T_frame=5, H=7]; expected H=8.
    with pytest.raises(ValueError, match="dialect_embedding"):
        module(torch.randn(2, 5, 8), torch.randn(4))
        # Invalid dialect embedding [E=4]; batch dimension is missing.
    with pytest.raises(ValueError, match="Batch size"):
        module(torch.randn(2, 5, 8), torch.randn(3, 4))
        # Hidden batch B=2 differs from embedding batch B=3.
    with pytest.raises(ValueError, match="Dialect embedding size"):
        module(torch.randn(2, 5, 8), torch.randn(2, 3))
        # Invalid dialect embedding [B=2, E=3]; expected E=4.
    with pytest.raises(ValueError, match="feature_attention_mask"):
        module(
            torch.randn(2, 5, 8),  # [B=2, T_frame=5, H=8].
            torch.randn(2, 4),  # [B=2, E=4].
            torch.ones(2, 4),  # Invalid [B=2, T_frame=4].
        )
