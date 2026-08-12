import pytest
import torch
from torch import nn

from dialect_asr.modules import DialectBranch, masked_mean_pool


def test_dialect_branch_returns_logits_and_pooled_states() -> None:
    branch = DialectBranch(
        hidden_size=8,
        num_regions=3,
        bottleneck_size=4,
        dropout=0.0,
    )
    hidden_states = torch.randn(2, 5, 8)  # [B=2, T_frame=5, H=8].

    logits, pooled = branch(hidden_states)
    # [B=2, T_frame=5, H=8] -> logits [B=2, R=3], pooled [B=2, H=8].

    assert logits.shape == (2, 3)  # [B=2, R=3].
    assert pooled.shape == (2, 8)  # [B=2, H=8].
    assert isinstance(branch.classifier[0], nn.Linear)
    assert branch.classifier[0].in_features == 8
    assert branch.classifier[0].out_features == 4
    assert isinstance(branch.classifier[1], nn.GELU)
    assert isinstance(branch.classifier[2], nn.Dropout)
    assert isinstance(branch.classifier[3], nn.Linear)
    assert branch.classifier[3].in_features == 4
    assert branch.classifier[3].out_features == 3


def test_masked_mean_pool_ignores_padded_frames() -> None:
    valid_states = torch.tensor([[[1.0, 3.0], [3.0, 5.0]]])
    # Valid encoder frames: [B=1, T_valid=2, H=2].
    padded_states = torch.tensor([[[1.0, 3.0], [3.0, 5.0], [100.0, 100.0]]])
    # Add one padded frame: [B=1, T_frame=3, H=2].
    valid_mask = torch.tensor([[1, 1]])  # [B=1, T_valid=2].
    padded_mask = torch.tensor([[1, 1, 0]])  # [B=1, T_frame=3].

    valid_pooled = masked_mean_pool(valid_states, valid_mask)
    # [B=1, T_valid=2, H=2] -> [B=1, H=2].
    padded_pooled = masked_mean_pool(padded_states, padded_mask)
    # [B=1, T_frame=3, H=2] -> [B=1, H=2].

    assert torch.allclose(valid_pooled, padded_pooled)
    assert torch.allclose(valid_pooled, torch.tensor([[2.0, 4.0]]))


def test_dialect_branch_backpropagates_to_hidden_states() -> None:
    branch = DialectBranch(
        hidden_size=4,
        num_regions=3,
        bottleneck_size=2,
        dropout=0.0,
    )
    hidden_states = torch.randn(2, 6, 4, requires_grad=True)
    # Encoder features: [B=2, T_frame=6, H=4].
    logits, pooled = branch(hidden_states)
    # [B=2, T_frame=6, H=4] -> logits [B=2, R=3], pooled [B=2, H=4].
    loss = logits.sum() + pooled.sum()  # [2, 3] and [2, 4] -> scalar loss [].

    loss.backward()

    assert hidden_states.grad is not None
    assert hidden_states.grad.shape == hidden_states.shape  # [2, 6, 4].


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"hidden_size": 0}, "hidden_size"),
        ({"hidden_size": 8, "num_regions": 1}, "num_regions"),
        ({"hidden_size": 8, "bottleneck_size": 0}, "bottleneck_size"),
        ({"hidden_size": 8, "dropout": 1.0}, "dropout"),
    ],
)
def test_dialect_branch_rejects_invalid_configuration(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        DialectBranch(**kwargs)


def test_dialect_branch_rejects_incompatible_shapes() -> None:
    branch = DialectBranch(hidden_size=8)

    with pytest.raises(ValueError, match="hidden_states"):
        branch(torch.randn(2, 8))  # Invalid [B=2, H=8]; T_frame is missing.
    with pytest.raises(ValueError, match="Hidden size"):
        branch(torch.randn(2, 5, 4))  # Invalid H=4; expected H=8.
    with pytest.raises(ValueError, match="feature_attention_mask"):
        branch(
            torch.randn(2, 5, 8),  # [B=2, T_frame=5, H=8].
            torch.ones(2, 4),  # Invalid [B=2, T_frame=4].
        )
