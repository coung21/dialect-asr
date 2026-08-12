import pytest
import torch

from dialect_asr.modules import DialectBranch


def test_dialect_branch_returns_one_logit_vector_per_utterance() -> None:
    branch = DialectBranch(hidden_size=8, num_dialects=3, dropout=0.0)
    hidden_states = torch.randn(2, 5, 8)  # [B=2, T_frame=5, H=8].

    logits = branch(hidden_states)  # [B=2, T_frame=5, H=8] -> [B=2, D=3].

    assert logits.shape == (2, 3)  # [B=2, D=3].


def test_masked_pooling_ignores_padded_frames() -> None:
    branch = DialectBranch(hidden_size=2, num_dialects=3, dropout=0.0)
    valid_states = torch.tensor([[[1.0, 3.0], [3.0, 5.0]]])
    # Two valid frames: [B=1, T_valid=2, H=2].
    padded_states = torch.tensor([[[1.0, 3.0], [3.0, 5.0], [100.0, 100.0]]])
    # Add one padded frame: [B=1, T_frame=3, H=2].
    valid_mask = torch.tensor([[1, 1]])  # [B=1, T_valid=2].
    padded_mask = torch.tensor([[1, 1, 0]])  # [B=1, T_frame=3].

    valid_logits = branch(valid_states, valid_mask)  # [1, 2, 2] -> [1, 3].
    padded_logits = branch(padded_states, padded_mask)  # [1, 3, 2] -> [1, 3].

    assert torch.allclose(valid_logits, padded_logits)


def test_dialect_branch_backpropagates_to_hidden_states() -> None:
    branch = DialectBranch(hidden_size=4, num_dialects=3, dropout=0.0)
    hidden_states = torch.randn(2, 6, 4, requires_grad=True)
    # Encoder features: [B=2, T_frame=6, H=4].
    logits = branch(hidden_states)  # [B=2, T_frame=6, H=4] -> [B=2, D=3].
    loss = logits.sum()  # [B=2, D=3] -> scalar loss [].

    loss.backward()

    assert hidden_states.grad is not None
    assert hidden_states.grad.shape == hidden_states.shape  # [2, 6, 4].


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"hidden_size": 0}, "hidden_size"),
        ({"hidden_size": 8, "num_dialects": 1}, "num_dialects"),
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
    with pytest.raises(ValueError, match="frame_mask"):
        branch(
            torch.randn(2, 5, 8),  # [B=2, T_frame=5, H=8].
            torch.ones(2, 4),  # Invalid [B=2, T_frame=4].
        )
