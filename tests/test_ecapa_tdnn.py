import pytest
import torch
from torch import nn

from dialect_asr.modules import (
    AttentiveStatisticsPooling,
    ECAPA_TDNN_DID,
    Res2NetLayer,
    SERes2Block,
    SqueezeExcitation,
)


def test_squeeze_excitation_preserves_shape_and_gates_channels() -> None:
    se = SqueezeExcitation(channels=8, bottleneck_channels=4)
    x = torch.randn(2, 8, 6)  # [B=2, C=8, T_frame=6].

    out = se(x)

    assert out.shape == x.shape  # [2, 8, 6].


def test_squeeze_excitation_ignores_masked_frames_when_pooling() -> None:
    se = SqueezeExcitation(channels=2, bottleneck_channels=2)
    se.eval()
    valid = torch.tensor([[[1.0, 3.0], [3.0, 5.0]]])  # [B=1, C=2, T_valid=2].
    padded = torch.cat([valid, torch.full((1, 2, 1), 100.0)], dim=2)  # [B=1, C=2, T=3].
    valid_mask = torch.tensor([[1, 1]])
    padded_mask = torch.tensor([[1, 1, 0]])

    with torch.no_grad():
        valid_out = se(valid, valid_mask)  # [B=1, C=2, T_valid=2].
        padded_out = se(padded, padded_mask)  # [B=1, C=2, T=3].

    # The gate is computed from the masked pool, so it must match on the
    # shared valid frames regardless of what the padded frame contains.
    assert torch.allclose(padded_out[:, :, :2], valid_out)


def test_res2net_layer_preserves_temporal_length() -> None:
    layer = Res2NetLayer(channels=8, kernel_size=3, dilation=2, scale=4)
    x = torch.randn(2, 8, 10)  # [B=2, C=8, T_frame=10].

    out = layer(x)

    assert out.shape == x.shape  # [2, 8, 10].


def test_res2net_layer_rejects_channels_not_divisible_by_scale() -> None:
    with pytest.raises(ValueError, match="scale"):
        Res2NetLayer(channels=6, kernel_size=3, dilation=1, scale=4)


def test_se_res2block_preserves_shape() -> None:
    block = SERes2Block(channels=8, kernel_size=3, dilation=1, res2net_scale=4)
    x = torch.randn(2, 8, 10)  # [B=2, C=8, T_frame=10].

    out = block(x)

    assert out.shape == x.shape  # [2, 8, 10].


def test_attentive_statistics_pooling_output_shape() -> None:
    pooling = AttentiveStatisticsPooling(channels=8, attention_channels=4)
    x = torch.randn(2, 8, 10)  # [B=2, C=8, T_frame=10].

    pooled = pooling(x)

    assert pooled.shape == (2, 16)  # [B=2, 2*C=16].


def test_attentive_statistics_pooling_ignores_padded_frames() -> None:
    pooling = AttentiveStatisticsPooling(channels=4, attention_channels=4)
    pooling.eval()
    valid = torch.randn(1, 4, 5)  # [B=1, C=4, T_valid=5].
    padded = torch.cat([valid, torch.full((1, 4, 3), 1000.0)], dim=2)  # [B=1, C=4, T=8].
    valid_mask = torch.ones(1, 5)
    padded_mask = torch.cat([torch.ones(1, 5), torch.zeros(1, 3)], dim=1)

    with torch.no_grad():
        valid_pooled = pooling(valid, valid_mask)
        padded_pooled = pooling(padded, padded_mask)

    assert torch.allclose(valid_pooled, padded_pooled, atol=1e-5)


def test_ecapa_tdnn_did_returns_logits_and_embedding() -> None:
    did = ECAPA_TDNN_DID(
        hidden_size=16,
        num_regions=3,
        channels=8,
        embedding_size=6,
        res2net_scale=2,
        se_bottleneck_channels=4,
        attention_channels=4,
        dropout=0.0,
    )
    hidden_states = torch.randn(2, 12, 16)  # [B=2, T_frame=12, H=16].

    logits, embedding = did(hidden_states)
    # [B=2, T_frame=12, H=16] -> logits [B=2, R=3], embedding [B=2, E=6].

    assert logits.shape == (2, 3)
    assert embedding.shape == (2, 6)
    assert isinstance(did.classifier[0], nn.BatchNorm1d)
    assert isinstance(did.classifier[-1], nn.Linear)
    assert did.classifier[-1].out_features == 3


def test_ecapa_tdnn_did_backpropagates_to_hidden_states() -> None:
    did = ECAPA_TDNN_DID(
        hidden_size=8,
        num_regions=3,
        channels=8,
        embedding_size=4,
        res2net_scale=2,
        se_bottleneck_channels=4,
        attention_channels=4,
        dropout=0.0,
    )
    hidden_states = torch.randn(2, 10, 8, requires_grad=True)
    # Encoder features: [B=2, T_frame=10, H=8].

    logits, embedding = did(hidden_states)
    loss = logits.sum() + embedding.sum()  # [2, 3] and [2, 4] -> scalar loss [].
    loss.backward()

    assert hidden_states.grad is not None
    assert hidden_states.grad.shape == hidden_states.shape  # [2, 10, 8].


def test_ecapa_tdnn_did_respects_feature_attention_mask_shape() -> None:
    did = ECAPA_TDNN_DID(
        hidden_size=8,
        channels=8,
        res2net_scale=2,
        se_bottleneck_channels=4,
        attention_channels=4,
    )
    hidden_states = torch.randn(2, 10, 8)  # [B=2, T_frame=10, H=8].
    feature_attention_mask = torch.ones(2, 10)

    logits, embedding = did(hidden_states, feature_attention_mask)

    assert logits.shape == (2, did.num_regions)
    assert embedding.shape == (2, 192)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"hidden_size": 0}, "hidden_size"),
        ({"hidden_size": 8, "num_regions": 1}, "num_regions"),
        ({"hidden_size": 8, "channels": 0}, "channels"),
        ({"hidden_size": 8, "embedding_size": 0}, "embedding_size"),
        ({"hidden_size": 8, "kernel_sizes": (5, 3, 3)}, "kernel_sizes"),
        ({"hidden_size": 8, "dropout": 1.0}, "dropout"),
    ],
)
def test_ecapa_tdnn_did_rejects_invalid_configuration(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        ECAPA_TDNN_DID(**kwargs)


def test_ecapa_tdnn_did_rejects_incompatible_shapes() -> None:
    did = ECAPA_TDNN_DID(hidden_size=8, channels=8, res2net_scale=2, se_bottleneck_channels=4)

    with pytest.raises(ValueError, match="hidden_states"):
        did(torch.randn(2, 8))  # Invalid [B=2, H=8]; T_frame is missing.
    with pytest.raises(ValueError, match="Hidden size"):
        did(torch.randn(2, 10, 4))  # Invalid H=4; expected H=8.
    with pytest.raises(ValueError, match="feature_attention_mask"):
        did(
            torch.randn(2, 10, 8),  # [B=2, T_frame=10, H=8].
            torch.ones(2, 4),  # Invalid [B=2, T_frame=4].
        )
