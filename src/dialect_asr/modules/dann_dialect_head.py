"""Adversarial dialect classifier attached to encoder hidden states."""

from __future__ import annotations

from torch import Tensor, nn

from .gradient_reversal import GradientReversal


def masked_mean_pool(
    hidden_states: Tensor,
    feature_attention_mask: Tensor | None = None,
) -> Tensor:
    """Mean-pool valid frames from ``[B, T_frame, H]`` into ``[B, H]``."""
    if hidden_states.ndim != 3:
        raise ValueError("hidden_states phải có shape [B, T_frame, H]")

    if feature_attention_mask is None:
        return hidden_states.mean(dim=1)  # [B, T_frame, H] -> [B, H].

    expected_shape = hidden_states.shape[:2]  # [B, T_frame, H] -> (B, T_frame).
    if feature_attention_mask.shape != expected_shape:
        raise ValueError(
            "feature_attention_mask phải có shape [B, T_frame]; "
            f"cần {tuple(expected_shape)}, nhận {tuple(feature_attention_mask.shape)}"
        )

    float_mask = feature_attention_mask.to(
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    ).unsqueeze(-1)
    # [B, T_frame] -> [B, T_frame, 1] for broadcasting over H.
    summed_states = (hidden_states * float_mask).sum(dim=1)
    # [B, T_frame, H] * [B, T_frame, 1] -> sum over frames -> [B, H].
    valid_lengths = float_mask.sum(dim=1).clamp_min(1.0)
    # [B, T_frame, 1] -> sum over frames -> [B, 1].
    return summed_states / valid_lengths  # [B, H] / [B, 1] -> [B, H].


class DANNDialectHead(nn.Module):
    """Apply GRL, temporal pooling and an MLP dialect classifier."""

    def __init__(
        self,
        hidden_size: int,
        num_regions: int = 3,
        bottleneck_size: int = 256,
        dropout: float = 0.1,
        grl_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("hidden_size phải > 0")
        if num_regions <= 1:
            raise ValueError("num_regions phải > 1")
        if bottleneck_size <= 0:
            raise ValueError("bottleneck_size phải > 0")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout phải nằm trong khoảng [0, 1)")

        self.hidden_size = hidden_size
        self.num_regions = num_regions
        self.grl = GradientReversal(grl_scale)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, bottleneck_size),
            # [B, H] -> [B, bottleneck_size].
            nn.GELU(),  # [B, bottleneck_size] -> [B, bottleneck_size].
            nn.Dropout(dropout),  # [B, bottleneck_size] -> same shape.
            nn.Linear(bottleneck_size, num_regions),
            # [B, bottleneck_size] -> dialect logits [B, R].
        )

    def forward(
        self,
        hidden_states: Tensor,
        feature_attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if hidden_states.ndim != 3 or hidden_states.shape[-1] != self.hidden_size:
            raise ValueError(
                f"hidden_states phải có shape [B, T_frame, {self.hidden_size}]"
            )

        reversed_states = self.grl(hidden_states)
        # [B, T_frame, H] -> identical forward values [B, T_frame, H].
        pooled_states = masked_mean_pool(
            reversed_states,  # [B, T_frame, H].
            feature_attention_mask,  # [B, T_frame] or None.
        )  # [B, T_frame, H] -> [B, H].
        logits = self.classifier(pooled_states)
        # [B, H] -> [B, bottleneck_size] -> [B, R].
        return logits, pooled_states
