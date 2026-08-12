"""Auxiliary dialect classification branch for encoder hidden states."""

from __future__ import annotations

from torch import Tensor, nn


def masked_mean_pool(
    hidden_states: Tensor,
    feature_attention_mask: Tensor | None = None,
) -> Tensor:
    """Pool valid encoder frames into one representation per utterance.

    Args:
        hidden_states: Encoder features with shape ``[B, T_frame, H]``.
        feature_attention_mask: Optional valid-frame mask ``[B, T_frame]``.

    Returns:
        Mean-pooled representations with shape ``[B, H]``.
    """
    if hidden_states.ndim != 3:
        raise ValueError("hidden_states phải có shape [B, T_frame, H]")

    if feature_attention_mask is None:
        # [B, T_frame, H] -> mean over T_frame -> [B, H].
        return hidden_states.mean(dim=1)

    # [B, T_frame, H] -> expected mask shape [B, T_frame].
    expected_mask_shape = hidden_states.shape[:2]
    if feature_attention_mask.shape != expected_mask_shape:
        raise ValueError(
            "feature_attention_mask phải có shape [B, T_frame]; "
            f"cần {tuple(expected_mask_shape)}, "
            f"nhận {tuple(feature_attention_mask.shape)}"
        )

    # [B, T_frame] -> [B, T_frame, 1] for broadcasting over H.
    float_mask = feature_attention_mask.to(
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    ).unsqueeze(-1)
    # [B, T_frame, H] * [B, T_frame, 1] -> sum over T_frame -> [B, H].
    summed_states = (hidden_states * float_mask).sum(dim=1)
    # [B, T_frame, 1] -> sum over T_frame -> [B, 1].
    valid_lengths = float_mask.sum(dim=1).clamp_min(1.0)
    # [B, H] / [B, 1] -> [B, H].
    return summed_states / valid_lengths


class DialectBranch(nn.Module):
    """Predict a region and expose its pooled utterance representation."""

    def __init__(
        self,
        hidden_size: int,
        num_regions: int = 3,
        bottleneck_size: int = 256,
        dropout: float = 0.1,
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
        self.bottleneck_size = bottleneck_size
        self.classifier = nn.Sequential(
            # [B, H] -> [B, bottleneck_size].
            nn.Linear(hidden_size, bottleneck_size),
            # GELU preserves shape [B, bottleneck_size].
            nn.GELU(),
            # Dropout preserves shape [B, bottleneck_size].
            nn.Dropout(dropout),
            # [B, bottleneck_size] -> [B, num_regions].
            nn.Linear(bottleneck_size, num_regions),
        )

    def forward(
        self,
        hidden_states: Tensor,
        feature_attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Return ``(region_logits [B, R], pooled_states [B, H])``."""
        if hidden_states.ndim != 3:
            raise ValueError("hidden_states phải có shape [B, T_frame, H]")
        if hidden_states.shape[-1] != self.hidden_size:
            raise ValueError(
                "Hidden size không khớp: "
                f"cần {self.hidden_size}, nhận {hidden_states.shape[-1]}"
            )

        pooled_states = masked_mean_pool(
            hidden_states,  # [B, T_frame, H].
            feature_attention_mask,  # [B, T_frame] or None.
        )  # [B, T_frame, H] -> [B, H].
        logits = self.classifier(pooled_states)
        # [B, H] -> [B, bottleneck_size] -> [B, num_regions].

        return logits, pooled_states
