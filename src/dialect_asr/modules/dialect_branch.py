"""Auxiliary dialect classification branch for encoder hidden states."""

from __future__ import annotations

from torch import Tensor, nn


class DialectBranch(nn.Module):
    """Predict a dialect from a sequence of Wav2Vec2 hidden states.

    The temporal dimension is reduced with masked mean pooling so padded audio
    frames do not contribute to the utterance representation.
    """

    def __init__(
        self,
        hidden_size: int,
        num_dialects: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("hidden_size phải > 0")
        if num_dialects <= 1:
            raise ValueError("num_dialects phải > 1")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout phải nằm trong khoảng [0, 1)")

        self.hidden_size = hidden_size
        self.num_dialects = num_dialects
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_dialects)

    def forward(
        self,
        hidden_states: Tensor,
        frame_mask: Tensor | None = None,
    ) -> Tensor:
        """Return dialect logits with shape ``[B, num_dialects]``.

        Args:
            hidden_states: Encoder features with shape ``[B, T_frame, H]``.
            frame_mask: Optional valid-frame mask with shape ``[B, T_frame]``.
        """
        if hidden_states.ndim != 3:
            raise ValueError("hidden_states phải có shape [B, T_frame, H]")
        if hidden_states.shape[-1] != self.hidden_size:
            raise ValueError(
                "Hidden size không khớp: "
                f"cần {self.hidden_size}, nhận {hidden_states.shape[-1]}"
            )

        if frame_mask is None:
            # [B, T_frame, H] -> mean over T_frame -> [B, H].
            pooled_states = hidden_states.mean(dim=1)
        else:
            # [B, T_frame, H] -> expected mask shape [B, T_frame].
            expected_shape = hidden_states.shape[:2]
            if frame_mask.shape != expected_shape:
                raise ValueError(
                    "frame_mask phải có shape [B, T_frame]; "
                    f"cần {tuple(expected_shape)}, nhận {tuple(frame_mask.shape)}"
                )

            # [B, T_frame] -> [B, T_frame, 1] for broadcasting over H.
            float_mask = frame_mask.to(
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            ).unsqueeze(-1)
            # [B, T_frame, H] * [B, T_frame, 1] -> sum over T_frame -> [B, H].
            summed_states = (hidden_states * float_mask).sum(dim=1)
            # [B, T_frame, 1] -> sum over T_frame -> [B, 1].
            valid_lengths = float_mask.sum(dim=1).clamp_min(1.0)
            # [B, H] / [B, 1] -> [B, H].
            pooled_states = summed_states / valid_lengths

        # Dropout preserves the utterance representation shape [B, H].
        pooled_states = self.dropout(pooled_states)
        # [B, H] -> dialect logits [B, num_dialects].
        return self.classifier(pooled_states)
