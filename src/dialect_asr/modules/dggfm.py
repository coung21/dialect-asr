"""Dialect-Guided Gated Feature Modulation module."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .dialect_branch import masked_mean_pool


class DGGFM(nn.Module):
    """Apply gated dialect-conditioned residual modulation to encoder states."""

    def __init__(
        self,
        hidden_size: int,
        dialect_dim: int = 64,
        gate_hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("hidden_size phải > 0")
        if dialect_dim <= 0:
            raise ValueError("dialect_dim phải > 0")
        if gate_hidden_dim <= 0:
            raise ValueError("gate_hidden_dim phải > 0")

        self.hidden_size = hidden_size
        self.dialect_dim = dialect_dim
        self.gate_hidden_dim = gate_hidden_dim
        self.norm = nn.LayerNorm(hidden_size)

        # [B, dialect_dim] -> delta-gamma [B, hidden_size].
        self.to_gamma = nn.Linear(dialect_dim, hidden_size)
        # [B, dialect_dim] -> beta [B, hidden_size].
        self.to_beta = nn.Linear(dialect_dim, hidden_size)

        self.gate_network = nn.Sequential(
            # [B, dialect_dim + hidden_size] -> [B, gate_hidden_dim].
            nn.Linear(dialect_dim + hidden_size, gate_hidden_dim),
            # GELU preserves shape [B, gate_hidden_dim].
            nn.GELU(),
            # [B, gate_hidden_dim] -> gate logits [B, hidden_size].
            nn.Linear(gate_hidden_dim, hidden_size),
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize modulation as identity and gates as mostly closed."""
        # Both parameter tensors map [dialect_dim] -> [hidden_size].
        nn.init.zeros_(self.to_gamma.weight)
        nn.init.zeros_(self.to_gamma.bias)
        nn.init.zeros_(self.to_beta.weight)
        nn.init.zeros_(self.to_beta.bias)

        final_gate_layer = self.gate_network[-1]
        if not isinstance(final_gate_layer, nn.Linear):
            raise TypeError("Layer cuối của gate_network phải là nn.Linear")
        # Final gate maps [B, gate_hidden_dim] -> [B, hidden_size].
        nn.init.zeros_(final_gate_layer.weight)
        nn.init.constant_(final_gate_layer.bias, -2.0)

    def forward(
        self,
        hidden_states: Tensor,
        dialect_embedding: Tensor,
        feature_attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Return modulated states and detached gate summary statistics.

        Args:
            hidden_states: Encoder features ``[B, T_frame, H]``.
            dialect_embedding: Soft dialect embedding ``[B, E]``.
            feature_attention_mask: Optional valid-frame mask ``[B, T_frame]``.
        """
        if hidden_states.ndim != 3:
            raise ValueError("hidden_states phải có shape [B, T_frame, H]")
        if hidden_states.shape[-1] != self.hidden_size:
            raise ValueError(
                "Hidden size không khớp: "
                f"cần {self.hidden_size}, nhận {hidden_states.shape[-1]}"
            )
        if dialect_embedding.ndim != 2:
            raise ValueError("dialect_embedding phải có shape [B, dialect_dim]")
        if dialect_embedding.shape[0] != hidden_states.shape[0]:
            raise ValueError("Batch size của hidden_states và dialect_embedding phải bằng nhau")
        if dialect_embedding.shape[-1] != self.dialect_dim:
            raise ValueError(
                "Dialect embedding size không khớp: "
                f"cần {self.dialect_dim}, nhận {dialect_embedding.shape[-1]}"
            )

        normalized = self.norm(hidden_states)
        # LayerNorm preserves encoder shape [B, T_frame, H].
        gamma = self.to_gamma(dialect_embedding).unsqueeze(1)
        # [B, E] -> [B, H] -> [B, 1, H].
        beta = self.to_beta(dialect_embedding).unsqueeze(1)
        # [B, E] -> [B, H] -> [B, 1, H].

        modulation = gamma * normalized + beta
        # [B, 1, H] * [B, T_frame, H] + [B, 1, H] -> [B, T_frame, H].

        h_pool = masked_mean_pool(
            hidden_states,  # [B, T_frame, H].
            feature_attention_mask,  # [B, T_frame] or None.
        )  # [B, T_frame, H] -> [B, H].
        gate_input = torch.cat(
            [dialect_embedding, h_pool],
            dim=-1,
        )  # [B, E] concatenated with [B, H] -> [B, E + H].
        gate_logits = self.gate_network(gate_input)
        # [B, E + H] -> [B, gate_hidden_dim] -> [B, H].
        gate = torch.sigmoid(gate_logits).unsqueeze(1)
        # [B, H] -> sigmoid [B, H] -> [B, 1, H].

        output = hidden_states + gate * modulation
        # [B, T_frame, H] + [B, 1, H] * [B, T_frame, H] -> [B, T_frame, H].

        statistics = {
            "gate": gate,  # [B, 1, H].
            "gate_mean": gate.mean().detach(),  # [B, 1, H] -> scalar [].
            "gate_std": gate.std().detach(),  # [B, 1, H] -> scalar [].
        }

        return output, statistics
