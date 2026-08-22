"""Adaptive LayerNorm (AdaLN) conditioning for the PhoWhisper encoder.

Each encoder layer's two ``LayerNorm``s are swapped for :class:`AdaLN`, which
keeps the same ``weight``/``bias`` parameters (so a pretrained checkpoint's
LayerNorm state loads unchanged) and adds a per-utterance affine correction
predicted from a conditioning vector — here, a frozen dialect-identification
(DID) embedding. The correction head is zero-initialized (DiT-style), so at
the start of fine-tuning ``AdaLN(x, cond) == LayerNorm(x)`` for *any* ``cond``,
and the model only starts leaning on dialect conditioning once training moves
the head away from zero.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class AdaLN(nn.Module):
    """Drop-in replacement for ``nn.LayerNorm`` conditioned on an extra vector.

    ``forward(x, cond) = LayerNorm(x) * (1 + gamma(cond)) + beta(cond)``, with
    ``gamma``/``beta`` predicted by a zero-initialized linear layer so the
    module starts out numerically identical to a plain ``LayerNorm``.
    """

    def __init__(self, normalized_shape: int, cond_dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        if normalized_shape <= 0:
            raise ValueError("normalized_shape phải > 0")
        if cond_dim <= 0:
            raise ValueError("cond_dim phải > 0")

        self.normalized_shape = normalized_shape
        self.eps = eps
        # Same parameter names as `nn.LayerNorm` so a checkpoint saved for the
        # plain layer ("...norm.weight"/"...norm.bias") loads unchanged here.
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

        self.to_gamma_beta = nn.Linear(cond_dim, 2 * normalized_shape)
        nn.init.zeros_(self.to_gamma_beta.weight)
        nn.init.zeros_(self.to_gamma_beta.bias)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        """Apply conditioned LayerNorm.

        Args:
            x: Hidden states with shape ``[B, T, H]`` (``H == normalized_shape``).
            cond: Conditioning vector with shape ``[B, cond_dim]``.
        """
        if x.shape[-1] != self.normalized_shape:
            raise ValueError(
                "x phải có chiều cuối = normalized_shape: "
                f"cần {self.normalized_shape}, nhận {x.shape[-1]}"
            )
        if cond.ndim != 2 or cond.shape[0] != x.shape[0]:
            raise ValueError(
                "cond phải có shape [B, cond_dim] khớp batch size với x: "
                f"nhận {tuple(cond.shape)} với x batch={x.shape[0]}"
            )

        normed = F.layer_norm(x, (self.normalized_shape,), self.weight, self.bias, self.eps)
        # [B, cond_dim] -> [B, 2*H] -> 2x [B, H].
        gamma, beta = self.to_gamma_beta(cond).chunk(2, dim=-1)
        gamma = gamma.unsqueeze(1)  # [B, H] -> [B, 1, H], broadcast over T.
        beta = beta.unsqueeze(1)
        return normed * (1 + gamma) + beta  # [B, T, H].


class DialectConditioner(nn.Module):
    """Project a frozen DID embedding into the encoder's AdaLN conditioning space."""

    def __init__(self, did_embedding_size: int, cond_dim: int) -> None:
        super().__init__()
        if did_embedding_size <= 0:
            raise ValueError("did_embedding_size phải > 0")
        if cond_dim <= 0:
            raise ValueError("cond_dim phải > 0")

        self.proj = nn.Sequential(
            nn.Linear(did_embedding_size, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

    def forward(self, did_embedding: Tensor) -> Tensor:
        """``[B, did_embedding_size] -> [B, cond_dim]``."""
        return self.proj(did_embedding)
