"""Reusable neural-network modules for dialect ASR experiments."""

from .dann_dialect_head import DANNDialectHead, masked_mean_pool
from .gradient_reversal import GradientReversal, reverse_gradient

__all__ = [
    "DANNDialectHead",
    "GradientReversal",
    "masked_mean_pool",
    "reverse_gradient",
]
