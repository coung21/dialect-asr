"""Reusable neural-network modules for dialect-aware ASR models."""

from .dialect_branch import DialectBranch, masked_mean_pool
from .dggfm import DGGFM
from .soft_embedding import SoftDialectEmbedding

__all__ = ["DGGFM", "DialectBranch", "SoftDialectEmbedding", "masked_mean_pool"]
