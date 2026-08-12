"""Differentiable dialect embedding from region classification logits."""

from __future__ import annotations

from torch import Tensor, nn
from torch.nn import functional as F


class SoftDialectEmbedding(nn.Module):
    """Convert soft region posteriors into a weighted dialect embedding."""

    def __init__(
        self,
        num_regions: int = 3,
        embedding_dim: int = 64,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if num_regions <= 1:
            raise ValueError("num_regions phải > 1")
        if embedding_dim <= 0:
            raise ValueError("embedding_dim phải > 0")
        if temperature <= 0.0:
            raise ValueError("temperature phải > 0")

        self.num_regions = num_regions
        self.embedding_dim = embedding_dim
        self.temperature = temperature
        self.embedding = nn.Embedding(num_regions, embedding_dim)
        # Embedding weight has shape [num_regions, embedding_dim].
        nn.init.normal_(
            self.embedding.weight,
            mean=0.0,
            std=0.02,
        )

    def forward(self, dialect_logits: Tensor) -> tuple[Tensor, Tensor]:
        """Return ``(dialect_embedding [B, E], posterior [B, R])``."""
        if dialect_logits.ndim != 2:
            raise ValueError("dialect_logits phải có shape [B, num_regions]")
        if dialect_logits.shape[-1] != self.num_regions:
            raise ValueError(
                "Số region logits không khớp: "
                f"cần {self.num_regions}, nhận {dialect_logits.shape[-1]}"
            )

        posterior = F.softmax(
            dialect_logits / self.temperature,
            dim=-1,
        )  # [B, num_regions] -> posterior [B, num_regions].
        dialect_embedding = posterior @ self.embedding.weight
        # [B, num_regions] @ [num_regions, embedding_dim] -> [B, embedding_dim].

        return dialect_embedding, posterior
