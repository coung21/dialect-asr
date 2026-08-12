"""Abstract model contract shared by Wav2Vec2 CTC architectures."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Self

from transformers import Wav2Vec2ForCTC

from .reproducibility import DEFAULT_SEED, seed_everything


DEFAULT_PRETRAINED_MODEL = "nguyenvulebinh/wav2vec2-base-vi-vlsp2020"


class AbstractWav2Vec2CTC(Wav2Vec2ForCTC, ABC):
    """Common lifecycle for every project Wav2Vec2 CTC architecture.

    Subclasses keep the Hugging Face model contract:

    - ``input_values``: ``[B, T_audio]``.
    - ``attention_mask``: ``[B, T_audio]`` (optional).
    - ``labels``: ``[B, T_text]`` with padding positions equal to ``-100``.
    - ``output.logits``: ``[B, T_frame, V]``.
    - ``output.loss``: scalar tensor ``[]`` when labels are provided.
    """

    @classmethod
    @abstractmethod
    def architecture_name(cls) -> str:
        """Return the stable name used to select this architecture in config."""

    @classmethod
    def from_vietnamese_pretrained(
        cls,
        pretrained_model_name: str = DEFAULT_PRETRAINED_MODEL,
        *,
        freeze_feature_encoder: bool = True,
        freeze_base_model: bool = False,
        gradient_checkpointing: bool = False,
        seed: int = DEFAULT_SEED,
        full_determinism: bool = False,
        **from_pretrained_kwargs: Any,
    ) -> Self:
        """Load a checkpoint and apply shared fine-tuning configuration."""
        if freeze_feature_encoder and freeze_base_model:
            raise ValueError(
                "Chỉ chọn một trong freeze_feature_encoder hoặc freeze_base_model"
            )

        # Seed before construction so every newly initialized parameter is stable.
        seed_everything(seed, deterministic=full_determinism)
        model = cls.from_pretrained(
            pretrained_model_name,
            **from_pretrained_kwargs,
        )

        if freeze_base_model:
            model.freeze_base_model()
        elif freeze_feature_encoder:
            model.freeze_feature_encoder()

        if gradient_checkpointing:
            model.gradient_checkpointing_enable()

        return model

    def parameter_counts(self) -> dict[str, int]:
        """Return total, trainable and frozen parameter counts."""
        # Each parameter tensor [*] is reduced by numel() to one scalar count [].
        total = sum(parameter.numel() for parameter in self.parameters())
        # Parameter tensors [*] requiring gradients are reduced to one count [].
        trainable = sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )
        return {
            "total": total,
            "trainable": trainable,
            "frozen": total - trainable,
        }
