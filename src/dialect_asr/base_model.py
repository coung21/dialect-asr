"""Abstract model contract shared by PhoWhisper seq2seq architectures."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Self

from transformers import WhisperForConditionalGeneration

from .reproducibility import DEFAULT_SEED, seed_everything


DEFAULT_PRETRAINED_MODEL = "vinai/PhoWhisper-base"


class AbstractPhoWhisperASR(WhisperForConditionalGeneration, ABC):
    """Common lifecycle for every project PhoWhisper seq2seq architecture.

    Subclasses keep the Hugging Face model contract:

    - ``input_features``: ``[B, num_mel_bins, T_frame]`` (log-mel spectrogram).
    - ``labels``: ``[B, T_text]`` with padding positions equal to ``-100``.
    - ``output.logits``: ``[B, T_text, V]``.
    - ``output.loss``: scalar tensor ``[]`` when labels are provided.

    ``decoder_input_ids`` are derived from ``labels`` automatically by
    right-shifting, so callers only need to supply the target token IDs.
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
        freeze_encoder: bool = True,
        gradient_checkpointing: bool = False,
        seed: int = DEFAULT_SEED,
        full_determinism: bool = False,
        **from_pretrained_kwargs: Any,
    ) -> Self:
        """Load a checkpoint and apply shared fine-tuning configuration."""
        # Seed before construction so every newly initialized parameter is stable.
        seed_everything(seed, deterministic=full_determinism)
        model = cls.from_pretrained(
            pretrained_model_name,
            **from_pretrained_kwargs,
        )

        if freeze_encoder:
            # Freezes the whole Whisper encoder (conv feature stack + transformer
            # blocks); only the decoder trains, which is the standard recipe for
            # fine-tuning Whisper on a lower-resource target language/domain.
            model.freeze_encoder()

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
