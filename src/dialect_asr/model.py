"""Baseline Wav2Vec2 CTC model for Vietnamese speech recognition."""

from __future__ import annotations

from typing import Any, Self

from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from .reproducibility import DEFAULT_SEED, seed_everything


DEFAULT_PRETRAINED_MODEL = "nguyenvulebinh/wav2vec2-base-vi-vlsp2020"


class BaselineWav2Vec2CTC(Wav2Vec2ForCTC):
    """Wav2Vec2 CTC baseline that remains compatible with HF ``Trainer``.

    Shape contract of the inherited ``forward`` method:

    - ``input_values``: ``[B, T_audio]``.
    - ``attention_mask``: ``[B, T_audio]`` (optional).
    - ``labels``: ``[B, T_text]`` with padding positions equal to ``-100``.
    - ``output.logits``: ``[B, T_frame, V]``.
    - ``output.loss``: scalar tensor ``[]`` when labels are provided.

    ``T_frame`` is shorter than ``T_audio`` because the convolutional feature
    encoder downsamples the waveform. ``V`` is the CTC vocabulary size (98 for
    the default Vietnamese checkpoint).
    """

    @classmethod
    def from_vietnamese_pretrained(
        cls,
        pretrained_model_name: str = DEFAULT_PRETRAINED_MODEL,
        *,
        freeze_feature_encoder: bool = True,
        freeze_base_model: bool = False,
        gradient_checkpointing: bool = False,
        seed: int = DEFAULT_SEED,
        **from_pretrained_kwargs: Any,
    ) -> Self:
        """Load the Vietnamese checkpoint and configure fine-tuning behavior.

        ``freeze_feature_encoder`` freezes only the convolutional audio feature
        extractor. ``freeze_base_model`` freezes the entire Wav2Vec2 encoder and
        trains only the CTC projection head. The two modes are mutually exclusive.
        """
        if freeze_feature_encoder and freeze_base_model:
            raise ValueError(
                "Chỉ chọn một trong freeze_feature_encoder hoặc freeze_base_model"
            )

        # Seed before model construction so any newly initialized tensor is stable.
        seed_everything(seed)
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
        """Return total and trainable parameter counts for experiment logging."""
        # numel() flattens each parameter conceptually: parameter[*] -> scalar count.
        total = sum(parameter.numel() for parameter in self.parameters())
        # Same scalar reduction, restricted to tensors that receive gradients.
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


def load_vietnamese_processor(
    pretrained_model_name: str = DEFAULT_PRETRAINED_MODEL,
    **from_pretrained_kwargs: Any,
) -> Wav2Vec2Processor:
    """Load feature extractor and tokenizer without the optional n-gram LM.

    Fine-tuning and greedy CTC decoding do not require ``pyctcdecode`` or the
    large language-model files included in the checkpoint repository.
    """
    return Wav2Vec2Processor.from_pretrained(
        pretrained_model_name,
        **from_pretrained_kwargs,
    )
