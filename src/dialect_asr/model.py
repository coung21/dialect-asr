"""Baseline Wav2Vec2 CTC model and Vietnamese processor loading."""

from __future__ import annotations

from typing import Any

from transformers import Wav2Vec2Processor

from .base_model import AbstractWav2Vec2CTC, DEFAULT_PRETRAINED_MODEL


class BaselineWav2Vec2CTC(AbstractWav2Vec2CTC):
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
    def architecture_name(cls) -> str:
        """Return the Hydra/registry identifier for the unchanged baseline."""
        return "baseline"


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
