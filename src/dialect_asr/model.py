"""Baseline PhoWhisper seq2seq model and Vietnamese processor loading."""

from __future__ import annotations

from typing import Any

from transformers import WhisperProcessor

from .base_model import AbstractPhoWhisperASR, DEFAULT_PRETRAINED_MODEL


class BaselinePhoWhisperASR(AbstractPhoWhisperASR):
    """PhoWhisper seq2seq baseline that remains compatible with HF ``Seq2SeqTrainer``.

    Shape contract of the inherited ``forward`` method:

    - ``input_features``: ``[B, num_mel_bins, T_frame]`` (log-mel spectrogram).
    - ``labels``: ``[B, T_text]`` with padding positions equal to ``-100``.
    - ``output.logits``: ``[B, T_text, V]``.
    - ``output.loss``: scalar tensor ``[]`` when labels are provided.

    ``T_frame`` is fixed by the feature extractor (30s of audio by default).
    ``V`` is the decoder vocabulary size.
    """

    @classmethod
    def architecture_name(cls) -> str:
        """Return the Hydra/registry identifier for the unchanged baseline."""
        return "baseline"


def load_vietnamese_processor(
    pretrained_model_name: str = DEFAULT_PRETRAINED_MODEL,
    **from_pretrained_kwargs: Any,
) -> WhisperProcessor:
    """Load the feature extractor and tokenizer for Vietnamese transcription."""
    return WhisperProcessor.from_pretrained(
        pretrained_model_name,
        **from_pretrained_kwargs,
    )
