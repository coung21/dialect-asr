"""Model registry and construction helpers for ASR architectures."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from transformers import WhisperConfig

from .base_model import AbstractPhoWhisperASR
from .model import BaselinePhoWhisperASR


MODEL_REGISTRY: dict[str, type[AbstractPhoWhisperASR]] = {
    BaselinePhoWhisperASR.architecture_name(): BaselinePhoWhisperASR,
}


def get_model_class(architecture: str) -> type[AbstractPhoWhisperASR]:
    """Return the model class registered for an architecture name."""
    try:
        return MODEL_REGISTRY[architecture]
    except KeyError as exc:
        supported = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(
            f"Architecture không hợp lệ {architecture!r}; hỗ trợ: {supported}"
        ) from exc


def architecture_from_checkpoint(
    checkpoint: str,
    *,
    fallback: str = "baseline",
    local_files_only: bool = False,
) -> str:
    """Read the saved project architecture, falling back for old baselines."""
    config = WhisperConfig.from_pretrained(
        checkpoint,
        local_files_only=local_files_only,
    )
    architecture = str(getattr(config, "architecture", fallback))
    get_model_class(architecture)  # Validate before model construction.
    return architecture


def build_project_model(
    *,
    architecture: str,
    source: str,
    evaluation: bool,
    model_options: Mapping[str, Any],
    freeze_encoder: bool = True,
    gradient_checkpointing: bool = False,
    seed: int = 42,
    full_determinism: bool = False,
    local_files_only: bool = False,
) -> AbstractPhoWhisperASR:
    """Load either a final checkpoint or pretrained initialization."""
    del model_options  # Reserved for future architecture-specific config.
    model_class = get_model_class(architecture)
    if evaluation:
        return model_class.from_pretrained(
            source,
            local_files_only=local_files_only,
        )

    return model_class.from_vietnamese_pretrained(
        pretrained_model_name=source,
        freeze_encoder=freeze_encoder,
        gradient_checkpointing=gradient_checkpointing,
        seed=seed,
        full_determinism=full_determinism,
        local_files_only=local_files_only,
    )
