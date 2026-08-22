"""Model registry and construction helpers for ASR architectures."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from transformers import WhisperConfig

from .adaln_model import PhoWhisperAdaLNASR
from .base_model import AbstractPhoWhisperASR
from .model import BaselinePhoWhisperASR


MODEL_REGISTRY: dict[str, type[AbstractPhoWhisperASR]] = {
    BaselinePhoWhisperASR.architecture_name(): BaselinePhoWhisperASR,
    PhoWhisperAdaLNASR.architecture_name(): PhoWhisperAdaLNASR,
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


# `cfg.model`'s own fields, already forwarded as explicit keyword arguments;
# anything else in `model_options` is architecture-specific (e.g. "adaln"'s
# `did_checkpoint_path`/`did_kwargs`) and passed through as-is.
_STANDARD_MODEL_CONFIG_KEYS = {
    "architecture",
    "pretrained_model_name",
    "freeze_encoder",
    "gradient_checkpointing",
    "local_files_only",
}
# Only meaningful for `from_vietnamese_pretrained` (attaches a fresh DID
# checkpoint); an eval checkpoint's DID weights are already in its state
# dict, so passing this to `from_pretrained` would raise a TypeError.
_TRAIN_ONLY_MODEL_OPTION_KEYS = {"did_checkpoint_path"}


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
    model_class = get_model_class(architecture)
    extra_options = {
        key: value
        for key, value in model_options.items()
        if key not in _STANDARD_MODEL_CONFIG_KEYS
    }

    if evaluation:
        # `extra_options` (e.g. `did_kwargs` for "adaln") must reproduce the
        # architecture's shape so the saved checkpoint's state dict matches.
        eval_options = {
            key: value
            for key, value in extra_options.items()
            if key not in _TRAIN_ONLY_MODEL_OPTION_KEYS
        }
        return model_class.from_pretrained(
            source,
            local_files_only=local_files_only,
            **eval_options,
        )

    return model_class.from_vietnamese_pretrained(
        pretrained_model_name=source,
        freeze_encoder=freeze_encoder,
        gradient_checkpointing=gradient_checkpointing,
        seed=seed,
        full_determinism=full_determinism,
        local_files_only=local_files_only,
        **extra_options,
    )
