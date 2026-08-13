"""Model registry and checkpoint-aware construction helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from transformers import Wav2Vec2Config

from .dann_model import DANNWav2Vec2CTC
from .model import BaselineWav2Vec2CTC


MODEL_REGISTRY = {
    "baseline": BaselineWav2Vec2CTC,
    "dann": DANNWav2Vec2CTC,
}

DANN_CONFIG_MAPPING = {
    "num_regions": "num_regions",
    "dialect_bottleneck_size": "dialect_bottleneck_size",
    "dialect_dropout": "dialect_dropout",
    "dialect_loss_weight": "dialect_loss_weight",
    "grl_scale": "grl_scale",
}


def get_model_class(architecture: str) -> type[BaselineWav2Vec2CTC]:
    """Return the registered model class for an architecture name."""
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
    """Read architecture stored in a checkpoint, with legacy fallback."""
    config = Wav2Vec2Config.from_pretrained(
        checkpoint,
        local_files_only=local_files_only,
    )
    architecture = str(getattr(config, "architecture", fallback))
    get_model_class(architecture)
    return architecture


def build_project_model(
    *,
    architecture: str,
    source: str,
    evaluation: bool,
    model_options: Mapping[str, Any],
    freeze_feature_encoder: bool = True,
    freeze_base_model: bool = False,
    gradient_checkpointing: bool = False,
    seed: int = 42,
    full_determinism: bool = False,
    local_files_only: bool = False,
) -> BaselineWav2Vec2CTC:
    """Build a baseline or DANN model from pretrained/final checkpoint."""
    model_class = get_model_class(architecture)
    if evaluation:
        return model_class.from_pretrained(
            source,
            local_files_only=local_files_only,
        )

    pretrained_kwargs: dict[str, Any] = {}
    if architecture == "dann":
        pretrained_config = Wav2Vec2Config.from_pretrained(
            source,
            local_files_only=local_files_only,
        )
        for option_name, config_name in DANN_CONFIG_MAPPING.items():
            if option_name in model_options:
                setattr(pretrained_config, config_name, model_options[option_name])
        pretrained_kwargs["config"] = pretrained_config

    return model_class.from_vietnamese_pretrained(
        pretrained_model_name=source,
        freeze_feature_encoder=freeze_feature_encoder,
        freeze_base_model=freeze_base_model,
        gradient_checkpointing=gradient_checkpointing,
        seed=seed,
        full_determinism=full_determinism,
        local_files_only=local_files_only,
        **pretrained_kwargs,
    )
