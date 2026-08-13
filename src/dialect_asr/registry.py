"""Model registry and construction helpers for ASR architectures."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from transformers import Wav2Vec2Config

from .base_model import AbstractWav2Vec2CTC
from .dggfm_model import DGGFMWav2Vec2CTC
from .model import BaselineWav2Vec2CTC
from .multitask_model import MultitaskWav2Vec2CTC


MODEL_REGISTRY: dict[str, type[AbstractWav2Vec2CTC]] = {
    BaselineWav2Vec2CTC.architecture_name(): BaselineWav2Vec2CTC,
    DGGFMWav2Vec2CTC.architecture_name(): DGGFMWav2Vec2CTC,
    MultitaskWav2Vec2CTC.architecture_name(): MultitaskWav2Vec2CTC,
}

DGGFM_CONFIG_MAPPING = {
    "branch_block": "dggfm_branch_block",
    "fusion_blocks": "dggfm_fusion_blocks",
    "num_regions": "num_regions",
    "dialect_bottleneck_size": "dialect_bottleneck_size",
    "dialect_dim": "dialect_dim",
    "gate_hidden_dim": "dggfm_gate_hidden_dim",
    "temperature": "dialect_temperature",
    "dialect_loss_weight": "dialect_loss_weight",
    "dialect_dropout": "dialect_dropout",
}

MULTITASK_CONFIG_MAPPING = {
    "branch_block": "multitask_branch_block",
    "num_regions": "num_regions",
    "dialect_bottleneck_size": "dialect_bottleneck_size",
    "dialect_dropout": "dialect_dropout",
    "dialect_loss_weight": "dialect_loss_weight",
}


def get_model_class(architecture: str) -> type[AbstractWav2Vec2CTC]:
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
    config = Wav2Vec2Config.from_pretrained(
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
    freeze_feature_encoder: bool = True,
    freeze_base_model: bool = False,
    gradient_checkpointing: bool = False,
    seed: int = 42,
    full_determinism: bool = False,
    local_files_only: bool = False,
) -> AbstractWav2Vec2CTC:
    """Load either a final checkpoint or pretrained initialization."""
    model_class = get_model_class(architecture)
    if evaluation:
        return model_class.from_pretrained(
            source,
            local_files_only=local_files_only,
        )

    pretrained_kwargs: dict[str, Any] = {}
    if architecture in {
        DGGFMWav2Vec2CTC.architecture_name(),
        MultitaskWav2Vec2CTC.architecture_name(),
    }:
        pretrained_config = Wav2Vec2Config.from_pretrained(
            source,
            local_files_only=local_files_only,
        )
        config_mapping = (
            DGGFM_CONFIG_MAPPING
            if architecture == DGGFMWav2Vec2CTC.architecture_name()
            else MULTITASK_CONFIG_MAPPING
        )
        for option_name, config_name in config_mapping.items():
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
