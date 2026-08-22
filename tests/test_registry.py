from types import SimpleNamespace

import pytest

from dialect_asr.adaln_model import PhoWhisperAdaLNASR
from dialect_asr.model import BaselinePhoWhisperASR
from dialect_asr.registry import (
    MODEL_REGISTRY,
    architecture_from_checkpoint,
    build_project_model,
    get_model_class,
)


def test_registry_contains_baseline_and_adaln() -> None:
    assert MODEL_REGISTRY == {
        "baseline": BaselinePhoWhisperASR,
        "adaln": PhoWhisperAdaLNASR,
    }
    assert get_model_class("baseline") is BaselinePhoWhisperASR
    assert get_model_class("adaln") is PhoWhisperAdaLNASR


def test_registry_rejects_unknown_architecture() -> None:
    with pytest.raises(ValueError, match="Architecture không hợp lệ"):
        get_model_class("unknown")


def test_architecture_is_read_from_checkpoint_config(monkeypatch) -> None:
    monkeypatch.setattr(
        "dialect_asr.registry.WhisperConfig.from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(architecture="baseline"),
    )

    assert architecture_from_checkpoint("checkpoint") == "baseline"


def test_old_checkpoint_without_architecture_uses_fallback(monkeypatch) -> None:
    monkeypatch.setattr(
        "dialect_asr.registry.WhisperConfig.from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(),
    )

    assert architecture_from_checkpoint("checkpoint", fallback="baseline") == "baseline"


def test_architecture_from_checkpoint_rejects_unknown_architecture(monkeypatch) -> None:
    monkeypatch.setattr(
        "dialect_asr.registry.WhisperConfig.from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(architecture="unknown"),
    )

    with pytest.raises(ValueError, match="Architecture không hợp lệ"):
        architecture_from_checkpoint("checkpoint")


def test_build_train_model_uses_vietnamese_pretrained_factory(monkeypatch) -> None:
    sentinel = object()
    received: dict[str, object] = {}

    def fake_factory(**kwargs):
        received.update(kwargs)
        return sentinel

    monkeypatch.setattr(
        BaselinePhoWhisperASR,
        "from_vietnamese_pretrained",
        fake_factory,
    )

    model = build_project_model(
        architecture="baseline",
        source="pretrained",
        evaluation=False,
        model_options={},
        freeze_encoder=False,
        gradient_checkpointing=True,
        seed=7,
        full_determinism=True,
        local_files_only=True,
    )

    assert model is sentinel
    assert received == {
        "pretrained_model_name": "pretrained",
        "freeze_encoder": False,
        "gradient_checkpointing": True,
        "seed": 7,
        "full_determinism": True,
        "local_files_only": True,
    }


def test_eval_uses_registered_checkpoint_class(monkeypatch) -> None:
    sentinel = object()
    received: dict[str, object] = {}

    def fake_from_pretrained(source, **kwargs):
        received.update(source=source, kwargs=kwargs)
        return sentinel

    monkeypatch.setattr(BaselinePhoWhisperASR, "from_pretrained", fake_from_pretrained)

    model = build_project_model(
        architecture="baseline",
        source="checkpoint",
        evaluation=True,
        model_options={},
        local_files_only=True,
    )

    assert model is sentinel
    assert received == {
        "source": "checkpoint",
        "kwargs": {"local_files_only": True},
    }


def test_build_model_strips_standard_config_keys_from_model_options(monkeypatch) -> None:
    """`model_options` mirrors the whole `cfg.model` mapping; only the extra
    (architecture-specific) keys should reach the model class factory."""
    sentinel = object()
    received: dict[str, object] = {}

    def fake_factory(**kwargs):
        received.update(kwargs)
        return sentinel

    monkeypatch.setattr(BaselinePhoWhisperASR, "from_vietnamese_pretrained", fake_factory)

    build_project_model(
        architecture="baseline",
        source="pretrained",
        evaluation=False,
        model_options={
            "architecture": "baseline",
            "pretrained_model_name": "pretrained",
            "freeze_encoder": False,
            "gradient_checkpointing": True,
            "local_files_only": True,
            "did_kwargs": {"channels": 8},
        },
        freeze_encoder=False,
        gradient_checkpointing=True,
        seed=7,
        full_determinism=True,
        local_files_only=True,
    )

    assert received["did_kwargs"] == {"channels": 8}
    assert "architecture" not in received
    # "pretrained_model_name" legitimately reaches the factory as `source`,
    # forwarded explicitly (not duplicated from `model_options`).
    assert received["pretrained_model_name"] == "pretrained"


def test_build_eval_model_drops_train_only_options(monkeypatch) -> None:
    sentinel = object()
    received: dict[str, object] = {}

    def fake_from_pretrained(source, **kwargs):
        received.update(kwargs)
        return sentinel

    monkeypatch.setattr(PhoWhisperAdaLNASR, "from_pretrained", fake_from_pretrained)

    build_project_model(
        architecture="adaln",
        source="checkpoint",
        evaluation=True,
        model_options={
            "did_checkpoint_path": "outputs/did-ecapa-tdnn/final_model.pt",
            "did_kwargs": {"channels": 8},
        },
        local_files_only=True,
    )

    assert "did_checkpoint_path" not in received
    assert received["did_kwargs"] == {"channels": 8}
