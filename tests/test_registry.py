from types import SimpleNamespace

import pytest

from dialect_asr.model import BaselinePhoWhisperASR
from dialect_asr.registry import (
    MODEL_REGISTRY,
    architecture_from_checkpoint,
    build_project_model,
    get_model_class,
)


def test_registry_contains_baseline() -> None:
    assert MODEL_REGISTRY == {"baseline": BaselinePhoWhisperASR}
    assert get_model_class("baseline") is BaselinePhoWhisperASR


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
