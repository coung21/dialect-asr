from types import SimpleNamespace

import pytest

from dialect_asr.dggfm_model import DGGFMWav2Vec2CTC
from dialect_asr.model import BaselineWav2Vec2CTC
from dialect_asr.registry import (
    MODEL_REGISTRY,
    architecture_from_checkpoint,
    build_project_model,
    get_model_class,
)


def test_registry_contains_baseline_and_dggfm() -> None:
    assert MODEL_REGISTRY == {
        "baseline": BaselineWav2Vec2CTC,
        "dggfm": DGGFMWav2Vec2CTC,
    }
    assert get_model_class("baseline") is BaselineWav2Vec2CTC
    assert get_model_class("dggfm") is DGGFMWav2Vec2CTC


def test_registry_rejects_unknown_architecture() -> None:
    with pytest.raises(ValueError, match="Architecture không hợp lệ"):
        get_model_class("unknown")


def test_architecture_is_read_from_checkpoint_config(monkeypatch) -> None:
    monkeypatch.setattr(
        "dialect_asr.registry.Wav2Vec2Config.from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(architecture="dggfm"),
    )

    assert architecture_from_checkpoint("checkpoint") == "dggfm"


def test_old_checkpoint_without_architecture_uses_fallback(monkeypatch) -> None:
    monkeypatch.setattr(
        "dialect_asr.registry.Wav2Vec2Config.from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(),
    )

    assert architecture_from_checkpoint("checkpoint", fallback="baseline") == "baseline"


def test_build_dggfm_model_maps_project_options_to_hf_config(monkeypatch) -> None:
    pretrained_config = SimpleNamespace()
    sentinel = object()
    received: dict[str, object] = {}
    monkeypatch.setattr(
        "dialect_asr.registry.Wav2Vec2Config.from_pretrained",
        lambda *args, **kwargs: pretrained_config,
    )

    def fake_factory(**kwargs):
        received.update(kwargs)
        return sentinel

    monkeypatch.setattr(
        DGGFMWav2Vec2CTC,
        "from_vietnamese_pretrained",
        fake_factory,
    )

    model = build_project_model(
        architecture="dggfm",
        source="pretrained",
        evaluation=False,
        model_options={
            "branch_block": 6,
            "fusion_blocks": [6, 8, 10, 12],
            "dialect_dim": 64,
            "gate_hidden_dim": 256,
        },
    )

    assert model is sentinel
    assert pretrained_config.dggfm_branch_block == 6
    assert pretrained_config.dggfm_fusion_blocks == [6, 8, 10, 12]
    assert pretrained_config.dialect_dim == 64
    assert pretrained_config.dggfm_gate_hidden_dim == 256
    assert received["config"] is pretrained_config


def test_eval_uses_registered_checkpoint_class(monkeypatch) -> None:
    sentinel = object()
    received: dict[str, object] = {}

    def fake_from_pretrained(source, **kwargs):
        received.update(source=source, kwargs=kwargs)
        return sentinel

    monkeypatch.setattr(DGGFMWav2Vec2CTC, "from_pretrained", fake_from_pretrained)

    model = build_project_model(
        architecture="dggfm",
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
