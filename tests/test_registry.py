from types import SimpleNamespace

import pytest

from dialect_asr.dann_model import DANNWav2Vec2CTC
from dialect_asr.model import BaselineWav2Vec2CTC
from dialect_asr.registry import build_project_model, get_model_class


def test_registry_selects_baseline_and_dann() -> None:
    assert get_model_class("baseline") is BaselineWav2Vec2CTC
    assert get_model_class("dann") is DANNWav2Vec2CTC


def test_registry_rejects_unknown_architecture() -> None:
    with pytest.raises(ValueError, match="Architecture không hợp lệ"):
        get_model_class("unknown")


def test_build_dann_maps_hydra_options_into_pretrained_config(monkeypatch) -> None:
    pretrained_config = SimpleNamespace()
    sentinel = object()
    received = {}

    monkeypatch.setattr(
        "dialect_asr.registry.Wav2Vec2Config.from_pretrained",
        lambda *_args, **_kwargs: pretrained_config,
    )

    def fake_factory(**kwargs):
        received.update(kwargs)
        return sentinel

    monkeypatch.setattr(
        DANNWav2Vec2CTC,
        "from_vietnamese_pretrained",
        fake_factory,
    )
    model = build_project_model(
        architecture="dann",
        source="owner/checkpoint",
        evaluation=False,
        model_options={
            "num_regions": 3,
            "dialect_bottleneck_size": 256,
            "dialect_dropout": 0.1,
            "dialect_loss_weight": 1.0,
            "grl_scale": 0.1,
        },
    )

    assert model is sentinel
    assert received["config"] is pretrained_config
    assert pretrained_config.num_regions == 3
    assert pretrained_config.dialect_loss_weight == pytest.approx(1.0)
    assert pretrained_config.grl_scale == pytest.approx(0.1)
