import inspect
from types import SimpleNamespace

import pytest
import torch
from transformers import Wav2Vec2Config

from dialect_asr.base_model import AbstractWav2Vec2CTC
from dialect_asr.model import (
    DEFAULT_PRETRAINED_MODEL,
    BaselineWav2Vec2CTC,
    load_vietnamese_processor,
)


def test_base_model_is_abstract_and_baseline_declares_architecture() -> None:
    assert inspect.isabstract(AbstractWav2Vec2CTC)
    assert BaselineWav2Vec2CTC.architecture_name() == "baseline"


def tiny_config() -> Wav2Vec2Config:
    return Wav2Vec2Config(
        vocab_size=12,
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=16,
        conv_dim=(8, 8, 8),
        conv_stride=(5, 2, 2),
        conv_kernel=(10, 3, 3),
        num_conv_pos_embedding_groups=2,
        num_conv_pos_embeddings=16,
        pad_token_id=0,
        ctc_zero_infinity=True,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        feat_proj_dropout=0.0,
        final_dropout=0.0,
    )


def test_forward_returns_ctc_loss_and_logits() -> None:
    model = BaselineWav2Vec2CTC(tiny_config())
    model.eval()
    input_values = torch.randn(2, 1_600)
    attention_mask = torch.ones_like(input_values, dtype=torch.long)
    labels = torch.tensor([[1, 2, 3], [4, 5, -100]])

    with torch.no_grad():
        output = model(
            input_values=input_values,
            attention_mask=attention_mask,
            labels=labels,
        )

    assert output.logits.shape[0] == 2
    assert output.logits.shape[-1] == 12
    assert output.loss.ndim == 0
    assert torch.isfinite(output.loss)


def test_pretrained_factory_uses_default_checkpoint_and_freezes_features(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_model = SimpleNamespace(
        feature_frozen=False,
        base_frozen=False,
        checkpointing=False,
    )
    fake_model.freeze_feature_encoder = lambda: setattr(
        fake_model, "feature_frozen", True
    )
    fake_model.freeze_base_model = lambda: setattr(fake_model, "base_frozen", True)
    fake_model.gradient_checkpointing_enable = lambda: setattr(
        fake_model, "checkpointing", True
    )
    received: dict[str, object] = {}

    def fake_from_pretrained(name: str, **kwargs):
        received.update(name=name, kwargs=kwargs)
        return fake_model

    monkeypatch.setattr(BaselineWav2Vec2CTC, "from_pretrained", fake_from_pretrained)

    model = BaselineWav2Vec2CTC.from_vietnamese_pretrained(
        gradient_checkpointing=True,
        full_determinism=False,
        local_files_only=True,
    )

    assert model is fake_model
    assert received == {
        "name": DEFAULT_PRETRAINED_MODEL,
        "kwargs": {"local_files_only": True},
    }
    assert fake_model.feature_frozen
    assert not fake_model.base_frozen
    assert fake_model.checkpointing


def test_freeze_modes_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="Chỉ chọn một"):
        BaselineWav2Vec2CTC.from_vietnamese_pretrained(
            freeze_feature_encoder=True,
            freeze_base_model=True,
        )


def test_parameter_counts() -> None:
    model = BaselineWav2Vec2CTC(tiny_config())
    model.lm_head.weight.requires_grad = False

    counts = model.parameter_counts()

    assert counts["total"] > 0
    assert counts["trainable"] > 0
    assert counts["frozen"] == model.lm_head.weight.numel()
    assert counts["total"] == counts["trainable"] + counts["frozen"]


def test_load_processor_avoids_lm_processor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = object()
    received: dict[str, object] = {}

    def fake_from_pretrained(name: str, **kwargs):
        received.update(name=name, kwargs=kwargs)
        return sentinel

    monkeypatch.setattr(
        "dialect_asr.model.Wav2Vec2Processor.from_pretrained",
        fake_from_pretrained,
    )

    processor = load_vietnamese_processor(local_files_only=True)

    assert processor is sentinel
    assert received == {
        "name": DEFAULT_PRETRAINED_MODEL,
        "kwargs": {"local_files_only": True},
    }
