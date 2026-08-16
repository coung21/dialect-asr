import inspect
from types import SimpleNamespace

import pytest
import torch
from transformers import WhisperConfig

from dialect_asr.base_model import AbstractPhoWhisperASR
from dialect_asr.model import (
    DEFAULT_PRETRAINED_MODEL,
    BaselinePhoWhisperASR,
    load_vietnamese_processor,
)


def test_base_model_is_abstract_and_baseline_declares_architecture() -> None:
    assert inspect.isabstract(AbstractPhoWhisperASR)
    assert BaselinePhoWhisperASR.architecture_name() == "baseline"


def tiny_config() -> WhisperConfig:
    return WhisperConfig(
        vocab_size=30,
        num_mel_bins=10,
        encoder_layers=1,
        encoder_attention_heads=2,
        decoder_layers=1,
        decoder_attention_heads=2,
        d_model=16,
        encoder_ffn_dim=32,
        decoder_ffn_dim=32,
        max_source_positions=10,
        max_target_positions=20,
        pad_token_id=0,
        decoder_start_token_id=1,
        eos_token_id=2,
        bos_token_id=1,
    )


def test_forward_returns_seq2seq_loss_and_logits() -> None:
    config = tiny_config()
    model = BaselinePhoWhisperASR(config)
    model.eval()
    # input_features: [B=2, num_mel_bins=10, T_frame=20]; T_frame must equal
    # 2 * max_source_positions because the encoder downsamples time by 2.
    input_features = torch.randn(2, config.num_mel_bins, config.max_source_positions * 2)
    labels = torch.tensor([[3, 4, 5, 2], [6, 7, -100, -100]])

    with torch.no_grad():
        output = model(input_features=input_features, labels=labels)

    assert output.logits.shape[0] == 2
    assert output.logits.shape[-1] == config.vocab_size
    assert output.loss.ndim == 0
    assert torch.isfinite(output.loss)


def test_pretrained_factory_uses_default_checkpoint_and_freezes_encoder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_model = SimpleNamespace(
        encoder_frozen=False,
        checkpointing=False,
    )
    fake_model.freeze_encoder = lambda: setattr(fake_model, "encoder_frozen", True)
    fake_model.gradient_checkpointing_enable = lambda: setattr(
        fake_model, "checkpointing", True
    )
    received: dict[str, object] = {}

    def fake_from_pretrained(name: str, **kwargs):
        received.update(name=name, kwargs=kwargs)
        return fake_model

    monkeypatch.setattr(BaselinePhoWhisperASR, "from_pretrained", fake_from_pretrained)

    model = BaselinePhoWhisperASR.from_vietnamese_pretrained(
        gradient_checkpointing=True,
        full_determinism=False,
        local_files_only=True,
    )

    assert model is fake_model
    assert received == {
        "name": DEFAULT_PRETRAINED_MODEL,
        "kwargs": {"local_files_only": True},
    }
    assert fake_model.encoder_frozen
    assert fake_model.checkpointing


def test_pretrained_factory_can_leave_encoder_trainable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_model = SimpleNamespace(encoder_frozen=False)
    fake_model.freeze_encoder = lambda: setattr(fake_model, "encoder_frozen", True)
    monkeypatch.setattr(
        BaselinePhoWhisperASR, "from_pretrained", lambda name, **kwargs: fake_model
    )

    model = BaselinePhoWhisperASR.from_vietnamese_pretrained(freeze_encoder=False)

    assert model is fake_model
    assert not fake_model.encoder_frozen


def test_parameter_counts() -> None:
    model = BaselinePhoWhisperASR(tiny_config())
    before = model.parameter_counts()
    assert before["total"] == before["trainable"] + before["frozen"]

    # proj_out is weight-tied to the decoder embedding, so freezing it removes
    # exactly one parameter tensor's worth of trainable weights.
    model.proj_out.weight.requires_grad = False
    after = model.parameter_counts()

    assert after["total"] == before["total"]
    assert after["frozen"] - before["frozen"] == model.proj_out.weight.numel()
    assert after["trainable"] == before["trainable"] - model.proj_out.weight.numel()
    assert after["total"] == after["trainable"] + after["frozen"]


def test_load_processor_uses_default_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = object()
    received: dict[str, object] = {}

    def fake_from_pretrained(name: str, **kwargs):
        received.update(name=name, kwargs=kwargs)
        return sentinel

    monkeypatch.setattr(
        "dialect_asr.model.WhisperProcessor.from_pretrained",
        fake_from_pretrained,
    )

    processor = load_vietnamese_processor(local_files_only=True)

    assert processor is sentinel
    assert received == {
        "name": DEFAULT_PRETRAINED_MODEL,
        "kwargs": {"local_files_only": True},
    }
