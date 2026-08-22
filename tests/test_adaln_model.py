from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import WhisperConfig

from dialect_asr.adaln_model import (
    AdaLNWhisperEncoder,
    AdaLNWhisperEncoderLayer,
    PhoWhisperAdaLNASR,
    inject_adaln,
    resolve_did_checkpoint_path,
)
from dialect_asr.modules import AdaLN, ECAPA_TDNN_DID


def tiny_config() -> WhisperConfig:
    return WhisperConfig(
        vocab_size=30,
        num_mel_bins=10,
        encoder_layers=2,
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


TINY_DID_KWARGS = {
    "num_regions": 3,
    "channels": 8,
    "embedding_size": 6,
    "res2net_scale": 2,
    "se_bottleneck_channels": 4,
    "attention_channels": 4,
    "dropout": 0.0,
}


def test_adaln_matches_plain_layernorm_at_init_regardless_of_cond() -> None:
    """Zero-initialized `to_gamma_beta` -> AdaLN(x, cond) == LayerNorm(x) for any cond."""
    adaln = AdaLN(normalized_shape=8, cond_dim=4)
    x = torch.randn(3, 5, 8)  # [B=3, T=5, H=8].
    reference = torch.nn.functional.layer_norm(x, (8,), adaln.weight, adaln.bias)

    for cond in (torch.zeros(3, 4), torch.randn(3, 4) * 10):
        out = adaln(x, cond)
        assert torch.allclose(out, reference, atol=1e-6)


def test_inject_adaln_preserves_pretrained_norm_weights() -> None:
    config = tiny_config()
    from transformers import WhisperForConditionalGeneration

    plain = WhisperForConditionalGeneration(config)
    original_weight = plain.model.encoder.layers[0].self_attn_layer_norm.weight.clone()
    original_bias = plain.model.encoder.layers[0].self_attn_layer_norm.bias.clone()

    inject_adaln(plain.model.encoder, cond_dim=config.d_model)

    layer = plain.model.encoder.layers[0]
    assert isinstance(layer, AdaLNWhisperEncoderLayer)
    assert isinstance(plain.model.encoder, AdaLNWhisperEncoder)
    assert torch.allclose(layer.self_attn_layer_norm.weight, original_weight)
    assert torch.allclose(layer.self_attn_layer_norm.bias, original_bias)


def test_inject_adaln_leaves_encoder_output_unchanged_at_init() -> None:
    config = tiny_config()
    from transformers import WhisperForConditionalGeneration

    plain = WhisperForConditionalGeneration(config)
    plain.eval()
    input_features = torch.randn(2, config.num_mel_bins, config.max_source_positions * 2)

    with torch.no_grad():
        before = plain.model.encoder(input_features).last_hidden_state

    inject_adaln(plain.model.encoder, cond_dim=config.d_model)
    cond = torch.randn(2, config.d_model)  # Arbitrary; AdaLN heads are zero-init.
    with torch.no_grad():
        after = plain.model.encoder(input_features, cond=cond).last_hidden_state

    assert torch.allclose(before, after, atol=1e-5)


def test_adaln_encoder_requires_cond() -> None:
    config = tiny_config()
    from transformers import WhisperForConditionalGeneration

    plain = WhisperForConditionalGeneration(config)
    inject_adaln(plain.model.encoder, cond_dim=config.d_model)
    input_features = torch.randn(1, config.num_mel_bins, config.max_source_positions * 2)

    try:
        plain.model.encoder(input_features)
    except ValueError as error:
        assert "cond" in str(error)
    else:
        raise AssertionError("Expected ValueError when cond is missing")


def test_phowhisper_adaln_forward_returns_loss_and_logits() -> None:
    config = tiny_config()
    model = PhoWhisperAdaLNASR(config, did_kwargs=TINY_DID_KWARGS)
    model.eval()
    input_features = torch.randn(2, config.num_mel_bins, config.max_source_positions * 2)
    labels = torch.tensor([[3, 4, 5, 2], [6, 7, -100, -100]])

    with torch.no_grad():
        output = model(input_features=input_features, labels=labels)

    assert output.logits.shape == (2, 4, config.vocab_size)
    assert torch.isfinite(output.loss)


def test_phowhisper_adaln_did_branch_is_frozen_but_conditioner_trains() -> None:
    config = tiny_config()
    model = PhoWhisperAdaLNASR(config, did_kwargs=TINY_DID_KWARGS)

    assert all(not parameter.requires_grad for parameter in model.did_model.parameters())
    assert all(parameter.requires_grad for parameter in model.conditioner.parameters())
    assert isinstance(model.did_model, ECAPA_TDNN_DID)


def test_phowhisper_adaln_gradients_reach_adaln_heads_and_conditioner() -> None:
    config = tiny_config()
    model = PhoWhisperAdaLNASR(config, did_kwargs=TINY_DID_KWARGS)
    model.train()
    input_features = torch.randn(2, config.num_mel_bins, config.max_source_positions * 2)
    labels = torch.tensor([[3, 4, 5, 2], [6, 7, -100, -100]])

    output = model(input_features=input_features, labels=labels)
    output.loss.backward()

    first_adaln = model.model.encoder.layers[0].self_attn_layer_norm
    assert first_adaln.to_gamma_beta.weight.grad is not None
    assert any(
        parameter.grad is not None for parameter in model.conditioner.parameters()
    )
    assert all(parameter.grad is None for parameter in model.did_model.parameters())


def test_phowhisper_adaln_train_keeps_did_model_in_eval_mode() -> None:
    config = tiny_config()
    model = PhoWhisperAdaLNASR(config, did_kwargs=TINY_DID_KWARGS)

    model.train()

    assert model.training
    assert not model.did_model.training


def test_resolve_did_checkpoint_path_passes_through_local_paths() -> None:
    assert resolve_did_checkpoint_path("outputs/did-ecapa-tdnn/final_model.pt") == Path(
        "outputs/did-ecapa-tdnn/final_model.pt"
    )


def test_resolve_did_checkpoint_path_downloads_wandb_artifact(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "final_model.pt").touch()
    received: dict[str, object] = {}

    class FakeArtifact:
        def download(self) -> str:
            return str(tmp_path)

    class FakeApi:
        def artifact(self, reference: str) -> FakeArtifact:
            received["reference"] = reference
            return FakeArtifact()

    monkeypatch.setattr("wandb.Api", FakeApi)

    resolved = resolve_did_checkpoint_path(
        "wandb-artifact:my-team/dialect-asr/did-ecapa-tdnn-did-model:latest"
    )

    assert resolved == tmp_path / "final_model.pt"
    assert received["reference"] == "my-team/dialect-asr/did-ecapa-tdnn-did-model:latest"


def test_resolve_did_checkpoint_path_rejects_artifact_without_pt_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "wandb.Api", lambda: SimpleNamespace(artifact=lambda ref: SimpleNamespace(download=lambda: str(tmp_path)))
    )

    with pytest.raises(FileNotFoundError, match="không chứa file .pt"):
        resolve_did_checkpoint_path("wandb-artifact:my-team/dialect-asr/did-ecapa-tdnn-did-model:latest")


def test_resolve_did_checkpoint_path_rejects_artifact_with_multiple_pt_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "a.pt").touch()
    (tmp_path / "b.pt").touch()
    monkeypatch.setattr(
        "wandb.Api", lambda: SimpleNamespace(artifact=lambda ref: SimpleNamespace(download=lambda: str(tmp_path)))
    )

    with pytest.raises(ValueError, match="chứa nhiều file .pt"):
        resolve_did_checkpoint_path("wandb-artifact:my-team/dialect-asr/did-ecapa-tdnn-did-model:latest")
