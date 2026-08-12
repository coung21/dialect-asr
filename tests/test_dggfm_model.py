import torch
from transformers import Wav2Vec2Config

from dialect_asr.dggfm_model import DGGFMWav2Vec2CTC
from dialect_asr.trainer import TrainerConfig, create_trainer


def tiny_dggfm_config() -> Wav2Vec2Config:
    config = Wav2Vec2Config(
        vocab_size=12,
        hidden_size=8,
        num_hidden_layers=12,
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
        activation_dropout=0.0,
        feat_proj_dropout=0.0,
        final_dropout=0.0,
        layerdrop=0.0,
        mask_time_prob=0.0,
    )
    config.dialect_bottleneck_size = 4
    config.dialect_dim = 3
    config.dggfm_gate_hidden_dim = 5
    config.dialect_dropout = 0.0
    return config


def test_dggfm_model_branches_after_block_six_and_fuses_selected_blocks() -> None:
    model = DGGFMWav2Vec2CTC(tiny_dggfm_config())
    model.eval()
    captured_block_six: list[torch.Tensor] = []

    def capture_block_six(_module, _inputs, output) -> None:
        captured_block_six.append(output[0].detach())
        # Transformer block output tuple contains hidden states [B, T_frame, H].

    hook = model.wav2vec2.encoder.layers[5].register_forward_hook(capture_block_six)
    input_values = torch.randn(2, 1_600)  # [B=2, T_audio=1600].
    attention_mask = torch.ones_like(input_values, dtype=torch.long)
    # Audio mask preserves input shape [B=2, T_audio=1600].

    with torch.no_grad():
        output = model(input_values, attention_mask=attention_mask)
    hook.remove()

    assert model.branch_block == 6
    assert model.fusion_blocks == (6, 8, 10, 12)
    assert set(output.gate_statistics) == {6, 8, 10, 12}
    assert output.dialect_logits.shape == (2, 3)  # [B=2, R=3].
    assert output.dialect_embedding.shape == (2, 3)  # [B=2, E=3].
    assert output.dialect_posterior.shape == (2, 3)  # [B=2, R=3].
    expected_pooled = captured_block_six[0].mean(dim=1)
    # Block-6 output [B=2, T_frame, H=8] -> mean pooled [B=2, H=8].
    assert torch.allclose(output.dialect_pooled, expected_pooled, atol=1.0e-6)


def test_dggfm_model_returns_joint_ctc_and_dialect_loss() -> None:
    model = DGGFMWav2Vec2CTC(tiny_dggfm_config())
    model.eval()
    input_values = torch.randn(2, 1_600)  # [B=2, T_audio=1600].
    labels = torch.tensor([[1, 2, 3], [4, 5, -100]])
    # CTC targets: [B=2, T_text=3].
    region_labels = torch.tensor([0, 2])  # [B=2].

    output = model(
        input_values,       # [B=2, T_audio=1600].
        labels=labels,      # [B=2, T_text=3].
        region_labels=region_labels,  # [B=2].
    )

    assert output.logits.ndim == 3  # [B=2, T_frame, V=12].
    assert output.logits.shape[0] == 2
    assert output.logits.shape[-1] == 12
    assert output.ctc_loss.ndim == 0  # Scalar [].
    assert output.dialect_loss.ndim == 0  # Scalar [].
    assert output.loss.ndim == 0  # Scalar [].
    assert torch.allclose(
        output.loss,
        output.ctc_loss + model.dialect_loss_weight * output.dialect_loss,
    )


def test_dggfm_model_starts_with_identity_modulation() -> None:
    model = DGGFMWav2Vec2CTC(tiny_dggfm_config())

    for block in model.fusion_blocks:
        module = model.dggfm_layers[str(block)]
        assert torch.count_nonzero(module.to_gamma.weight) == 0
        assert torch.count_nonzero(module.to_beta.weight) == 0
        assert torch.all(module.gate_network[-1].bias == -2.0)


def test_dggfm_custom_config_is_saved_on_wav2vec2_config() -> None:
    model = DGGFMWav2Vec2CTC(tiny_dggfm_config())

    assert model.architecture_name() == "dggfm"
    assert model.config.architecture == "dggfm"
    assert model.config.dggfm_branch_block == 6
    assert model.config.dggfm_fusion_blocks == [6, 8, 10, 12]
    assert model.config.dialect_dim == 3


def test_loading_baseline_checkpoint_restores_identity_initialization(tmp_path) -> None:
    from dialect_asr.model import BaselineWav2Vec2CTC

    baseline_dir = tmp_path / "baseline"
    BaselineWav2Vec2CTC(tiny_dggfm_config()).save_pretrained(baseline_dir)

    model = DGGFMWav2Vec2CTC.from_pretrained(baseline_dir)

    for block in model.fusion_blocks:
        module = model.dggfm_layers[str(block)]
        assert torch.count_nonzero(module.to_gamma.weight) == 0
        assert torch.count_nonzero(module.to_gamma.bias) == 0
        assert torch.count_nonzero(module.to_beta.weight) == 0
        assert torch.count_nonzero(module.to_beta.bias) == 0
        assert torch.count_nonzero(module.gate_network[-1].weight) == 0
        assert torch.all(module.gate_network[-1].bias == -2.0)


def test_dggfm_checkpoint_round_trip_keeps_learned_modulation(tmp_path) -> None:
    model = DGGFMWav2Vec2CTC(tiny_dggfm_config())
    with torch.no_grad():
        model.dggfm_layers["6"].to_gamma.weight.fill_(0.25)
        # Learned gamma projection remains [H=8, E=3].
        model.dggfm_layers["6"].gate_network[-1].bias.fill_(1.5)
        # Learned final gate bias remains [H=8].
    checkpoint_dir = tmp_path / "dggfm"
    model.save_pretrained(checkpoint_dir)

    reloaded = DGGFMWav2Vec2CTC.from_pretrained(checkpoint_dir)

    assert torch.all(reloaded.dggfm_layers["6"].to_gamma.weight == 0.25)
    assert torch.all(reloaded.dggfm_layers["6"].gate_network[-1].bias == 1.5)


def test_dggfm_runs_one_trainer_step_and_evaluation(tmp_path) -> None:
    model = DGGFMWav2Vec2CTC(tiny_dggfm_config())
    dataset = [
        {
            "input_values": torch.randn(1_600),  # [T_audio=1600].
            "labels": torch.tensor([1, 2, 3]),  # [T_text=3].
            "region_labels": torch.tensor(0),  # Scalar [].
        },
        {
            "input_values": torch.randn(1_600),  # [T_audio=1600].
            "labels": torch.tensor([3, 2, 1]),  # [T_text=3].
            "region_labels": torch.tensor(2),  # Scalar [].
        },
    ]

    def collate(examples):
        return {
            "input_values": torch.stack([item["input_values"] for item in examples]),
            # B tensors [T_audio=1600] -> [B, T_audio=1600].
            "labels": torch.stack([item["labels"] for item in examples]),
            # B tensors [T_text=3] -> [B, T_text=3].
            "region_labels": torch.stack(
                [item["region_labels"] for item in examples]
            ),
            # B scalar region IDs [] -> [B].
        }

    def compute_metrics(prediction):
        # Trainer preprocessing produces greedy IDs [N, T_frame].
        return {"prediction_rank": float(prediction.predictions.ndim)}

    trainer = create_trainer(
        model=model,
        processor=None,
        data_collator=collate,
        train_dataset=dataset,
        eval_dataset=dataset,
        compute_metrics=compute_metrics,
        config=TrainerConfig(
            output_dir=str(tmp_path / "trainer"),
            num_train_epochs=1,
            global_train_batch_size=2,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            warmup_ratio=0.0,
            dataloader_num_workers=0,
            group_by_length=False,
            report_to="none",
            use_cpu=True,
        ),
    )

    train_result = trainer.train()
    eval_metrics = trainer.evaluate()

    assert torch.isfinite(torch.tensor(train_result.training_loss))
    assert torch.isfinite(torch.tensor(eval_metrics["eval_loss"]))
    assert eval_metrics["eval_prediction_rank"] == 2.0
