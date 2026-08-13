import torch
from transformers import Wav2Vec2Config

from dialect_asr.dann_model import DANNWav2Vec2CTC
from dialect_asr.modules import reverse_gradient
from dialect_asr.trainer import TrainerConfig, create_trainer


def tiny_dann_config() -> Wav2Vec2Config:
    config = Wav2Vec2Config(
        vocab_size=12,
        hidden_size=8,
        num_hidden_layers=3,
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
    config.dialect_dropout = 0.0
    config.dialect_loss_weight = 1.0
    config.grl_scale = 1.0
    return config


def test_gradient_reversal_preserves_values_and_reverses_gradient() -> None:
    inputs = torch.tensor([[1.0, -2.0]], requires_grad=True)  # [B=1, H=2].
    outputs = reverse_gradient(inputs, scale=0.5)
    # GRL forward: [B=1, H=2] -> unchanged [B=1, H=2].
    outputs.sum().backward()  # [B=1, H=2] -> scalar objective [].

    assert torch.equal(outputs, inputs)
    assert torch.allclose(inputs.grad, torch.full_like(inputs, -0.5))


def test_dann_head_receives_output_after_final_transformer_block() -> None:
    model = DANNWav2Vec2CTC(tiny_dann_config())
    model.eval()
    captured_final_block: list[torch.Tensor] = []

    def capture_final_block(_module, _inputs, output) -> None:
        captured_final_block.append(output[0].detach())
        # Final block output tuple contains hidden states [B, T_frame, H].

    hook = model.wav2vec2.encoder.layers[-1].register_forward_hook(
        capture_final_block
    )
    input_values = torch.randn(2, 1_600)  # [B=2, T_audio=1600].
    with torch.no_grad():
        output = model(input_values)
    hook.remove()

    expected_pooled = captured_final_block[0].mean(dim=1)
    # Final states [B=2, T_frame, H=8] -> temporal mean [B=2, H=8].
    assert output.dialect_logits.shape == (2, 3)  # [B=2, R=3].
    assert output.dialect_pooled.shape == (2, 8)  # [B=2, H=8].
    assert torch.allclose(output.dialect_pooled, expected_pooled, atol=1.0e-6)
    assert model.config.dann_branch_block == 3


def test_dann_returns_ctc_plus_weighted_dialect_loss() -> None:
    model = DANNWav2Vec2CTC(tiny_dann_config())
    input_values = torch.randn(2, 1_600)  # [B=2, T_audio=1600].
    labels = torch.tensor([[1, 2, 3], [4, 5, -100]])
    # CTC targets [B=2, T_text=3].
    region_labels = torch.tensor([0, 2])  # Region targets [B=2].

    output = model(
        input_values,  # [B=2, T_audio=1600].
        labels=labels,  # [B=2, T_text=3].
        region_labels=region_labels,  # [B=2].
    )

    assert output.logits.ndim == 3  # [B=2, T_frame, V=12].
    assert output.ctc_loss.ndim == 0  # Scalar [].
    assert output.dialect_loss.ndim == 0  # Scalar [].
    assert output.loss.ndim == 0  # Scalar [].
    assert torch.allclose(
        output.loss,
        output.ctc_loss + output.dialect_loss,
    )


def test_dann_checkpoint_round_trip_preserves_head_and_grl(tmp_path) -> None:
    model = DANNWav2Vec2CTC(tiny_dann_config())
    with torch.no_grad():
        model.dialect_head.classifier[-1].bias.fill_(1.25)
        # Final classifier bias is a dialect vector [R=3].
    checkpoint_dir = tmp_path / "dann"
    model.save_pretrained(checkpoint_dir)

    reloaded = DANNWav2Vec2CTC.from_pretrained(checkpoint_dir)

    assert reloaded.config.architecture == "dann"
    assert reloaded.config.dann_branch_block == 3
    assert reloaded.grl_scale == 1.0
    assert torch.all(reloaded.dialect_head.classifier[-1].bias == 1.25)


def test_dann_runs_one_trainer_step_and_evaluation(tmp_path) -> None:
    model = DANNWav2Vec2CTC(tiny_dann_config())
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
            "input_values": torch.stack(
                [item["input_values"] for item in examples]
            ),
            # B audio tensors [T_audio] -> [B, T_audio].
            "labels": torch.stack([item["labels"] for item in examples]),
            # B text tensors [T_text] -> [B, T_text].
            "region_labels": torch.stack(
                [item["region_labels"] for item in examples]
            ),
            # B scalar region IDs [] -> [B].
        }

    trainer = create_trainer(
        model=model,
        processor=None,
        data_collator=collate,
        train_dataset=dataset,
        eval_dataset=dataset,
        compute_metrics=lambda prediction: {
            "ctc_rank": float(prediction.predictions[0].ndim),
            "dialect_rank": float(prediction.predictions[1].ndim),
            "ctc_loss": float(prediction.predictions[2].mean()),
            "dialect_loss": float(prediction.predictions[3].mean()),
        },
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
    assert eval_metrics["eval_ctc_rank"] == 2.0
    assert eval_metrics["eval_dialect_rank"] == 2.0
    assert torch.isfinite(torch.tensor(eval_metrics["eval_ctc_loss"]))
    assert torch.isfinite(torch.tensor(eval_metrics["eval_dialect_loss"]))
    train_logs = [entry for entry in trainer.state.log_history if "loss" in entry]
    assert "ctc_loss" in train_logs[0]
    assert "dialect_loss" in train_logs[0]
