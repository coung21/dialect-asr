import os

import pytest
import torch
from transformers import WhisperConfig

from dialect_asr.model import BaselinePhoWhisperASR
from dialect_asr.trainer import (
    TrainerConfig,
    build_training_arguments,
    configure_wandb_environment,
    create_trainer,
)


def test_build_training_arguments_with_validation(tmp_path) -> None:
    config = TrainerConfig(
        output_dir=str(tmp_path / "output"),
        num_train_epochs=2,
        global_train_batch_size=24,
        per_device_train_batch_size=3,
        dataloader_num_workers=0,
        use_cpu=True,
    )

    arguments = build_training_arguments(
        config,
        has_eval_dataset=True,
        world_size=2,
    )

    assert arguments.eval_strategy.value == "epoch"
    assert arguments.save_strategy.value == "epoch"
    assert arguments.load_best_model_at_end
    assert arguments.metric_for_best_model == "eval_loss"
    assert arguments.train_batch_size == 3
    assert arguments.gradient_accumulation_steps == 4
    assert arguments.predict_with_generate
    assert arguments.generation_max_length == 225
    assert arguments.generation_num_beams == 1
    assert (tmp_path / "output").is_dir()


def test_build_training_arguments_without_validation(tmp_path) -> None:
    config = TrainerConfig(
        output_dir=str(tmp_path / "output"),
        dataloader_num_workers=0,
        use_cpu=True,
    )

    arguments = build_training_arguments(config, has_eval_dataset=False)

    assert arguments.eval_strategy.value == "no"
    assert not arguments.load_best_model_at_end
    assert arguments.metric_for_best_model is None
    assert not arguments.predict_with_generate


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"learning_rate": 0}, "learning_rate"),
        ({"global_train_batch_size": 0}, "global_train_batch_size"),
        ({"generation_max_length": 0}, "generation_max_length"),
        ({"generation_num_beams": 0}, "generation_num_beams"),
        ({"fp16": True, "bf16": True}, "đồng thời"),
        ({"warmup_ratio": -0.1}, "warmup_ratio"),
        ({"warmup_ratio": 1.0}, "warmup_ratio"),
        ({"wandb_mode": "invalid"}, "wandb_mode"),
        ({"wandb_log_model": "yes"}, "wandb_log_model"),
        ({"wandb_log_model": "checkpoint"}, "wandb_log_model"),
        ({"wandb_log_model": True}, "wandb_log_model"),
    ],
)
def test_trainer_config_rejects_invalid_values(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        TrainerConfig(**kwargs)


def test_default_hyperparameters_are_shared_experiment_values(tmp_path) -> None:
    config = TrainerConfig(
        output_dir=str(tmp_path / "output"),
        dataloader_num_workers=0,
        use_cpu=True,
    )
    arguments = build_training_arguments(config, world_size=1)

    assert arguments.num_train_epochs == 15
    assert arguments.learning_rate == pytest.approx(1e-4)
    assert arguments.weight_decay == pytest.approx(0.005)
    assert arguments.optim.value == "adamw_torch"
    assert arguments.warmup_steps == pytest.approx(0.1)
    assert arguments.per_device_train_batch_size == 8
    assert arguments.gradient_accumulation_steps == 1
    assert arguments.train_sampling_strategy == "group_by_length"
    assert arguments.length_column_name == "length"
    assert arguments.seed == 42
    assert arguments.data_seed == 42
    assert not arguments.full_determinism
    assert arguments.report_to == ["wandb"]
    assert os.environ["WANDB_LOG_MODEL"] == "end"


def test_default_batch_size_is_eight_without_accumulation_on_one_gpu(tmp_path) -> None:
    config = TrainerConfig(
        output_dir=str(tmp_path / "output"),
        dataloader_num_workers=0,
        use_cpu=True,
    )
    arguments = build_training_arguments(config, world_size=1)

    assert arguments.per_device_train_batch_size == 8
    assert arguments.gradient_accumulation_steps == 1
    assert (
        arguments.per_device_train_batch_size
        * 1
        * arguments.gradient_accumulation_steps
        == 8
    )


def test_configure_wandb_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        "WANDB_PROJECT",
        "WANDB_MODE",
        "WANDB_LOG_MODEL",
        "WANDB_WATCH",
        "WANDB_TAGS",
        "WANDB_ENTITY",
        "WANDB_RUN_GROUP",
    ):
        monkeypatch.delenv(key, raising=False)

    configure_wandb_environment(
        TrainerConfig(
            wandb_project="dialect-asr-test",
            wandb_entity="research-team",
            wandb_group="baseline-test",
            wandb_tags=["vimd", "test"],
            wandb_mode="offline",
            wandb_log_model="false",
            wandb_watch="false",
        )
    )

    assert os.environ["WANDB_PROJECT"] == "dialect-asr-test"
    assert os.environ["WANDB_ENTITY"] == "research-team"
    assert os.environ["WANDB_RUN_GROUP"] == "baseline-test"
    assert os.environ["WANDB_TAGS"] == "vimd,test"
    assert os.environ["WANDB_MODE"] == "offline"
    assert os.environ["WANDB_LOG_MODEL"] == "false"


def test_hydra_boolean_false_disables_wandb_model_upload() -> None:
    config = TrainerConfig(wandb_log_model=False)

    assert config.wandb_log_model == "false"


def test_create_trainer_wires_all_components(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    received = {}
    sentinel = object()

    def fake_trainer(**kwargs):
        received.update(kwargs)
        return sentinel

    monkeypatch.setattr("dialect_asr.trainer.Seq2SeqTrainer", fake_trainer)
    model = object()
    processor = object()
    collator = object()
    train_dataset = [1]
    eval_dataset = [2]
    metrics = object()

    trainer = create_trainer(
        model=model,
        processor=processor,
        data_collator=collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=metrics,
        config=TrainerConfig(
            output_dir=str(tmp_path / "output"),
            dataloader_num_workers=0,
            report_to="none",
            use_cpu=True,
        ),
    )

    assert trainer is sentinel
    assert received["model"] is model
    assert received["processing_class"] is processor
    assert received["data_collator"] is collator
    assert received["train_dataset"] is train_dataset
    assert received["eval_dataset"] is eval_dataset
    assert received["compute_metrics"] is metrics


def _tiny_whisper_config() -> WhisperConfig:
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


def test_trainer_runs_one_cpu_optimization_step(tmp_path) -> None:
    config = _tiny_whisper_config()
    model = BaselinePhoWhisperASR(config)
    # Each record: input_features [num_mel_bins=10, T_frame=20], labels [T_text=3].
    dataset = [
        {
            "input_features": torch.randn(config.num_mel_bins, config.max_source_positions * 2),
            "labels": torch.tensor([3, 4, 5]),
        },
        {
            "input_features": torch.randn(config.num_mel_bins, config.max_source_positions * 2),
            "labels": torch.tensor([5, 4, 3]),
        },
    ]

    def collate(examples):
        return {
            # Two [num_mel_bins, T_frame] tensors -> [B=2, num_mel_bins, T_frame].
            "input_features": torch.stack([item["input_features"] for item in examples]),
            # Two [T_text] tensors -> [B=2, T_text=3].
            "labels": torch.stack([item["labels"] for item in examples]),
        }

    trainer = create_trainer(
        model=model,
        processor=None,
        data_collator=collate,
        train_dataset=dataset,
        config=TrainerConfig(
            output_dir=str(tmp_path / "smoke-train"),
            num_train_epochs=1,
            global_train_batch_size=2,
            per_device_train_batch_size=2,
            warmup_ratio=0,
            logging_steps=1,
            dataloader_num_workers=0,
            group_by_length=False,
            report_to="none",
            use_cpu=True,
        ),
    )

    result = trainer.train()

    assert result.global_step == 1
    assert result.training_loss > 0
