import os
from types import SimpleNamespace

import pytest
import torch
from transformers import Wav2Vec2Config

from dialect_asr.model import BaselineWav2Vec2CTC
from dialect_asr.trainer import (
    TrainerConfig,
    build_training_arguments,
    configure_wandb_environment,
    create_trainer,
    preprocess_logits_for_ctc,
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


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"learning_rate": 0}, "learning_rate"),
        ({"global_train_batch_size": 0}, "global_train_batch_size"),
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


def test_preprocess_logits_for_ctc_reduces_vocab_dimension() -> None:
    # logits: [B=2, T_frame=3, V=4].
    logits = torch.tensor(
        [
            [[0.0, 4.0, 1.0, 2.0], [3.0, 1.0, 0.0, 2.0], [0.0, 1.0, 5.0, 2.0]],
            [[6.0, 4.0, 1.0, 2.0], [3.0, 7.0, 0.0, 2.0], [0.0, 1.0, 2.0, 8.0]],
        ]
    )

    token_ids = preprocess_logits_for_ctc(logits)

    assert token_ids.shape == (2, 3)  # [B=2, T_frame=3].
    assert token_ids.tolist() == [[1, 0, 2], [0, 1, 3]]


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
    assert arguments.per_device_train_batch_size == 4
    assert arguments.gradient_accumulation_steps == 2
    assert arguments.train_sampling_strategy == "group_by_length"
    assert arguments.length_column_name == "length"
    assert arguments.seed == 42
    assert arguments.data_seed == 42
    assert not arguments.full_determinism
    assert arguments.report_to == ["wandb"]
    assert os.environ["WANDB_LOG_MODEL"] == "end"


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

    monkeypatch.setattr("dialect_asr.trainer.Trainer", fake_trainer)
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
    assert received["preprocess_logits_for_metrics"] is preprocess_logits_for_ctc


def test_trainer_runs_one_cpu_optimization_step(tmp_path) -> None:
    model = BaselineWav2Vec2CTC(
        Wav2Vec2Config(
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
    )
    # Each record: input_values [T_audio=1600], labels [T_text=3].
    dataset = [
        {"input_values": torch.randn(1_600), "labels": torch.tensor([1, 2, 3])},
        {"input_values": torch.randn(1_600), "labels": torch.tensor([3, 2, 1])},
    ]

    def collate(examples):
        return {
            # Two [T_audio] tensors -> [B=2, T_audio=1600].
            "input_values": torch.stack([item["input_values"] for item in examples]),
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
