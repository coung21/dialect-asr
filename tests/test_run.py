from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import pytest

from run import (
    _limit_split,
    _resolve_checkpoint,
    _selected_raw_splits,
    _validate_config,
)


CONFIG_DIR = str((Path(__file__).parents[1] / "configs").resolve())


class FakeSplit:
    def __init__(self, values):
        self.values = list(values)

    def __len__(self):
        return len(self.values)

    def select(self, indices):
        return FakeSplit([self.values[index] for index in indices])


def compose_config(*overrides: str):
    with initialize_config_dir(version_base="1.3", config_dir=CONFIG_DIR):
        return compose(config_name="config", overrides=list(overrides))


def test_hydra_config_composes_expected_defaults() -> None:
    cfg = compose_config()

    assert cfg.mode == "train"
    assert cfg.data.data_dir == "data/ViMD_Dataset/data"
    assert cfg.data.region_column == "region"
    assert cfg.model.pretrained_model_name == (
        "nguyenvulebinh/wav2vec2-base-vi-vlsp2020"
    )
    assert cfg.model.architecture == "baseline"
    assert cfg.trainer.num_train_epochs == 15
    assert cfg.trainer.global_train_batch_size == 8
    assert cfg.trainer.group_by_length is True
    assert cfg.trainer.full_determinism is False
    assert cfg.trainer.wandb_log_model == "end"
    assert OmegaConf.to_container(cfg.trainer.wandb_tags) == [
        "vimd",
        "wav2vec2",
        "baseline",
    ]


def test_dggfm_experiment_composes_model_and_run_settings() -> None:
    cfg = compose_config("experiment=dggfm")

    assert cfg.model.architecture == "dggfm"
    assert cfg.model.branch_block == 6
    assert OmegaConf.to_container(cfg.model.fusion_blocks) == [6, 8, 10, 12]
    assert cfg.model.dialect_dim == 64
    assert cfg.model.dialect_loss_weight == pytest.approx(0.1)
    assert cfg.trainer.output_dir == "outputs/dggfm"
    assert cfg.trainer.wandb_group == "dggfm"


def test_multitask_experiment_composes_model_and_run_settings() -> None:
    cfg = compose_config("experiment=multitask")

    assert cfg.model.architecture == "multitask"
    assert cfg.model.branch_block == 6
    assert cfg.model.num_regions == 3
    assert cfg.model.dialect_bottleneck_size == 256
    assert cfg.model.dialect_loss_weight == pytest.approx(1.0)
    assert cfg.trainer.output_dir == "outputs/multitask"
    assert cfg.trainer.wandb_group == "multitask"


def test_eval_mode_requires_checkpoint() -> None:
    cfg = compose_config("mode=eval", "split=test")

    with pytest.raises(ValueError, match="yêu cầu checkpoint"):
        _validate_config(cfg)


def test_eval_mode_accepts_validation_split_and_checkpoint() -> None:
    cfg = compose_config(
        "mode=eval",
        "split=validation",
        "checkpoint=owner/model",
    )

    _validate_config(cfg)


def test_limit_split_is_deterministic_and_bounded() -> None:
    split = FakeSplit(range(5))

    assert _limit_split(split, 3).values == [0, 1, 2]
    assert _limit_split(split, 99).values == [0, 1, 2, 3, 4]
    assert _limit_split(split, None) is split


def test_train_selects_only_required_splits() -> None:
    cfg = compose_config(
        "data.max_train_samples=2",
        "data.max_validation_samples=1",
        "evaluate_after_train=false",
    )
    raw = {
        "train": FakeSplit(range(4)),
        "validation": FakeSplit(range(3)),
        "test": FakeSplit(range(2)),
    }

    selected = _selected_raw_splits(cfg, raw)

    assert list(selected) == ["train", "validation"]
    assert selected["train"].values == [0, 1]
    assert selected["validation"].values == [0]


def test_resolve_checkpoint_requires_existing_local_resume(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint-10"
    checkpoint.mkdir()

    assert _resolve_checkpoint(str(checkpoint), must_exist=True) == str(
        checkpoint.resolve()
    )
    with pytest.raises(FileNotFoundError):
        _resolve_checkpoint(str(tmp_path / "missing"), must_exist=True)
