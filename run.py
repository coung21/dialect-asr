"""Hydra entry point for training and evaluating the ViMD ASR baseline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from datasets import DatasetDict
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from dialect_asr import (
    AbstractPhoWhisperASR,
    DataCollatorSpeechSeq2SeqWithPadding,
    TrainerConfig,
    architecture_from_checkpoint,
    build_compute_metrics,
    build_project_model,
    create_trainer,
    load_vietnamese_processor,
    load_vimd,
    prepare_dataset,
    seed_everything,
)


LOGGER = logging.getLogger(__name__)
VALID_MODES = {"train", "eval"}
VALID_EVAL_SPLITS = {"validation", "test"}


def _absolute_local_path(value: str) -> str:
    return str(Path(to_absolute_path(value)).resolve())


def _resolve_checkpoint(value: str | None, *, must_exist: bool) -> str | None:
    if not value:
        return None

    local_path = Path(to_absolute_path(value)).resolve()
    if local_path.exists():
        return str(local_path)
    if must_exist:
        raise FileNotFoundError(f"Không tìm thấy checkpoint local: {local_path}")
    # Eval may also load a remote Hugging Face repository ID.
    return value


def _limit_split(dataset: Any, max_samples: int | None) -> Any:
    if max_samples is None:
        return dataset
    if max_samples <= 0:
        raise ValueError("max_*_samples phải > 0 hoặc null")
    return dataset.select(range(min(max_samples, len(dataset))))


def _selected_raw_splits(cfg: DictConfig, dataset: DatasetDict) -> DatasetDict:
    if cfg.mode == "train":
        splits: dict[str, Any] = {
            "train": _limit_split(dataset["train"], cfg.data.max_train_samples),
            "validation": _limit_split(
                dataset["validation"], cfg.data.max_validation_samples
            ),
        }
        if cfg.evaluate_after_train:
            splits["test"] = _limit_split(
                dataset["test"], cfg.data.max_test_samples
            )
        return DatasetDict(splits)

    split = str(cfg.split)
    limit = cfg.data[f"max_{split}_samples"]
    return DatasetDict({split: _limit_split(dataset[split], limit)})


def _trainer_config(cfg: DictConfig) -> TrainerConfig:
    values = OmegaConf.to_container(cfg.trainer, resolve=True)
    if not isinstance(values, dict):
        raise TypeError("trainer config phải là một mapping")
    values["output_dir"] = _absolute_local_path(str(values["output_dir"]))
    return TrainerConfig(**values)


def _validate_config(cfg: DictConfig) -> None:
    if cfg.mode not in VALID_MODES:
        raise ValueError(f"mode phải là một trong {sorted(VALID_MODES)}")
    if cfg.mode == "eval" and cfg.split not in VALID_EVAL_SPLITS:
        raise ValueError(f"split phải là một trong {sorted(VALID_EVAL_SPLITS)}")
    if cfg.mode == "eval" and not cfg.checkpoint:
        raise ValueError("mode=eval yêu cầu checkpoint=<path hoặc Hugging Face repo>")


def _load_processor(cfg: DictConfig, checkpoint: str | None) -> Any:
    source = checkpoint if cfg.mode == "eval" and checkpoint else cfg.model.pretrained_model_name
    try:
        return load_vietnamese_processor(
            str(source),
            local_files_only=bool(cfg.model.local_files_only),
        )
    except OSError:
        if source == cfg.model.pretrained_model_name:
            raise
        LOGGER.warning(
            "Checkpoint không chứa processor; dùng processor từ %s",
            cfg.model.pretrained_model_name,
        )
        return load_vietnamese_processor(
            str(cfg.model.pretrained_model_name),
            local_files_only=bool(cfg.model.local_files_only),
        )


def _load_model(
    cfg: DictConfig,
    trainer_config: TrainerConfig,
    checkpoint: str | None,
) -> AbstractPhoWhisperASR:
    evaluation = cfg.mode == "eval"
    architecture = str(cfg.model.architecture)
    source = str(cfg.model.pretrained_model_name)
    if evaluation:
        if checkpoint is None:
            raise ValueError("Eval yêu cầu checkpoint")
        source = checkpoint
        architecture = architecture_from_checkpoint(
            source,
            fallback=architecture,
            local_files_only=bool(cfg.model.local_files_only),
        )

    model_options = OmegaConf.to_container(cfg.model, resolve=True)
    if not isinstance(model_options, dict):
        raise TypeError("model config phải là một mapping")
    return build_project_model(
        architecture=architecture,
        source=source,
        evaluation=evaluation,
        model_options=model_options,
        freeze_encoder=bool(cfg.model.freeze_encoder),
        gradient_checkpointing=bool(cfg.model.gradient_checkpointing),
        seed=trainer_config.seed,
        full_determinism=trainer_config.full_determinism,
        local_files_only=bool(cfg.model.local_files_only),
    )


def _save_metrics(trainer: Any, prefix: str, metrics: dict[str, float]) -> None:
    trainer.log_metrics(prefix, metrics)
    trainer.save_metrics(prefix, metrics)


def run(cfg: DictConfig) -> None:
    _validate_config(cfg)
    trainer_config = _trainer_config(cfg)
    checkpoint = _resolve_checkpoint(
        cfg.checkpoint,
        must_exist=cfg.mode == "train" and bool(cfg.checkpoint),
    )
    seed_everything(
        trainer_config.seed,
        deterministic=trainer_config.full_determinism,
    )

    LOGGER.info("Configuration:\n%s", OmegaConf.to_yaml(cfg, resolve=True))
    processor = _load_processor(cfg, checkpoint)
    # Load/download model before expensive audio preprocessing fails late.
    model = _load_model(cfg, trainer_config, checkpoint)
    LOGGER.info("Model parameters: %s", model.parameter_counts())
    raw_dataset = load_vimd(
        _absolute_local_path(str(cfg.data.data_dir)),
        sampling_rate=int(cfg.data.sampling_rate),
    )
    selected_dataset = _selected_raw_splits(cfg, raw_dataset)
    dataset = prepare_dataset(
        selected_dataset,
        processor,
        audio_column=str(cfg.data.audio_column),
        text_column=str(cfg.data.text_column),
        num_proc=cfg.data.num_proc,
    )
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    metric_split = "validation" if cfg.mode == "train" else str(cfg.split)
    compute_metrics = build_compute_metrics(
        processor,
        dataset[metric_split]["region"],
    )
    trainer = create_trainer(
        model=model,
        processor=processor,
        data_collator=data_collator,
        train_dataset=dataset.get("train"),
        eval_dataset=dataset[metric_split],
        compute_metrics=compute_metrics,
        config=trainer_config,
    )

    if cfg.mode == "train":
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        _save_metrics(trainer, "train", train_result.metrics)
        trainer.save_state()

        final_model_dir = _absolute_local_path(str(cfg.final_model_dir))
        trainer.save_model(final_model_dir)
        processor.save_pretrained(final_model_dir)
        LOGGER.info("Saved final model and processor to %s", final_model_dir)

        if cfg.evaluate_after_train:
            compute_metrics.set_regions(dataset["test"]["region"])
            test_metrics = trainer.evaluate(
                eval_dataset=dataset["test"],
                metric_key_prefix="test",
            )
            _save_metrics(trainer, "test", test_metrics)
        return

    eval_metrics = trainer.evaluate(
        eval_dataset=dataset[str(cfg.split)],
        metric_key_prefix=str(cfg.split),
    )
    _save_metrics(trainer, str(cfg.split), eval_metrics)


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
