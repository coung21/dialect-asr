"""Training utilities for the baseline PhoWhisper seq2seq model."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

from .reproducibility import DEFAULT_SEED, seed_everything


@dataclass(slots=True)
class TrainerConfig:
    """Serializable training configuration suitable for Hydra/YAML later."""

    output_dir: str = "outputs/phowhisper-baseline"
    num_train_epochs: float = 15.0
    learning_rate: float = 1e-4
    weight_decay: float = 0.005
    # Defaults target one GPU with a real batch of 8 and no accumulation.
    global_train_batch_size: int = 8
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 4
    warmup_ratio: float = 0.1
    optimizer: str = "adamw_torch"
    max_grad_norm: float = 1.0
    logging_steps: int = 25
    save_total_limit: int = 2
    dataloader_num_workers: int = 4
    group_by_length: bool = True
    length_column_name: str = "length"
    seed: int = DEFAULT_SEED
    full_determinism: bool = False
    fp16: bool = False
    bf16: bool = False
    tf32: bool | None = None
    gradient_checkpointing: bool = False
    # Greedy decoding by default; raise for beam search at eval-time cost.
    generation_max_length: int = 225
    generation_num_beams: int = 1
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    report_to: str | list[str] = "wandb"
    run_name: str | None = "phowhisper-baseline"
    wandb_project: str = "dialect-asr"
    wandb_entity: str | None = None
    wandb_group: str | None = "baseline"
    wandb_tags: tuple[str, ...] | list[str] = ("vimd", "phowhisper", "baseline")
    wandb_mode: str = "online"
    # Hydra parses an unquoted CLI value `false` as bool False.
    wandb_log_model: str | bool = "end"
    wandb_watch: str = "false"
    use_cpu: bool = False

    def __post_init__(self) -> None:
        positive_fields = {
            "num_train_epochs": self.num_train_epochs,
            "learning_rate": self.learning_rate,
            "global_train_batch_size": self.global_train_batch_size,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "per_device_eval_batch_size": self.per_device_eval_batch_size,
            "logging_steps": self.logging_steps,
            "generation_max_length": self.generation_max_length,
            "generation_num_beams": self.generation_num_beams,
        }
        invalid = [name for name, value in positive_fields.items() if value <= 0]
        if invalid:
            raise ValueError("Các giá trị phải > 0: " + ", ".join(invalid))
        if self.fp16 and self.bf16:
            raise ValueError("Không thể bật đồng thời fp16 và bf16")
        if not 0.0 <= self.warmup_ratio < 1.0:
            raise ValueError("warmup_ratio phải nằm trong khoảng [0, 1)")
        if self.wandb_mode not in {"online", "offline", "disabled"}:
            raise ValueError("wandb_mode phải là online, offline hoặc disabled")
        if self.wandb_log_model is False:
            self.wandb_log_model = "false"
        elif self.wandb_log_model is True:
            raise ValueError("wandb_log_model chỉ được là false hoặc end")
        if self.wandb_log_model not in {"false", "end"}:
            raise ValueError("wandb_log_model chỉ được là false hoặc end")
        if self.wandb_watch not in {"false", "gradients", "all"}:
            raise ValueError("wandb_watch phải là false, gradients hoặc all")


def _reports_to_wandb(report_to: str | list[str]) -> bool:
    if isinstance(report_to, str):
        return report_to == "wandb" or report_to == "all"
    return "wandb" in report_to or "all" in report_to


def configure_wandb_environment(config: TrainerConfig) -> None:
    """Configure the environment consumed by Transformers' WandbCallback."""
    if not _reports_to_wandb(config.report_to):
        return

    os.environ["WANDB_PROJECT"] = config.wandb_project
    os.environ["WANDB_MODE"] = config.wandb_mode
    os.environ["WANDB_LOG_MODEL"] = config.wandb_log_model
    os.environ["WANDB_WATCH"] = config.wandb_watch
    os.environ["WANDB_TAGS"] = ",".join(config.wandb_tags)
    if config.wandb_entity:
        os.environ["WANDB_ENTITY"] = config.wandb_entity
    if config.wandb_group:
        os.environ["WANDB_RUN_GROUP"] = config.wandb_group


def _resolve_gradient_accumulation_steps(
    config: TrainerConfig,
    world_size: int,
) -> int:
    """Derive accumulation so the effective global batch is exactly configured."""
    if world_size <= 0:
        raise ValueError("world_size phải > 0")

    micro_global_batch = config.per_device_train_batch_size * world_size
    if config.global_train_batch_size % micro_global_batch != 0:
        raise ValueError(
            "global_train_batch_size phải chia hết cho "
            "per_device_train_batch_size * world_size; "
            f"nhận {config.global_train_batch_size} và {micro_global_batch}"
        )
    return config.global_train_batch_size // micro_global_batch


def build_training_arguments(
    config: TrainerConfig,
    *,
    has_eval_dataset: bool = True,
    world_size: int | None = None,
) -> Seq2SeqTrainingArguments:
    """Convert the project config into Hugging Face seq2seq training arguments."""
    configure_wandb_environment(config)
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    eval_strategy = "epoch" if has_eval_dataset else "no"
    save_strategy = "epoch"
    load_best_model = has_eval_dataset
    if world_size is None:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
    gradient_accumulation_steps = _resolve_gradient_accumulation_steps(
        config,
        world_size,
    )

    return Seq2SeqTrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        # Transformers 5 interprets float 0 <= warmup_steps < 1 as a ratio.
        warmup_steps=config.warmup_ratio,
        max_grad_norm=config.max_grad_norm,
        optim=config.optimizer,
        eval_strategy=eval_strategy,
        save_strategy=save_strategy,
        logging_strategy="steps",
        logging_steps=config.logging_steps,
        logging_first_step=True,
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=load_best_model,
        metric_for_best_model=(
            config.metric_for_best_model if has_eval_dataset else None
        ),
        greater_is_better=(config.greater_is_better if has_eval_dataset else None),
        fp16=config.fp16,
        bf16=config.bf16,
        tf32=config.tf32,
        gradient_checkpointing=config.gradient_checkpointing,
        dataloader_num_workers=config.dataloader_num_workers,
        dataloader_pin_memory=not config.use_cpu,
        # Transformers 5 renamed group_by_length to train_sampling_strategy.
        train_sampling_strategy=("group_by_length" if config.group_by_length else "random"),
        length_column_name=config.length_column_name,
        remove_unused_columns=True,
        label_names=["labels"],
        report_to=config.report_to,
        run_name=config.run_name,
        seed=config.seed,
        data_seed=config.seed,
        full_determinism=config.full_determinism,
        use_cpu=config.use_cpu,
        # Decode with generate() during evaluation instead of scoring raw logits.
        predict_with_generate=has_eval_dataset,
        generation_max_length=config.generation_max_length,
        generation_num_beams=config.generation_num_beams,
    )


def create_trainer(
    *,
    model: Any,
    processor: Any,
    data_collator: Any,
    train_dataset: Any,
    eval_dataset: Any | None = None,
    compute_metrics: Any | None = None,
    config: TrainerConfig | None = None,
    callbacks: list[Any] | None = None,
) -> Seq2SeqTrainer:
    """Create a Seq2SeqTrainer wired to the project's model, processor and datasets."""
    config = config or TrainerConfig()
    # Reset every RNG before Trainer builds random samplers and DataLoader workers.
    seed_everything(config.seed, deterministic=config.full_determinism)
    training_args = build_training_arguments(
        config,
        has_eval_dataset=eval_dataset is not None,
    )

    return Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processor,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )
