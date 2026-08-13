"""Training utilities for the baseline Wav2Vec2 CTC model."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from transformers import Trainer, TrainingArguments

from .reproducibility import DEFAULT_SEED, seed_everything


@dataclass(slots=True)
class TrainerConfig:
    """Serializable training configuration suitable for Hydra/YAML later."""

    output_dir: str = "outputs/wav2vec2-baseline"
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
    # CUDA CTC backward has no deterministic implementation in PyTorch.
    full_determinism: bool = False
    fp16: bool = False
    bf16: bool = False
    tf32: bool | None = None
    gradient_checkpointing: bool = False
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    report_to: str | list[str] = "wandb"
    run_name: str | None = "wav2vec2-baseline"
    wandb_project: str = "dialect-asr"
    wandb_entity: str | None = None
    wandb_group: str | None = "baseline"
    wandb_tags: tuple[str, ...] | list[str] = ("vimd", "wav2vec2", "baseline")
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
    label_names: list[str] | None = None,
) -> TrainingArguments:
    """Convert the project config into Hugging Face training arguments."""
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

    return TrainingArguments(
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
        label_names=label_names or ["labels"],
        report_to=config.report_to,
        run_name=config.run_name,
        seed=config.seed,
        data_seed=config.seed,
        full_determinism=config.full_determinism,
        use_cpu=config.use_cpu,
    )


def preprocess_logits_for_ctc(
    logits: Tensor | tuple[Tensor, ...],
    labels: Tensor | None = None,
) -> Tensor:
    """Keep only greedy token IDs while accumulating evaluation predictions."""
    del labels  # labels: [B, T_text]; unused for greedy selection.
    if isinstance(logits, tuple):
        logits = logits[0]  # First item: [B, T_frame, V].
    return torch.argmax(logits, dim=-1)  # [B, T_frame, V] -> [B, T_frame].


def preprocess_logits_for_multitask(
    logits: tuple[Tensor, Tensor, Tensor, Tensor],
    labels: Any = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Reduce CTC logits while preserving dialect outputs and component losses."""
    del labels
    if not isinstance(logits, tuple) or len(logits) != 4:
        raise ValueError(
            "Multitask logits phải gồm "
            "(ctc_logits, dialect_logits, ctc_losses, dialect_losses)"
        )
    ctc_logits, dialect_logits, ctc_losses, dialect_losses = logits
    ctc_ids = torch.argmax(ctc_logits, dim=-1)
    # [B, T_frame, V] -> greedy CTC IDs [B, T_frame].
    return ctc_ids, dialect_logits, ctc_losses, dialect_losses


class MultitaskTrainer(Trainer):
    """Trainer that logs loss components and retains dialect eval outputs."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._train_component_sums = {"ctc_loss": 0.0, "dialect_loss": 0.0}
        self._train_component_count = 0
        super().__init__(*args, **kwargs)
        self.model_accepts_loss_kwargs = False

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, Tensor | Any],
        return_outputs: bool = False,
        num_items_in_batch: Tensor | int | None = None,
    ) -> Tensor | tuple[Tensor, Any]:
        del num_items_in_batch
        outputs = model(**inputs)
        loss = outputs.loss
        if loss is None:
            raise ValueError("Multitask model không trả về total loss")

        if model.training:
            for name in self._train_component_sums:
                component = getattr(outputs, name, None)
                if component is None:
                    raise ValueError(f"Multitask model không trả về {name}")
                self._train_component_sums[name] += float(component.detach().mean())
                # Component loss tensor [] -> detached Python scalar.
            self._train_component_count += 1

        return (loss, outputs) if return_outputs else loss

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        if "loss" in logs and self._train_component_count > 0:
            logs = dict(logs)
            for name, component_sum in self._train_component_sums.items():
                logs[name] = component_sum / self._train_component_count
            self._train_component_sums = {"ctc_loss": 0.0, "dialect_loss": 0.0}
            self._train_component_count = 0
        super().log(logs, start_time=start_time)

    def prediction_step(
        self,
        model: torch.nn.Module,
        inputs: dict[str, Tensor | Any],
        prediction_loss_only: bool,
        ignore_keys: list[str] | None = None,
    ) -> tuple[Tensor | None, Any, Any]:
        del ignore_keys
        inputs = self._prepare_inputs(inputs)
        ctc_labels = inputs.get("labels")
        region_labels = inputs.get("region_labels")
        if ctc_labels is None or region_labels is None:
            raise ValueError("Multitask eval yêu cầu labels và region_labels")

        with torch.no_grad(), self.compute_loss_context_manager():
            outputs = model(**inputs)
        if outputs.loss is None or outputs.ctc_loss is None or outputs.dialect_loss is None:
            raise ValueError("Multitask eval không trả về đầy đủ loss components")

        loss = outputs.loss.detach().mean()  # Total loss [] -> detached scalar [].
        if prediction_loss_only:
            return loss, None, None

        batch_size = outputs.logits.shape[0]
        ctc_losses = outputs.ctc_loss.detach().reshape(1).expand(batch_size)
        # Scalar CTC loss [] -> per-example logging vector [B].
        dialect_losses = outputs.dialect_loss.detach().reshape(1).expand(batch_size)
        # Scalar dialect loss [] -> per-example logging vector [B].
        logits = (
            outputs.logits.detach(),  # [B, T_frame, V].
            outputs.dialect_logits.detach(),  # [B, R].
            ctc_losses,  # [B].
            dialect_losses,  # [B].
        )
        labels = (
            ctc_labels.detach(),  # [B, T_text].
            region_labels.detach(),  # [B].
        )
        return loss, logits, labels


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
) -> Trainer:
    """Create a Trainer wired to the project's model, processor and datasets."""
    config = config or TrainerConfig()
    # Reset every RNG before Trainer builds random samplers and DataLoader workers.
    seed_everything(config.seed, deterministic=config.full_determinism)
    training_args = build_training_arguments(
        config,
        has_eval_dataset=eval_dataset is not None,
        label_names=(
            ["labels", "region_labels"]
            if getattr(getattr(model, "config", None), "architecture", None)
            == "multitask"
            else ["labels"]
        ),
    )

    is_multitask = (
        getattr(getattr(model, "config", None), "architecture", None)
        == "multitask"
    )
    trainer_class = MultitaskTrainer if is_multitask else Trainer
    preprocess_logits = (
        preprocess_logits_for_multitask if is_multitask else preprocess_logits_for_ctc
    )
    return trainer_class(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processor,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
        # During eval, logits [B, T_frame, V] are reduced to IDs [B, T_frame].
        preprocess_logits_for_metrics=preprocess_logits,
    )
