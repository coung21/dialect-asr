"""Dialect-aware Vietnamese speech recognition."""

from .base_model import AbstractPhoWhisperASR
from .data import (
    DataCollatorSpeechSeq2SeqWithPadding,
    load_vimd,
    prepare_dataset,
    prepare_example,
)
from .evaluation import Seq2SeqMetrics, build_compute_metrics, compute_asr_metrics
from .model import (
    DEFAULT_PRETRAINED_MODEL,
    BaselinePhoWhisperASR,
    load_vietnamese_processor,
)
from .modules import ECAPA_TDNN_DID
from .reproducibility import (
    DEFAULT_SEED,
    seed_data_worker,
    seed_everything,
    seeded_generator,
)
from .registry import (
    MODEL_REGISTRY,
    architecture_from_checkpoint,
    build_project_model,
    get_model_class,
)
from .trainer import (
    TrainerConfig,
    build_training_arguments,
    configure_wandb_environment,
    create_trainer,
    reports_to_wandb,
)

__all__ = [
    "AbstractPhoWhisperASR",
    "DataCollatorSpeechSeq2SeqWithPadding",
    "load_vimd",
    "prepare_dataset",
    "prepare_example",
    "Seq2SeqMetrics",
    "build_compute_metrics",
    "compute_asr_metrics",
    "DEFAULT_PRETRAINED_MODEL",
    "BaselinePhoWhisperASR",
    "load_vietnamese_processor",
    "ECAPA_TDNN_DID",
    "DEFAULT_SEED",
    "seed_data_worker",
    "seed_everything",
    "seeded_generator",
    "MODEL_REGISTRY",
    "architecture_from_checkpoint",
    "build_project_model",
    "get_model_class",
    "TrainerConfig",
    "build_training_arguments",
    "configure_wandb_environment",
    "create_trainer",
    "reports_to_wandb",
]
