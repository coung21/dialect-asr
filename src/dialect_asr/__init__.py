"""Dialect-aware Vietnamese speech recognition."""

from .adaln_model import PhoWhisperAdaLNASR, inject_adaln, resolve_did_checkpoint_path
from .base_model import AbstractPhoWhisperASR
from .data import (
    DataCollatorDIDWithPadding,
    DataCollatorSpeechSeq2SeqWithPadding,
    load_vimd,
    prepare_combined_dataset,
    prepare_combined_example,
    prepare_dataset,
    prepare_did_dataset,
    prepare_did_example,
    prepare_example,
    region_to_label,
)
from .evaluation import Seq2SeqMetrics, build_compute_metrics, compute_asr_metrics
from .model import (
    DEFAULT_PRETRAINED_MODEL,
    BaselinePhoWhisperASR,
    load_vietnamese_processor,
)
from .modules import AdaLN, DialectConditioner, ECAPA_TDNN_DID
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
    "AdaLN",
    "DialectConditioner",
    "PhoWhisperAdaLNASR",
    "inject_adaln",
    "resolve_did_checkpoint_path",
    "DataCollatorDIDWithPadding",
    "DataCollatorSpeechSeq2SeqWithPadding",
    "load_vimd",
    "prepare_combined_dataset",
    "prepare_combined_example",
    "prepare_dataset",
    "prepare_did_dataset",
    "prepare_did_example",
    "prepare_example",
    "region_to_label",
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
