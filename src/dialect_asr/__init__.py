"""Dialect-aware Vietnamese speech recognition."""

from .data import (
    DataCollatorCTCWithPadding,
    load_vimd,
    prepare_dataset,
    prepare_example,
)
from .evaluation import CTCMetrics, build_compute_metrics, compute_asr_metrics
from .model import (
    DEFAULT_PRETRAINED_MODEL,
    BaselineWav2Vec2CTC,
    load_vietnamese_processor,
)
from .reproducibility import (
    DEFAULT_SEED,
    seed_data_worker,
    seed_everything,
    seeded_generator,
)
from .trainer import (
    TrainerConfig,
    build_training_arguments,
    configure_wandb_environment,
    create_trainer,
    preprocess_logits_for_ctc,
)

__all__ = [
    "DataCollatorCTCWithPadding",
    "load_vimd",
    "prepare_dataset",
    "prepare_example",
    "CTCMetrics",
    "build_compute_metrics",
    "compute_asr_metrics",
    "DEFAULT_PRETRAINED_MODEL",
    "BaselineWav2Vec2CTC",
    "load_vietnamese_processor",
    "DEFAULT_SEED",
    "seed_data_worker",
    "seed_everything",
    "seeded_generator",
    "TrainerConfig",
    "build_training_arguments",
    "configure_wandb_environment",
    "create_trainer",
    "preprocess_logits_for_ctc",
]
