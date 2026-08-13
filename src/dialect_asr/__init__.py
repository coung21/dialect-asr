"""Dialect-aware Vietnamese speech recognition."""

from .base_model import AbstractWav2Vec2CTC
from .data import (
    DataCollatorCTCWithPadding,
    load_vimd,
    prepare_dataset,
    prepare_example,
    region_to_label,
)
from .dggfm_model import DGGFMCTCOutput, DGGFMWav2Vec2CTC
from .evaluation import (
    CTCMetrics,
    MultitaskMetrics,
    build_compute_metrics,
    compute_asr_metrics,
)
from .model import (
    DEFAULT_PRETRAINED_MODEL,
    BaselineWav2Vec2CTC,
    load_vietnamese_processor,
)
from .modules import DGGFM, DialectBranch, SoftDialectEmbedding
from .multitask_model import MultitaskCTCOutput, MultitaskWav2Vec2CTC
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
    MultitaskTrainer,
    TrainerConfig,
    build_training_arguments,
    configure_wandb_environment,
    create_trainer,
    preprocess_logits_for_ctc,
    preprocess_logits_for_multitask,
)

__all__ = [
    "AbstractWav2Vec2CTC",
    "DataCollatorCTCWithPadding",
    "DGGFMCTCOutput",
    "DGGFMWav2Vec2CTC",
    "load_vimd",
    "prepare_dataset",
    "prepare_example",
    "region_to_label",
    "CTCMetrics",
    "MultitaskMetrics",
    "build_compute_metrics",
    "compute_asr_metrics",
    "DEFAULT_PRETRAINED_MODEL",
    "BaselineWav2Vec2CTC",
    "load_vietnamese_processor",
    "DialectBranch",
    "DGGFM",
    "SoftDialectEmbedding",
    "MultitaskCTCOutput",
    "MultitaskWav2Vec2CTC",
    "DEFAULT_SEED",
    "seed_data_worker",
    "seed_everything",
    "seeded_generator",
    "MODEL_REGISTRY",
    "architecture_from_checkpoint",
    "build_project_model",
    "get_model_class",
    "TrainerConfig",
    "MultitaskTrainer",
    "build_training_arguments",
    "configure_wandb_environment",
    "create_trainer",
    "preprocess_logits_for_ctc",
    "preprocess_logits_for_multitask",
]
