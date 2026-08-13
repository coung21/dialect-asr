"""Dialect-aware Vietnamese speech recognition."""

from .data import (
    DataCollatorCTCWithPadding,
    load_vimd,
    prepare_dataset,
    prepare_example,
    region_to_label,
)
from .dann_model import DANNCTCOutput, DANNWav2Vec2CTC
from .evaluation import (
    CTCMetrics,
    DANNMetrics,
    build_compute_metrics,
    compute_asr_metrics,
)
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
from .registry import (
    MODEL_REGISTRY,
    architecture_from_checkpoint,
    build_project_model,
    get_model_class,
)
from .trainer import (
    DANNTrainer,
    TrainerConfig,
    build_training_arguments,
    configure_wandb_environment,
    create_trainer,
    preprocess_logits_for_ctc,
    preprocess_logits_for_dann,
)

__all__ = [
    "DataCollatorCTCWithPadding",
    "load_vimd",
    "prepare_dataset",
    "prepare_example",
    "region_to_label",
    "DANNCTCOutput",
    "DANNWav2Vec2CTC",
    "CTCMetrics",
    "DANNMetrics",
    "build_compute_metrics",
    "compute_asr_metrics",
    "DEFAULT_PRETRAINED_MODEL",
    "BaselineWav2Vec2CTC",
    "load_vietnamese_processor",
    "DEFAULT_SEED",
    "seed_data_worker",
    "seed_everything",
    "seeded_generator",
    "MODEL_REGISTRY",
    "architecture_from_checkpoint",
    "build_project_model",
    "get_model_class",
    "DANNTrainer",
    "TrainerConfig",
    "build_training_arguments",
    "configure_wandb_environment",
    "create_trainer",
    "preprocess_logits_for_ctc",
    "preprocess_logits_for_dann",
]
