"""Single-source reproducibility controls for the whole project."""

from __future__ import annotations

import os
import random

import numpy as np
import torch
from transformers import enable_full_determinism, set_seed


DEFAULT_SEED = 42


def seed_everything(
    seed: int = DEFAULT_SEED,
    *,
    deterministic: bool = True,
    warn_only: bool = False,
) -> int:
    """Seed Python, NumPy, PyTorch CPU/CUDA and deterministic backends."""
    if seed < 0:
        raise ValueError("seed phải >= 0")

    # Affects child processes; the current interpreter's hash seed is fixed at startup.
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        # Seeds Python/NumPy/Torch and configures deterministic CUDA/cuDNN kernels.
        enable_full_determinism(seed, warn_only=warn_only)
    else:
        set_seed(seed)
    return seed


def seed_data_worker(worker_id: int) -> None:
    """Seed NumPy/Python inside a custom torch DataLoader worker."""
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def seeded_generator(seed: int = DEFAULT_SEED) -> torch.Generator:
    """Create a reproducible generator for custom DataLoader instances."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator
