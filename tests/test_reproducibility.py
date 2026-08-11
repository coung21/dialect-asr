import random

import numpy as np
import pytest
import torch

from dialect_asr.reproducibility import (
    DEFAULT_SEED,
    seed_everything,
    seeded_generator,
)


def random_snapshot() -> tuple[float, np.ndarray, torch.Tensor]:
    return (
        random.random(),
        np.random.rand(4),  # [4].
        torch.rand(4),  # [4].
    )


def test_seed_everything_reproduces_python_numpy_and_torch() -> None:
    seed_everything(DEFAULT_SEED)
    first = random_snapshot()
    seed_everything(DEFAULT_SEED)
    second = random_snapshot()

    assert first[0] == second[0]
    np.testing.assert_array_equal(first[1], second[1])  # [4] == [4].
    torch.testing.assert_close(first[2], second[2], rtol=0, atol=0)  # [4] == [4].


def test_seeded_generator_reproduces_torch_values() -> None:
    # Each call produces a vector with shape [5].
    first = torch.rand(5, generator=seeded_generator(DEFAULT_SEED))
    second = torch.rand(5, generator=seeded_generator(DEFAULT_SEED))

    torch.testing.assert_close(first, second, rtol=0, atol=0)  # [5] == [5].


def test_seed_must_be_non_negative() -> None:
    with pytest.raises(ValueError, match="seed"):
        seed_everything(-1)
