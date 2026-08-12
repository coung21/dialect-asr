from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from dialect_asr.data import (
    DataCollatorCTCWithPadding,
    _split_files,
    prepare_example,
    region_to_label,
)


class FakeProcessor:
    """Minimal processor double; tests data logic without downloading a model."""

    def __init__(self) -> None:
        self.audio_calls: list[tuple[object, int]] = []
        self.text_calls: list[str] = []

    def __call__(self, audio=None, *, sampling_rate=None, text=None):
        if text is not None:
            self.text_calls.append(text)
            return SimpleNamespace(input_ids=[ord(char) % 20 + 1 for char in text])
        self.audio_calls.append((audio, sampling_rate))
        return SimpleNamespace(input_values=[[float(value) for value in audio]])

    def pad(
        self,
        features=None,
        *,
        labels=None,
        padding=True,
        pad_to_multiple_of=None,
        return_tensors=None,
    ):
        del padding, return_tensors
        values = labels if labels is not None else features
        key = "input_ids" if labels is not None else "input_values"
        max_length = max(len(item[key]) for item in values)
        if pad_to_multiple_of:
            remainder = max_length % pad_to_multiple_of
            if remainder:
                max_length += pad_to_multiple_of - remainder

        padded, masks = [], []
        for item in values:
            value = list(item[key])
            padding_length = max_length - len(value)
            padded.append(value + [0] * padding_length)
            masks.append([1] * len(value) + [0] * padding_length)

        dtype = torch.long if key == "input_ids" else torch.float32
        return {
            key: torch.tensor(padded, dtype=dtype),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
        }


def test_prepare_example_uses_vimd_audio_and_text() -> None:
    processor = FakeProcessor()
    example = {
        "audio": {"array": [0.1, -0.2, 0.3], "sampling_rate": 16_000},
        "text": "  XIN,   Chào!  ",
        "region": "North",
    }

    prepared = prepare_example(example, processor)

    assert prepared["input_values"] == pytest.approx([0.1, -0.2, 0.3])
    assert len(prepared["labels"]) == len("xin chào")
    assert prepared["region_labels"] == 0
    assert processor.text_calls == ["xin chào"]
    assert processor.audio_calls == [([0.1, -0.2, 0.3], 16_000)]


def test_collator_pads_audio_and_replaces_label_padding_with_minus_100() -> None:
    collator = DataCollatorCTCWithPadding(
        FakeProcessor(),
        pad_to_multiple_of=4,
    )
    batch = collator(
        [
            {"input_values": [0.1, 0.2, 0.3], "labels": [4, 5]},
            {"input_values": [0.4], "labels": [6]},
        ]
    )

    assert batch["input_values"].shape == (2, 4)
    assert batch["attention_mask"].tolist() == [[1, 1, 1, 0], [1, 0, 0, 0]]
    assert batch["labels"].tolist() == [[4, 5], [6, -100]]


def test_collator_builds_region_label_tensor() -> None:
    collator = DataCollatorCTCWithPadding(FakeProcessor())

    batch = collator(
        [
            {"input_values": [0.1, 0.2], "labels": [4], "region_labels": 0},
            {"input_values": [0.3], "labels": [5], "region_labels": 2},
        ]
    )

    assert batch["region_labels"].shape == (2,)  # B scalar IDs -> [B=2].
    assert batch["region_labels"].dtype == torch.long
    assert batch["region_labels"].tolist() == [0, 2]


def test_collator_rejects_partially_missing_region_labels() -> None:
    collator = DataCollatorCTCWithPadding(FakeProcessor())

    with pytest.raises(ValueError, match="cùng có region_labels"):
        collator(
            [
                {"input_values": [0.1], "labels": [4], "region_labels": 0},
                {"input_values": [0.2], "labels": [5]},
            ]
        )


@pytest.mark.parametrize(
    ("region", "expected"),
    [("North", 0), (" central ", 1), ("SOUTH", 2)],
)
def test_region_to_label(region, expected) -> None:
    assert region_to_label(region) == expected


def test_region_to_label_rejects_unknown_region() -> None:
    with pytest.raises(ValueError, match="Region không hợp lệ"):
        region_to_label("West")


def test_collator_rejects_empty_batch() -> None:
    with pytest.raises(ValueError, match="batch rỗng"):
        DataCollatorCTCWithPadding(FakeProcessor())([])


def test_split_files_maps_vimd_names(tmp_path: Path) -> None:
    for name in ("train-000.parquet", "valid-000.parquet", "test-000.parquet"):
        (tmp_path / name).touch()

    files = _split_files(tmp_path)

    assert files == {
        "train": str(tmp_path / "train-*.parquet"),
        "validation": str(tmp_path / "valid-*.parquet"),
        "test": str(tmp_path / "test-*.parquet"),
    }


def test_split_files_reports_missing_shards(tmp_path: Path) -> None:
    (tmp_path / "train-000.parquet").touch()

    with pytest.raises(FileNotFoundError, match="valid-.*test-"):
        _split_files(tmp_path)
