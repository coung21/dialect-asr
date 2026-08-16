from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from dialect_asr.data import (
    DataCollatorSpeechSeq2SeqWithPadding,
    _split_files,
    prepare_example,
)


class FakeProcessor:
    """Minimal processor double; tests data logic without downloading a model."""

    def __init__(self) -> None:
        self.audio_calls: list[tuple[object, int]] = []
        self.text_calls: list[str] = []
        self.feature_extractor = SimpleNamespace(pad=self._pad_features)
        self.tokenizer = SimpleNamespace(pad=self._pad_labels, bos_token_id=1)

    def __call__(self, audio=None, *, sampling_rate=None, text=None):
        if text is not None:
            self.text_calls.append(text)
            # Every example is prefixed with the shared bos/decoder-start token.
            return SimpleNamespace(
                input_ids=[1] + [ord(char) % 20 + 2 for char in text]
            )
        self.audio_calls.append((audio, sampling_rate))
        # [T_audio] -> [1, num_mel_bins=1, T_audio]; a single "mel bin" is enough
        # to exercise the batch/feature-dimension bookkeeping in tests.
        return SimpleNamespace(input_features=[[[float(value) for value in audio]]])

    @staticmethod
    def _pad_features(features, *, return_tensors=None):
        del return_tensors
        max_length = max(len(item["input_features"][0]) for item in features)
        padded = []
        for item in features:
            row = list(item["input_features"][0])
            row += [0.0] * (max_length - len(row))
            padded.append([row])
        return {"input_features": torch.tensor(padded, dtype=torch.float32)}

    @staticmethod
    def _pad_labels(features, *, pad_to_multiple_of=None, return_tensors=None):
        del return_tensors
        max_length = max(len(item["input_ids"]) for item in features)
        if pad_to_multiple_of:
            remainder = max_length % pad_to_multiple_of
            if remainder:
                max_length += pad_to_multiple_of - remainder

        padded, masks = [], []
        for item in features:
            value = list(item["input_ids"])
            padding_length = max_length - len(value)
            padded.append(value + [0] * padding_length)
            masks.append([1] * len(value) + [0] * padding_length)

        return {
            "input_ids": torch.tensor(padded, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
        }


def test_prepare_example_uses_vimd_audio_and_text() -> None:
    processor = FakeProcessor()
    example = {
        "audio": {"array": [0.1, -0.2, 0.3], "sampling_rate": 16_000},
        "text": "  XIN,   Chào!  ",
        "region": "North",
    }

    prepared = prepare_example(example, processor)

    assert prepared["input_features"] == [[pytest.approx(0.1), pytest.approx(-0.2), pytest.approx(0.3)]]
    assert len(prepared["labels"]) == len("xin chào") + 1
    assert processor.text_calls == ["xin chào"]
    assert processor.audio_calls == [([0.1, -0.2, 0.3], 16_000)]


def test_collator_stacks_features_and_replaces_label_padding_with_minus_100() -> None:
    collator = DataCollatorSpeechSeq2SeqWithPadding(
        FakeProcessor(),
        pad_to_multiple_of_labels=4,
    )
    batch = collator(
        [
            {"input_features": [[0.1, 0.2, 0.3]], "labels": [1, 4, 5]},
            {"input_features": [[0.4, 0.0, 0.0]], "labels": [1, 6]},
        ]
    )

    assert batch["input_features"].shape == (2, 1, 3)
    # The shared leading bos/decoder-start token (id=1) is stripped from every
    # example so forward()'s right-shift does not duplicate it.
    assert batch["labels"].tolist() == [[4, 5, -100], [6, -100, -100]]


def test_collator_keeps_labels_when_no_shared_leading_bos_token() -> None:
    collator = DataCollatorSpeechSeq2SeqWithPadding(FakeProcessor())

    batch = collator(
        [
            {"input_features": [[0.1, 0.2]], "labels": [4, 5]},
            {"input_features": [[0.3]], "labels": [6]},
        ]
    )

    assert batch["labels"].tolist() == [[4, 5], [6, -100]]


def test_collator_rejects_empty_batch() -> None:
    with pytest.raises(ValueError, match="batch rỗng"):
        DataCollatorSpeechSeq2SeqWithPadding(FakeProcessor())([])


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
