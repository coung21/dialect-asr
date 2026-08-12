"""Data loading and preprocessing utilities for the ViMD ASR dataset."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from datasets import Audio, load_dataset

import torch

from dialect_asr.text import normalize_vietnamese_text


VIMD_COLUMNS = {
    "audio",
    "text",
    "region",       
    "province_code",
    "province_name",
    "filename",
    "speakerID",
    "gender",
}

REGION_TO_LABEL = {
    "north": 0,
    "central": 1,
    "south": 2,
}


def region_to_label(region: str) -> int:
    """Map a ViMD region name to the class ID consumed by DGGFM."""
    normalized_region = str(region).strip().lower()
    try:
        return REGION_TO_LABEL[normalized_region]
    except KeyError as exc:
        expected = ", ".join(REGION_TO_LABEL)
        raise ValueError(
            f"Region không hợp lệ {region!r}; cần một trong: {expected}"
        ) from exc


def _split_files(data_dir: str | Path) -> dict[str, str]:
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Không tìm thấy thư mục ViMD: {data_dir}")

    patterns = {
        "train": "train-*.parquet",
        "validation": "valid-*.parquet",
        "test": "test-*.parquet",
    }
    missing = [pattern for pattern in patterns.values() if not any(data_dir.glob(pattern))]
    if missing:
        raise FileNotFoundError(
            f"Thiếu các shard ViMD trong {data_dir}: {', '.join(missing)}"
        )

    return {split: str(data_dir / pattern) for split, pattern in patterns.items()}


def load_vimd(
    data_dir: str | Path,
    sampling_rate: int = 16_000,
) -> Any:
    """Load local ViMD parquet shards and decode audio at ``sampling_rate``."""
    dataset = load_dataset("parquet", data_files=_split_files(data_dir))

    missing_columns = VIMD_COLUMNS.difference(dataset["train"].column_names)
    if missing_columns:
        raise ValueError(
            "Schema ViMD thiếu cột: " + ", ".join(sorted(missing_columns))
        )

    return dataset.cast_column("audio", Audio(sampling_rate=sampling_rate))


def _decoded_audio(audio: Any) -> tuple[Any, int]:
    """Return mono audio with shape ``[T_audio]`` and its sampling rate."""
    if isinstance(audio, Mapping):
        # audio["array"]: [T_audio] for mono audio.
        return audio["array"], int(audio["sampling_rate"])

    if hasattr(audio, "get_all_samples"):
        samples = audio.get_all_samples()
        array = samples.data  # [C, T_audio] from torchcodec.
        if getattr(array, "ndim", 1) == 2:
            array = array.mean(dim=0)  # [C, T_audio] -> [T_audio].
        return array, int(samples.sample_rate)

    raise TypeError(
        "Cột audio phải là dict {'array', 'sampling_rate'} hoặc AudioDecoder"
    )


def prepare_example(
    example: dict[str, Any],
    processor: Any,
    audio_column: str = "audio",
    text_column: str = "text",
    region_column: str = "region",
) -> dict[str, Any]:
    """Convert one ViMD record into audio, CTC and region model inputs."""
    audio_array, sampling_rate = _decoded_audio(example[audio_column])
    # audio_array: [T_audio] -> processor batch output: [1, T_audio].
    audio_output = processor(audio_array, sampling_rate=sampling_rate)
    normalized_text = normalize_vietnamese_text(str(example[text_column]))
    # One normalized transcript -> token IDs with shape [T_text].
    text_output = processor(text=normalized_text)

    return {
        # [1, T_audio] -> [T_audio]; collator restores the batch dimension.
        "input_values": audio_output.input_values[0],
        "labels": text_output.input_ids,  # [T_text].
        "region_labels": region_to_label(example[region_column]),  # Scalar class ID [].
    }


def prepare_dataset(
    dataset: Any,
    processor: Any,
    audio_column: str = "audio",
    text_column: str = "text",
    region_column: str = "region",
    num_proc: int | None = None,
) -> Any:
    """Preprocess every split while retaining transcript and dialect metadata."""

    def preprocess(example: dict[str, Any]) -> dict[str, Any]:
        return prepare_example(
            example,
            processor,
            audio_column,
            text_column,
            region_column,
        )

    return dataset.map(
        preprocess,
        remove_columns=[audio_column],
        num_proc=num_proc,
        desc="Preprocessing ViMD",
    )


@dataclass(slots=True)
class DataCollatorCTCWithPadding:
    """Dynamically pad variable-length audio and labels for CTC training."""

    processor: Any
    padding: bool | str = True
    pad_to_multiple_of: int | None = None
    pad_to_multiple_of_labels: int | None = None

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        if not examples:
            raise ValueError("Không thể collate một batch rỗng")

        audio_features = [
            # Each item remains [T_audio_i] before dynamic padding.
            {"input_values": example["input_values"]} for example in examples
        ]
        # Each label item has variable shape [T_text_i].
        label_features = [{"input_ids": example["labels"]} for example in examples]

        # [T_audio_i] -> input_values [B, T_audio_max].
        # The optional attention_mask has shape [B, T_audio_max].
        batch = self.processor.pad(
            audio_features,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        try:
            # [T_text_i] -> input_ids/attention_mask [B, T_text_max].
            label_batch = self.processor.pad(
                labels=label_features,
                padding=self.padding,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )
        except TypeError:
            # Compatibility with processors that delegate text padding to tokenizer.
            # [T_text_i] -> input_ids/attention_mask [B, T_text_max].
            label_batch = self.processor.tokenizer.pad(
                label_features,
                padding=self.padding,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # Both operands are [B, T_text_max]; output labels keep that shape.
        # Padding IDs become -100 so Wav2Vec2ForCTC ignores those positions.
        batch["labels"] = label_batch["input_ids"].masked_fill(
            label_batch["attention_mask"].ne(1), -100
        )

        has_region_labels = ["region_labels" in example for example in examples]
        if any(has_region_labels) and not all(has_region_labels):
            raise ValueError("Mọi example trong batch phải cùng có region_labels")
        if all(has_region_labels):
            batch["region_labels"] = torch.tensor(
                [example["region_labels"] for example in examples],
                dtype=torch.long,
            )  # B scalar region IDs [] -> region labels [B].
        return batch
