"""Data loading and preprocessing utilities for the ViMD ASR dataset."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any
from datasets import Audio, load_dataset

import torch

from dialect_asr.text import normalize_vietnamese_text


LOGGER = logging.getLogger(__name__)

# Whisper's decoder has a fixed 448-position embedding table
# (``config.max_target_positions``); any label longer than this crashes
# ``forward`` with "Labels' sequence length ... cannot exceed the maximum
# allowed length". A handful of ViMD transcripts exceed it, so they are
# dropped rather than truncated (truncation would train the model to emit
# mid-sentence cutoffs).
MAX_LABEL_TOKENS = 448


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
) -> dict[str, Any]:
    """Convert one ViMD record into log-mel features and CTC-free text labels."""
    audio_array, sampling_rate = _decoded_audio(example[audio_column])
    # audio_array: [T_audio] -> processor batch output: [1, num_mel_bins, T_frame].
    audio_output = processor(audio_array, sampling_rate=sampling_rate)
    normalized_text = normalize_vietnamese_text(str(example[text_column]))
    # One normalized transcript -> token IDs with shape [T_text].
    text_output = processor(text=normalized_text)

    return {
        # [1, num_mel_bins, T_frame] -> [num_mel_bins, T_frame]; collator restores
        # the batch dimension.
        "input_features": audio_output.input_features[0],
        "labels": text_output.input_ids,  # [T_text].
    }


def prepare_dataset(
    dataset: Any,
    processor: Any,
    audio_column: str = "audio",
    text_column: str = "text",
    num_proc: int | None = None,
) -> Any:
    """Preprocess every split while retaining transcript and dialect metadata."""

    def preprocess(example: dict[str, Any]) -> dict[str, Any]:
        return prepare_example(
            example,
            processor,
            audio_column,
            text_column,
        )

    prepared = dataset.map(
        preprocess,
        remove_columns=[audio_column],
        num_proc=num_proc,
        desc="Preprocessing ViMD",
    )
    filtered = prepared.filter(
        lambda example: len(example["labels"]) <= MAX_LABEL_TOKENS,
        num_proc=num_proc,
        desc="Dropping transcripts longer than Whisper's decoder limit",
    )
    for split_name in filtered:
        dropped = len(prepared[split_name]) - len(filtered[split_name])
        if dropped:
            LOGGER.warning(
                "%s: bỏ %d/%d mẫu có transcript > %d token (giới hạn decoder Whisper)",
                split_name,
                dropped,
                len(prepared[split_name]),
                MAX_LABEL_TOKENS,
            )
    return filtered


@dataclass(slots=True)
class DataCollatorSpeechSeq2SeqWithPadding:
    """Stack fixed-length log-mel features and dynamically pad text labels."""

    processor: Any
    pad_to_multiple_of_labels: int | None = None

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        if not examples:
            raise ValueError("Không thể collate một batch rỗng")

        audio_features = [
            # Each item is already [num_mel_bins, T_frame]; the feature extractor
            # pads/truncates every example to the same fixed length.
            {"input_features": example["input_features"]} for example in examples
        ]
        # Each label item has variable shape [T_text_i].
        label_features = [{"input_ids": example["labels"]} for example in examples]

        # [num_mel_bins, T_frame] -> input_features [B, num_mel_bins, T_frame].
        batch = self.processor.feature_extractor.pad(
            audio_features,
            return_tensors="pt",
        )

        # [T_text_i] -> input_ids/attention_mask [B, T_text_max].
        label_batch = self.processor.tokenizer.pad(
            label_features,
            pad_to_multiple_of=self.pad_to_multiple_of_labels,
            return_tensors="pt",
        )

        # Padding IDs become -100 so the CrossEntropyLoss ignores those positions.
        labels = label_batch["input_ids"].masked_fill(
            label_batch["attention_mask"].ne(1), -100
        )
        # ``forward`` derives decoder_input_ids from labels by right-shifting and
        # prepending decoder_start_token_id. If the tokenizer already prefixed
        # every example with that same token (bos == decoder_start for Whisper),
        # drop it here so it is not duplicated after the shift.
        bos_token_id = self.processor.tokenizer.bos_token_id
        if bos_token_id is not None and (labels[:, 0] == bos_token_id).all():
            labels = labels[:, 1:]
        batch["labels"] = labels

        return batch
