"""CTC decoding and regional ASR metrics for ViMD."""

from __future__ import annotations

from collections.abc import Sequence
import math
import re
import unicodedata
from typing import Any

import numpy as np
from jiwer import cer, wer
from transformers import EvalPrediction


REGION_TO_METRIC = {
    "north": "North",
    "central": "Central",
    "south": "South",
}


def normalize_vietnamese_text(text: str) -> str:
    """Normalize transcript text while preserving Vietnamese diacritics."""
    text = unicodedata.normalize("NFC", text).lower().strip()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def _canonical_region(region: str) -> str:
    key = region.strip().lower()
    try:
        return REGION_TO_METRIC[key]
    except KeyError as exc:
        expected = ", ".join(sorted(REGION_TO_METRIC))
        raise ValueError(f"Vùng không hợp lệ {region!r}; cần một trong: {expected}") from exc


def compute_asr_metrics(
    predictions: Sequence[str],
    references: Sequence[str],
    regions: Sequence[str],
) -> dict[str, float]:
    """Compute overall WER/CER and WER for the three ViMD regions."""
    if not (len(predictions) == len(references) == len(regions)):
        raise ValueError(
            "Số prediction, reference và region phải bằng nhau; "
            f"nhận {len(predictions)}, {len(references)}, {len(regions)}"
        )
    if not references:
        raise ValueError("Không thể tính metric trên tập dữ liệu rỗng")

    normalized_predictions = [normalize_vietnamese_text(text) for text in predictions]
    normalized_references = [normalize_vietnamese_text(text) for text in references]
    canonical_regions = [_canonical_region(region) for region in regions]

    metrics = {
        "WER": float(wer(normalized_references, normalized_predictions)),
        "CER": float(cer(normalized_references, normalized_predictions)),
    }

    for region_name in ("North", "Central", "South"):
        indices = [
            index
            for index, sample_region in enumerate(canonical_regions)
            if sample_region == region_name
        ]
        if not indices:
            metrics[f"WER_{region_name}"] = math.nan
            continue

        region_references = [normalized_references[index] for index in indices]
        region_predictions = [normalized_predictions[index] for index in indices]
        metrics[f"WER_{region_name}"] = float(
            wer(region_references, region_predictions)
        )

    return metrics


class CTCMetrics:
    """Callable metric adapter for Hugging Face Trainer predictions."""

    def __init__(self, processor: Any, regions: Sequence[str]) -> None:
        self.processor = processor
        self.set_regions(regions)

    def set_regions(self, regions: Sequence[str]) -> None:
        """Switch region metadata before evaluating a different dataset split."""
        self.regions = tuple(regions)

    def __call__(self, prediction: EvalPrediction) -> dict[str, float]:
        predicted_ids = prediction.predictions
        if isinstance(predicted_ids, tuple):
            predicted_ids = predicted_ids[0]
        predicted_ids = np.asarray(predicted_ids)
        if predicted_ids.ndim == 3:
            # [N, T_frame, V] -> [N, T_frame] for callers without preprocessing.
            predicted_ids = np.argmax(predicted_ids, axis=-1)
        if predicted_ids.ndim != 2:
            raise ValueError(
                "predictions phải có shape [N, T_frame] hoặc [N, T_frame, V]"
            )

        label_ids = prediction.label_ids
        if isinstance(label_ids, tuple):
            label_ids = label_ids[0]
        label_ids = np.asarray(label_ids).copy()  # [N, T_text].
        if label_ids.ndim != 2:
            raise ValueError("label_ids phải có shape [N, T_text]")

        pad_token_id = self.processor.tokenizer.pad_token_id
        if pad_token_id is None:
            raise ValueError("Processor tokenizer chưa định nghĩa pad_token_id")
        # [N, T_text] mask replacement keeps shape [N, T_text].
        label_ids[label_ids == -100] = pad_token_id

        # [N, T_frame] -> N decoded prediction strings; CTC repeats are grouped.
        decoded_predictions = self.processor.batch_decode(predicted_ids)
        # [N, T_text] -> N reference strings; target repeats must be preserved.
        decoded_references = self.processor.batch_decode(
            label_ids,
            group_tokens=False,
        )

        if len(decoded_predictions) != len(self.regions):
            raise ValueError(
                "Số prediction không khớp region metadata; "
                f"nhận {len(decoded_predictions)} và {len(self.regions)}"
            )
        return compute_asr_metrics(
            decoded_predictions,
            decoded_references,
            self.regions,
        )


def build_compute_metrics(processor: Any, regions: Sequence[str]) -> CTCMetrics:
    """Build the metric callable passed to ``create_trainer``."""
    return CTCMetrics(processor, regions)
