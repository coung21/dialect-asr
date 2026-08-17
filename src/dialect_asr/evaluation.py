"""Seq2seq decoding and regional ASR metrics for ViMD."""

from __future__ import annotations

from collections.abc import Sequence
import math
import random
from typing import Any

import numpy as np
from jiwer import cer, wer
from transformers import EvalPrediction
import wandb

from dialect_asr.reproducibility import DEFAULT_SEED
from dialect_asr.text import normalize_vietnamese_text


REGION_TO_METRIC = {
    "north": "North",
    "central": "Central",
    "south": "South",
}

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


def _sample_one_per_province_table(
    predictions: Sequence[str],
    references: Sequence[str],
    provinces: Sequence[str],
    rng: random.Random,
) -> "wandb.Table":
    """Build a W&B table with one random (reference, predicted) pair per province."""
    indices_by_province: dict[str, list[int]] = {}
    for index, province in enumerate(provinces):
        indices_by_province.setdefault(province, []).append(index)

    table = wandb.Table(columns=["province", "reference", "predicted"])
    for province in sorted(indices_by_province):
        index = rng.choice(indices_by_province[province])
        table.add_data(province, references[index], predictions[index])
    return table


class Seq2SeqMetrics:
    """Callable metric adapter for Hugging Face ``Seq2SeqTrainer`` predictions."""

    def __init__(
        self,
        processor: Any,
        regions: Sequence[str],
        provinces: Sequence[str] | None = None,
        log_wandb_table: bool = False,
        sample_seed: int = DEFAULT_SEED,
    ) -> None:
        self.processor = processor
        self.set_regions(regions)
        self.provinces: tuple[str, ...] | None = None
        if provinces is not None:
            self.set_provinces(provinces)
        # Only mode=eval logs the per-province sample table; per-epoch
        # validation evaluation during training stays off by default.
        self.log_wandb_table = log_wandb_table
        # Seeded independently from the global training RNGs so it never
        # perturbs training/eval determinism, but fixed so the same province
        # always yields the same logged sample across separate eval runs.
        self._rng = random.Random(sample_seed)

    def set_regions(self, regions: Sequence[str]) -> None:
        """Switch region metadata before evaluating a different dataset split."""
        self.regions = tuple(regions)

    def set_provinces(self, provinces: Sequence[str]) -> None:
        """Switch province metadata before evaluating a different dataset split."""
        self.provinces = tuple(provinces)

    def __call__(self, prediction: EvalPrediction) -> dict[str, float]:
        # With predict_with_generate=True these are already generated token IDs,
        # not logits, so no argmax reduction is needed before decoding.
        predicted_ids = prediction.predictions
        if isinstance(predicted_ids, tuple):
            predicted_ids = predicted_ids[0]
        predicted_ids = np.asarray(predicted_ids)
        if predicted_ids.ndim != 2:
            raise ValueError("predictions phải có shape [N, T_generated]")

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

        # [N, T_generated] -> N decoded prediction strings.
        decoded_predictions = self.processor.tokenizer.batch_decode(
            predicted_ids,
            skip_special_tokens=True,
        )
        # [N, T_text] -> N reference strings.
        decoded_references = self.processor.tokenizer.batch_decode(
            label_ids,
            skip_special_tokens=True,
        )

        if len(decoded_predictions) != len(self.regions):
            raise ValueError(
                "Số prediction không khớp region metadata; "
                f"nhận {len(decoded_predictions)} và {len(self.regions)}"
            )

        if self.provinces is not None:
            if len(decoded_predictions) != len(self.provinces):
                raise ValueError(
                    "Số prediction không khớp province metadata; "
                    f"nhận {len(decoded_predictions)} và {len(self.provinces)}"
                )
            # Only log when a W&B run is actually active (e.g. wandb_mode=disabled
            # or offline evaluation without wandb.init leaves wandb.run as None)
            # and this is an explicit mode=eval run, not per-epoch validation.
            if self.log_wandb_table and wandb.run is not None:
                table = _sample_one_per_province_table(
                    decoded_predictions,
                    decoded_references,
                    self.provinces,
                    self._rng,
                )
                wandb.log({"eval/predictions_by_province": table})

        return compute_asr_metrics(
            decoded_predictions,
            decoded_references,
            self.regions,
        )


def build_compute_metrics(
    processor: Any,
    regions: Sequence[str],
    provinces: Sequence[str] | None = None,
    log_wandb_table: bool = False,
    sample_seed: int = DEFAULT_SEED,
) -> Seq2SeqMetrics:
    """Build the metric callable passed to ``create_trainer``."""
    return Seq2SeqMetrics(
        processor, regions, provinces, log_wandb_table, sample_seed
    )
