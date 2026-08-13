from types import SimpleNamespace

import numpy as np
import pytest
from transformers import EvalPrediction

from dialect_asr.evaluation import CTCMetrics, DANNMetrics, compute_asr_metrics
from dialect_asr.text import normalize_vietnamese_text


class FakeProcessor:
    def __init__(self, decoded_batches: list[list[str]]) -> None:
        self.tokenizer = SimpleNamespace(pad_token_id=0)
        self.decoded_batches = iter(decoded_batches)
        self.decode_calls: list[tuple[np.ndarray, bool]] = []

    def batch_decode(self, token_ids, group_tokens=True):
        # token_ids: predictions [N, T_frame] or references [N, T_text].
        self.decode_calls.append((np.asarray(token_ids).copy(), group_tokens))
        return next(self.decoded_batches)


def test_normalize_vietnamese_text() -> None:
    assert normalize_vietnamese_text("  XIN,   Chào!  ") == "xin chào"


def test_compute_overall_and_regional_metrics() -> None:
    metrics = compute_asr_metrics(
        predictions=["xin sao", "miền trung", "sài", "hà nội"],
        references=["xin chào", "miền trung", "sài gòn", "hà nội"],
        regions=["North", "Central", "South", "North"],
    )

    assert metrics["WER"] == pytest.approx(0.25)
    assert metrics["WER_North"] == pytest.approx(0.25)
    assert metrics["WER_Central"] == pytest.approx(0.0)
    assert metrics["WER_South"] == pytest.approx(0.5)
    assert 0.0 < metrics["CER"] < 1.0


def test_missing_region_returns_nan() -> None:
    metrics = compute_asr_metrics(
        predictions=["xin chào"],
        references=["xin chào"],
        regions=["North"],
    )

    assert metrics["WER_North"] == 0
    assert np.isnan(metrics["WER_Central"])
    assert np.isnan(metrics["WER_South"])


def test_ctc_metrics_decodes_predictions_and_references() -> None:
    processor = FakeProcessor(
        decoded_batches=[
            ["xin chào", "sài"],
            ["xin chào", "sài gòn"],
        ]
    )
    metric = CTCMetrics(processor, regions=["North", "South"])
    prediction = EvalPrediction(
        # predicted IDs: [N=2, T_frame=3].
        predictions=np.array([[1, 1, 0], [2, 0, 0]]),
        # label IDs: [N=2, T_text=3], with -100 padding.
        label_ids=np.array([[1, 2, -100], [3, 4, 5]]),
    )

    metrics = metric(prediction)

    assert metrics["WER"] == pytest.approx(0.25)
    assert processor.decode_calls[0][1] is True
    assert processor.decode_calls[1][1] is False
    assert processor.decode_calls[1][0].tolist() == [[1, 2, 0], [3, 4, 5]]


def test_ctc_metrics_accepts_logits() -> None:
    processor = FakeProcessor(decoded_batches=[["đúng"], ["đúng"]])
    metric = CTCMetrics(processor, regions=["Central"])
    # logits: [N=1, T_frame=2, V=3].
    logits = np.array([[[0.0, 2.0, 1.0], [3.0, 1.0, 0.0]]])

    metrics = metric(
        EvalPrediction(
            predictions=logits,
            label_ids=np.array([[1, 0]]),  # [N=1, T_text=2].
        )
    )

    assert metrics["WER"] == 0
    assert processor.decode_calls[0][0].shape == (1, 2)  # [N=1, T_frame=2].


def test_metric_rejects_region_length_mismatch() -> None:
    processor = FakeProcessor(decoded_batches=[["a"], ["a"]])
    metric = CTCMetrics(processor, regions=["North", "South"])

    with pytest.raises(ValueError, match="không khớp"):
        metric(
            EvalPrediction(
                predictions=np.array([[1, 0]]),  # [N=1, T_frame=2].
                label_ids=np.array([[1, -100]]),  # [N=1, T_text=2].
            )
        )


def test_dann_metrics_compute_classification_quality_and_losses() -> None:
    processor = FakeProcessor(
        decoded_batches=[
            ["a", "b", "c"],
            ["a", "b", "c"],
        ]
    )
    metric = DANNMetrics(
        processor,
        regions=["North", "Central", "South"],
    )
    ctc_ids = np.array([[1, 0], [2, 0], [3, 0]])
    # Greedy CTC IDs [N=3, T_frame=2].
    dialect_logits = np.array(
        [[4.0, 0.0, 0.0], [0.0, 0.0, 4.0], [0.0, 0.0, 4.0]]
    )
    # Dialect logits [N=3, R=3] -> predicted region IDs [0, 2, 2].
    ctc_losses = np.array([1.0, 1.2, 1.4])  # Per-sample values [N=3].
    dialect_losses = np.array([0.2, 0.4, 0.6])  # Per-sample values [N=3].
    ctc_labels = np.array([[1, -100], [2, -100], [3, -100]])
    # CTC labels [N=3, T_text=2].
    region_labels = np.array([0, 1, 2])  # Region IDs [N=3].

    metrics = metric(
        EvalPrediction(
            predictions=(
                ctc_ids,
                dialect_logits,
                ctc_losses,
                dialect_losses,
            ),
            label_ids=(ctc_labels, region_labels),
        )
    )

    assert metrics["WER"] == 0.0
    assert metrics["CER"] == 0.0
    assert metrics["ctc_loss"] == pytest.approx(1.2)
    assert metrics["dialect_loss"] == pytest.approx(0.4)
    assert metrics["dialect_accuracy"] == pytest.approx(2 / 3)
    assert metrics["dialect_macro_f1"] == pytest.approx((1.0 + 0.0 + 2 / 3) / 3)
