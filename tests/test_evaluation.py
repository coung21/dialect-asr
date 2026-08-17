from types import SimpleNamespace

import numpy as np
import pytest
from transformers import EvalPrediction

import wandb
from dialect_asr.evaluation import Seq2SeqMetrics, compute_asr_metrics
from dialect_asr.text import normalize_vietnamese_text


class FakeProcessor:
    def __init__(self, decoded_batches: list[list[str]]) -> None:
        self.tokenizer = SimpleNamespace(
            pad_token_id=0,
            batch_decode=self._batch_decode,
        )
        self.decoded_batches = iter(decoded_batches)
        self.decode_calls: list[tuple[np.ndarray, bool]] = []

    def _batch_decode(self, token_ids, skip_special_tokens=False):
        # token_ids: predictions [N, T_generated] or references [N, T_text].
        self.decode_calls.append((np.asarray(token_ids).copy(), skip_special_tokens))
        return next(self.decoded_batches)


def test_normalize_vietnamese_text() -> None:
    assert normalize_vietnamese_text("  XIN,   Chào!  ") == "xin chào"


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


def test_seq2seq_metrics_decodes_predictions_and_references() -> None:
    processor = FakeProcessor(
        decoded_batches=[
            ["xin chào", "sài"],
            ["xin chào", "sài gòn"],
        ]
    )
    metric = Seq2SeqMetrics(processor, regions=["North", "South"])
    prediction = EvalPrediction(
        # generated IDs: [N=2, T_generated=3].
        predictions=np.array([[1, 1, 0], [2, 0, 0]]),
        # label IDs: [N=2, T_text=3], with -100 padding.
        label_ids=np.array([[1, 2, -100], [3, 4, 5]]),
    )

    metrics = metric(prediction)

    assert metrics["WER"] == pytest.approx(0.25)
    assert processor.decode_calls[0][1] is True
    assert processor.decode_calls[1][1] is True
    assert processor.decode_calls[1][0].tolist() == [[1, 2, 0], [3, 4, 5]]


def test_seq2seq_metrics_rejects_logits_shape() -> None:
    processor = FakeProcessor(decoded_batches=[["đúng"]])
    metric = Seq2SeqMetrics(processor, regions=["Central"])
    # predict_with_generate returns token IDs, never [N, T, V] logits.
    logits = np.array([[[0.0, 2.0, 1.0], [3.0, 1.0, 0.0]]])

    with pytest.raises(ValueError, match=r"\[N, T_generated\]"):
        metric(
            EvalPrediction(
                predictions=logits,
                label_ids=np.array([[1, 0]]),
            )
        )


def test_no_wandb_table_logged_when_disabled_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Per-epoch validation during training must not log the table."""
    processor = FakeProcessor(
        decoded_batches=[["xin chào", "sài"], ["xin chào", "sài gòn"]]
    )
    metric = Seq2SeqMetrics(
        processor,
        regions=["North", "South"],
        provinces=["Hanoi", "Saigon"],
    )
    logged = {}
    monkeypatch.setattr(wandb, "run", object())
    monkeypatch.setattr(wandb, "log", lambda payload: logged.update(payload))

    metric(
        EvalPrediction(
            predictions=np.array([[1, 1, 0], [2, 0, 0]]),
            label_ids=np.array([[1, 2, -100], [3, 4, 5]]),
        )
    )

    assert logged == {}
    assert metric.log_wandb_table is False


def test_no_wandb_table_logged_without_active_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processor = FakeProcessor(
        decoded_batches=[["xin chào", "sài"], ["xin chào", "sài gòn"]]
    )
    metric = Seq2SeqMetrics(
        processor,
        regions=["North", "South"],
        provinces=["Hanoi", "Saigon"],
        log_wandb_table=True,
    )
    logged = {}
    monkeypatch.setattr(wandb, "run", None)
    monkeypatch.setattr(wandb, "log", lambda payload: logged.update(payload))

    metric(
        EvalPrediction(
            predictions=np.array([[1, 1, 0], [2, 0, 0]]),
            label_ids=np.array([[1, 2, -100], [3, 4, 5]]),
        )
    )

    assert logged == {}


def test_wandb_table_logs_one_random_sample_per_province(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processor = FakeProcessor(
        decoded_batches=[
            ["xin chào", "sài", "hà nội hai"],
            ["xin chào", "sài gòn", "hà nội"],
        ]
    )
    metric = Seq2SeqMetrics(
        processor,
        regions=["North", "South", "North"],
        provinces=["Hanoi", "Saigon", "Hanoi"],
        log_wandb_table=True,
    )
    logged = {}
    monkeypatch.setattr(wandb, "run", object())
    monkeypatch.setattr(wandb, "log", lambda payload: logged.update(payload))

    metric(
        EvalPrediction(
            predictions=np.array([[1, 1, 0], [2, 0, 0], [3, 0, 0]]),
            label_ids=np.array([[1, 2, -100], [3, 4, 5], [6, 7, -100]]),
        )
    )

    table = logged["eval/predictions_by_province"]
    assert table.columns == ["province", "reference", "predicted"]
    rows = {row[0]: tuple(row[1:]) for row in table.data}
    assert set(rows) == {"Hanoi", "Saigon"}
    assert rows["Saigon"] == ("sài gòn", "sài")
    assert rows["Hanoi"] in {("xin chào", "xin chào"), ("hà nội", "hà nội hai")}


def test_metric_rejects_province_length_mismatch() -> None:
    processor = FakeProcessor(decoded_batches=[["a"], ["a"]])
    metric = Seq2SeqMetrics(
        processor,
        regions=["North"],
        provinces=["Hanoi", "Saigon"],
    )

    with pytest.raises(ValueError, match="province"):
        metric(
            EvalPrediction(
                predictions=np.array([[1, 0]]),  # [N=1, T_generated=2].
                label_ids=np.array([[1, -100]]),  # [N=1, T_text=2].
            )
        )


def test_metric_rejects_region_length_mismatch() -> None:
    processor = FakeProcessor(decoded_batches=[["a"], ["a"]])
    metric = Seq2SeqMetrics(processor, regions=["North", "South"])

    with pytest.raises(ValueError, match="không khớp"):
        metric(
            EvalPrediction(
                predictions=np.array([[1, 0]]),  # [N=1, T_generated=2].
                label_ids=np.array([[1, -100]]),  # [N=1, T_text=2].
            )
        )
