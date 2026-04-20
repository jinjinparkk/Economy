"""accuracy 모듈 단위 테스트."""
from __future__ import annotations

import json

import pytest

from src.accuracy import (
    DailyPrediction,
    AccuracyRecord,
    AccuracyStats,
    save_predictions,
    _load_predictions,
    _load_accuracy_log,
    _save_accuracy_log,
    evaluate_predictions,
    compute_accuracy_stats,
    _is_correct,
    _count_streak,
    _window_accuracy,
)


# ── 헬퍼 ──────────────────────────────────────────────────────────────

def _pred(code="005930", name="삼성전자", direction="상승",
          confidence=0.7, date="2026-04-19"):
    return DailyPrediction(date, code, name, direction, confidence, "RSI 상승 시그널")


def _record(pred=None, actual=2.5, correct=True, eval_date="2026-04-20"):
    if pred is None:
        pred = _pred()
    return AccuracyRecord(pred, actual, correct, eval_date)


# ── DailyPrediction ──────────────────────────────────────────────────

class TestDailyPrediction:
    def test_to_dict(self):
        p = _pred()
        d = p.to_dict()
        assert d["code"] == "005930"
        assert d["direction"] == "상승"
        assert d["confidence"] == 0.7

    def test_from_dict(self):
        d = {
            "date": "2026-04-19",
            "code": "005930",
            "name": "삼성전자",
            "direction": "상승",
            "confidence": 0.7,
            "signal_summary": "test",
        }
        p = DailyPrediction.from_dict(d)
        assert p.code == "005930"
        assert p.signal_summary == "test"

    def test_from_dict_missing_summary(self):
        d = {"date": "2026-04-19", "code": "005930", "name": "삼성전자",
             "direction": "하락", "confidence": 0.5}
        p = DailyPrediction.from_dict(d)
        assert p.signal_summary == ""


# ── AccuracyRecord ───────────────────────────────────────────────────

class TestAccuracyRecord:
    def test_to_dict(self):
        r = _record()
        d = r.to_dict()
        assert d["actual_change_pct"] == 2.5
        assert d["correct"] is True
        assert "prediction" in d

    def test_from_dict(self):
        d = {
            "prediction": _pred().to_dict(),
            "actual_change_pct": -1.5,
            "correct": False,
            "evaluated_date": "2026-04-20",
        }
        r = AccuracyRecord.from_dict(d)
        assert r.actual_change_pct == -1.5
        assert r.correct is False
        assert r.prediction.name == "삼성전자"


# ── AccuracyStats ────────────────────────────────────────────────────

class TestAccuracyStats:
    def test_to_dict(self):
        s = AccuracyStats(
            total=10, correct=7, accuracy_pct=70.0,
            by_direction={"상승": {"total": 5, "correct": 4, "accuracy_pct": 80.0}},
            streak=3,
            windows={"7일": 80.0, "30일": 70.0},
        )
        d = s.to_dict()
        assert d["total"] == 10
        assert d["accuracy_pct"] == 70.0
        assert d["streak"] == 3
        assert "7일" in d["windows"]

    def test_to_narrative_with_data(self):
        s = AccuracyStats(
            total=10, correct=7, accuracy_pct=70.0,
            streak=3,
            windows={"7일": 80.0},
        )
        narrative = s.to_narrative()
        assert "70.0%" in narrative
        assert "3연속 적중" in narrative
        assert "7일" in narrative

    def test_to_narrative_negative_streak(self):
        s = AccuracyStats(total=5, correct=2, accuracy_pct=40.0, streak=-2)
        narrative = s.to_narrative()
        assert "2연속 실패" in narrative

    def test_to_narrative_empty(self):
        s = AccuracyStats()
        narrative = s.to_narrative()
        assert "없음" in narrative


# ── _is_correct ──────────────────────────────────────────────────────

class TestIsCorrect:
    def test_up_correct(self):
        assert _is_correct("상승", 2.5) is True

    def test_up_wrong(self):
        assert _is_correct("상승", -1.0) is False

    def test_down_correct(self):
        assert _is_correct("하락", -2.0) is True

    def test_down_wrong(self):
        assert _is_correct("하락", 1.0) is False

    def test_sideways_correct(self):
        assert _is_correct("횡보", 0.5) is True

    def test_sideways_wrong(self):
        assert _is_correct("횡보", 2.0) is False

    def test_up_zero(self):
        assert _is_correct("상승", 0.0) is False

    def test_down_zero(self):
        assert _is_correct("하락", 0.0) is False

    def test_sideways_boundary(self):
        assert _is_correct("횡보", 0.99) is True
        assert _is_correct("횡보", 1.0) is False


# ── 저장/로드 ─────────────────────────────────────────────────────────

class TestSaveLoad:
    def test_save_and_load_predictions(self, tmp_path):
        preds = [_pred(), _pred(code="000660", name="SK하이닉스")]
        save_predictions(preds, tmp_path)
        loaded = _load_predictions(tmp_path)
        assert len(loaded) == 2
        assert loaded[0].code == "005930"
        assert loaded[1].code == "000660"

    def test_save_deduplicates(self, tmp_path):
        preds = [_pred()]
        save_predictions(preds, tmp_path)
        save_predictions(preds, tmp_path)  # 같은 것 다시
        loaded = _load_predictions(tmp_path)
        assert len(loaded) == 1

    def test_load_empty_dir(self, tmp_path):
        loaded = _load_predictions(tmp_path)
        assert loaded == []

    def test_load_corrupted(self, tmp_path):
        (tmp_path / "predictions.json").write_text("not json", encoding="utf-8")
        loaded = _load_predictions(tmp_path)
        assert loaded == []

    def test_save_and_load_accuracy_log(self, tmp_path):
        records = [_record(), _record(pred=_pred(code="000660"))]
        _save_accuracy_log(records, tmp_path)
        loaded = _load_accuracy_log(tmp_path)
        assert len(loaded) == 2

    def test_load_accuracy_empty(self, tmp_path):
        loaded = _load_accuracy_log(tmp_path)
        assert loaded == []


# ── evaluate_predictions ─────────────────────────────────────────────

class TestEvaluatePredictions:
    def test_basic_evaluation(self, tmp_path):
        preds = [
            _pred(code="005930", direction="상승"),
            _pred(code="000660", direction="하락"),
        ]
        save_predictions(preds, tmp_path)

        actual = {"005930": 2.5, "000660": -1.5}
        count = evaluate_predictions(tmp_path, "2026-04-20", actual)
        assert count == 2

        log = _load_accuracy_log(tmp_path)
        assert len(log) == 2
        samsung = [r for r in log if r.prediction.code == "005930"][0]
        assert samsung.correct is True
        sk = [r for r in log if r.prediction.code == "000660"][0]
        assert sk.correct is True

    def test_no_actual_data(self, tmp_path):
        preds = [_pred()]
        save_predictions(preds, tmp_path)
        count = evaluate_predictions(tmp_path, "2026-04-20", {})
        assert count == 0

    def test_no_predictions(self, tmp_path):
        count = evaluate_predictions(tmp_path, "2026-04-20", {"005930": 1.0})
        assert count == 0

    def test_no_double_evaluation(self, tmp_path):
        preds = [_pred()]
        save_predictions(preds, tmp_path)
        actual = {"005930": 2.5}
        evaluate_predictions(tmp_path, "2026-04-20", actual)
        count = evaluate_predictions(tmp_path, "2026-04-21", actual)
        assert count == 0  # 이미 평가됨

    def test_wrong_prediction(self, tmp_path):
        preds = [_pred(direction="상승")]
        save_predictions(preds, tmp_path)
        actual = {"005930": -3.0}
        evaluate_predictions(tmp_path, "2026-04-20", actual)
        log = _load_accuracy_log(tmp_path)
        assert log[0].correct is False


# ── _count_streak ────────────────────────────────────────────────────

class TestCountStreak:
    def test_empty(self):
        assert _count_streak([]) == 0

    def test_all_correct(self):
        records = [
            _record(eval_date=f"2026-04-{i:02d}")
            for i in range(1, 4)
        ]
        assert _count_streak(records) == 3

    def test_all_wrong(self):
        records = [
            _record(correct=False, eval_date=f"2026-04-{i:02d}")
            for i in range(1, 4)
        ]
        assert _count_streak(records) == -3

    def test_mixed(self):
        records = [
            _record(correct=True, eval_date="2026-04-01"),
            _record(correct=False, eval_date="2026-04-02"),
            _record(correct=True, eval_date="2026-04-03"),
            _record(correct=True, eval_date="2026-04-04"),
        ]
        assert _count_streak(records) == 2

    def test_latest_none(self):
        records = [_record(correct=None, eval_date="2026-04-01")]
        assert _count_streak(records) == 0


# ── _window_accuracy ─────────────────────────────────────────────────

class TestWindowAccuracy:
    def test_basic(self):
        records = [
            _record(correct=True, eval_date=f"2026-04-{i:02d}")
            for i in range(1, 11)
        ]
        assert _window_accuracy(records, 10) == 100.0

    def test_partial(self):
        records = [
            _record(correct=True, eval_date="2026-04-01"),
            _record(correct=False, eval_date="2026-04-02"),
        ]
        assert _window_accuracy(records, 7) == 50.0

    def test_empty(self):
        assert _window_accuracy([], 7) is None

    def test_window_larger_than_data(self):
        records = [_record(correct=True, eval_date="2026-04-01")]
        result = _window_accuracy(records, 30)
        assert result == 100.0


# ── compute_accuracy_stats ───────────────────────────────────────────

class TestComputeAccuracyStats:
    def test_empty(self, tmp_path):
        stats = compute_accuracy_stats(tmp_path)
        assert stats.total == 0
        assert stats.accuracy_pct == 0.0

    def test_basic_stats(self, tmp_path):
        records = [
            _record(correct=True, eval_date="2026-04-01"),
            _record(correct=True, eval_date="2026-04-02"),
            _record(correct=False, eval_date="2026-04-03"),
            _record(
                pred=_pred(direction="하락", code="000660"),
                actual=-2.0, correct=True, eval_date="2026-04-04",
            ),
        ]
        _save_accuracy_log(records, tmp_path)

        stats = compute_accuracy_stats(tmp_path)
        assert stats.total == 4
        assert stats.correct == 3
        assert stats.accuracy_pct == 75.0

    def test_by_direction(self, tmp_path):
        records = [
            _record(correct=True, eval_date="2026-04-01"),  # 상승 맞음
            _record(correct=False, eval_date="2026-04-02"),  # 상승 틀림
            _record(
                pred=_pred(direction="하락", code="000660"),
                actual=-2.0, correct=True, eval_date="2026-04-03",
            ),
        ]
        _save_accuracy_log(records, tmp_path)

        stats = compute_accuracy_stats(tmp_path)
        assert "상승" in stats.by_direction
        assert stats.by_direction["상승"]["total"] == 2
        assert stats.by_direction["상승"]["correct"] == 1
        assert "하락" in stats.by_direction
        assert stats.by_direction["하락"]["accuracy_pct"] == 100.0

    def test_streak(self, tmp_path):
        records = [
            _record(correct=False, eval_date="2026-04-01"),
            _record(correct=True, eval_date="2026-04-02"),
            _record(correct=True, eval_date="2026-04-03"),
            _record(correct=True, eval_date="2026-04-04"),
        ]
        _save_accuracy_log(records, tmp_path)

        stats = compute_accuracy_stats(tmp_path)
        assert stats.streak == 3

    def test_windows(self, tmp_path):
        records = [
            _record(correct=True, eval_date=f"2026-04-{i:02d}")
            for i in range(1, 11)
        ]
        _save_accuracy_log(records, tmp_path)

        stats = compute_accuracy_stats(tmp_path)
        assert "7일" in stats.windows
        assert stats.windows["7일"] == 100.0
