"""시그널 적중률 트래킹 모듈.

매일의 예측 → 다음날 실제 등락 비교 → 적중률 통계.
순수 파일 I/O로 동작하며 외부 의존성 없음.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_PREDICTIONS_FILE = "predictions.json"
_ACCURACY_LOG_FILE = "accuracy_log.json"


@dataclass
class DailyPrediction:
    """단일 종목 일일 예측."""
    date: str                  # 예측일 YYYY-MM-DD
    code: str
    name: str
    direction: str             # "상승" | "하락" | "횡보"
    confidence: float          # 0.0 ~ 1.0
    signal_summary: str = ""

    def to_dict(self) -> dict:
        return {
            "date": self.date,
            "code": self.code,
            "name": self.name,
            "direction": self.direction,
            "confidence": self.confidence,
            "signal_summary": self.signal_summary,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DailyPrediction":
        return cls(
            date=d["date"],
            code=d["code"],
            name=d["name"],
            direction=d["direction"],
            confidence=d["confidence"],
            signal_summary=d.get("signal_summary", ""),
        )


@dataclass
class AccuracyRecord:
    """평가 완료된 예측 기록."""
    prediction: DailyPrediction
    actual_change_pct: float | None = None
    correct: bool | None = None
    evaluated_date: str | None = None

    def to_dict(self) -> dict:
        return {
            "prediction": self.prediction.to_dict(),
            "actual_change_pct": self.actual_change_pct,
            "correct": self.correct,
            "evaluated_date": self.evaluated_date,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AccuracyRecord":
        return cls(
            prediction=DailyPrediction.from_dict(d["prediction"]),
            actual_change_pct=d.get("actual_change_pct"),
            correct=d.get("correct"),
            evaluated_date=d.get("evaluated_date"),
        )


@dataclass
class AccuracyStats:
    """적중률 통계."""
    total: int = 0
    correct: int = 0
    accuracy_pct: float = 0.0
    by_direction: dict[str, dict] = field(default_factory=dict)
    streak: int = 0               # +적중/-실패 연속
    windows: dict[str, float] = field(default_factory=dict)  # 7/30/90일 적중률

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "correct": self.correct,
            "accuracy_pct": self.accuracy_pct,
            "by_direction": self.by_direction,
            "streak": self.streak,
            "windows": self.windows,
        }

    def to_narrative(self) -> str:
        if self.total == 0:
            return "시그널 적중률 데이터 없음"

        parts = [f"전체 적중률 {self.accuracy_pct:.1f}% ({self.correct}/{self.total})"]

        if self.streak > 0:
            parts.append(f"{self.streak}연속 적중 중")
        elif self.streak < 0:
            parts.append(f"{abs(self.streak)}연속 실패 중")

        for window, pct in self.windows.items():
            parts.append(f"{window}: {pct:.1f}%")

        return ". ".join(parts)


# ── 예측 저장/로드 ───────────────────────────────────────────────────

def save_predictions(
    preds: list[DailyPrediction],
    output_dir: Path,
) -> None:
    """예측을 파일에 저장한다."""
    if not preds:
        return

    path = output_dir / _PREDICTIONS_FILE
    existing: list[dict] = []
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            existing = []

    # 같은 날짜+종목 중복 제거
    existing_keys = {(d["date"], d["code"]) for d in existing}
    for p in preds:
        key = (p.date, p.code)
        if key not in existing_keys:
            existing.append(p.to_dict())
            existing_keys.add(key)

    output_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("saved %d predictions (total %d)", len(preds), len(existing))


def _load_predictions(output_dir: Path) -> list[DailyPrediction]:
    """저장된 예측을 로드한다."""
    path = output_dir / _PREDICTIONS_FILE
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return [DailyPrediction.from_dict(d) for d in data]
    except Exception as exc:
        logger.warning("predictions load failed: %s", exc)
        return []


def _load_accuracy_log(output_dir: Path) -> list[AccuracyRecord]:
    """평가 로그를 로드한다."""
    path = output_dir / _ACCURACY_LOG_FILE
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return [AccuracyRecord.from_dict(d) for d in data]
    except Exception as exc:
        logger.warning("accuracy log load failed: %s", exc)
        return []


def _save_accuracy_log(records: list[AccuracyRecord], output_dir: Path) -> None:
    """평가 로그를 저장한다."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / _ACCURACY_LOG_FILE
    data = [r.to_dict() for r in records]
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# ── 예측 평가 ─────────────────────────────────────────────────────────

def _is_correct(direction: str, actual_pct: float) -> bool:
    """예측 방향과 실제 등락률 비교."""
    if direction == "상승":
        return actual_pct > 0
    elif direction == "하락":
        return actual_pct < 0
    else:  # 횡보
        return abs(actual_pct) < 1.0


def evaluate_predictions(
    output_dir: Path,
    trade_date: str,
    actual_changes: dict[str, float],
) -> int:
    """전일 예측을 실제 등락과 비교하여 평가한다.

    Args:
        output_dir: 파일 경로
        trade_date: 평가일 YYYY-MM-DD
        actual_changes: {종목코드: 등락률%}

    Returns:
        평가된 예측 수
    """
    preds = _load_predictions(output_dir)
    log = _load_accuracy_log(output_dir)

    # 이미 평가된 예측 키
    evaluated_keys = {(r.prediction.date, r.prediction.code) for r in log}

    count = 0
    for pred in preds:
        key = (pred.date, pred.code)
        if key in evaluated_keys:
            continue

        actual = actual_changes.get(pred.code)
        if actual is None:
            continue

        correct = _is_correct(pred.direction, actual)
        record = AccuracyRecord(
            prediction=pred,
            actual_change_pct=actual,
            correct=correct,
            evaluated_date=trade_date,
        )
        log.append(record)
        evaluated_keys.add(key)
        count += 1

    if count > 0:
        _save_accuracy_log(log, output_dir)
        logger.info("evaluated %d predictions on %s", count, trade_date)

    return count


# ── 통계 계산 ─────────────────────────────────────────────────────────

def _count_streak(records: list[AccuracyRecord]) -> int:
    """최근부터 연속 적중/실패 수. +적중/-실패."""
    if not records:
        return 0

    # 날짜순 정렬
    sorted_records = sorted(records, key=lambda r: r.evaluated_date or "")
    latest = sorted_records[-1]
    if latest.correct is None:
        return 0

    is_correct = latest.correct
    count = 0
    for r in reversed(sorted_records):
        if r.correct == is_correct:
            count += 1
        else:
            break

    return count if is_correct else -count


def _window_accuracy(records: list[AccuracyRecord], days: int) -> float | None:
    """최근 N일 적중률."""
    if not records:
        return None

    # 평가일 기준으로 최근 N일 필터
    sorted_records = sorted(records, key=lambda r: r.evaluated_date or "")
    if not sorted_records:
        return None

    # 최근 N일 간의 레코드
    recent = sorted_records[-days:] if len(sorted_records) >= days else sorted_records
    evaluated = [r for r in recent if r.correct is not None]
    if not evaluated:
        return None

    correct_count = sum(1 for r in evaluated if r.correct)
    return round(correct_count / len(evaluated) * 100, 1)


def compute_accuracy_stats(output_dir: Path) -> AccuracyStats:
    """적중률 통계를 계산한다."""
    records = _load_accuracy_log(output_dir)
    evaluated = [r for r in records if r.correct is not None]

    if not evaluated:
        return AccuracyStats()

    total = len(evaluated)
    correct = sum(1 for r in evaluated if r.correct)
    accuracy_pct = round(correct / total * 100, 1)

    # 방향별 통계
    by_direction: dict[str, dict] = {}
    for direction in ("상승", "하락", "횡보"):
        dir_records = [r for r in evaluated if r.prediction.direction == direction]
        if dir_records:
            dir_correct = sum(1 for r in dir_records if r.correct)
            by_direction[direction] = {
                "total": len(dir_records),
                "correct": dir_correct,
                "accuracy_pct": round(dir_correct / len(dir_records) * 100, 1),
            }

    # 연속 기록
    streak = _count_streak(evaluated)

    # 윈도우별 적중률
    windows: dict[str, float] = {}
    for label, days in [("7일", 7), ("30일", 30), ("90일", 90)]:
        w_acc = _window_accuracy(evaluated, days)
        if w_acc is not None:
            windows[label] = w_acc

    return AccuracyStats(
        total=total,
        correct=correct,
        accuracy_pct=accuracy_pct,
        by_direction=by_direction,
        streak=streak,
        windows=windows,
    )
