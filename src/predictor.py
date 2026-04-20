"""통계 패턴 분석 + ML 방향 예측 모듈.

1. 과거 유사 이벤트(급등/급락) 후 N일 수익률 통계
2. LogisticRegression 기반 익일 방향 예측 (sklearn optional)
3. OutlookData로 기술적 지표 + 패턴 + 예측 통합
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import FinanceDataReader as fdr
import numpy as np
import pandas as pd

from src.detect_movers import Mover
from src.technical import (
    TechnicalIndicators, compute_technical_from_df,
    rsi_series, macd_series, bb_pct_series,
)

logger = logging.getLogger(__name__)


@dataclass
class PatternStats:
    """과거 유사 이벤트 후 N일 수익률 통계."""

    event_type: str                     # "상한가"|"급등(10%+)"|"급등(5%+)"|"급락"
    sample_count: int                   # 과거 사례 수
    avg_return_1d: float | None = None  # 익일 평균 수익률 (%)
    avg_return_5d: float | None = None  # 5일 평균 수익률 (%)
    positive_rate_1d: float | None = None  # 익일 상승 확률 (%)
    positive_rate_5d: float | None = None  # 5일 상승 확률 (%)


@dataclass
class DirectionPrediction:
    """ML 기반 방향 예측."""

    direction: str          # "상승"|"하락"|"중립"
    confidence: float       # 0.0~1.0
    features_used: list[str] = field(default_factory=list)
    model_type: str = "LogisticRegression"
    confidence_grade: str = ""   # "높음"|"보통"|"낮음"|"판단불가"
    cv_accuracy: float | None = None  # 교차검증 정확도


@dataclass
class OutlookData:
    """전망 데이터 통합."""

    technical: TechnicalIndicators | None = None
    pattern: PatternStats | None = None
    prediction: DirectionPrediction | None = None


def _classify_event(change_pct: float) -> str:
    """등락률로 이벤트 유형 분류."""
    if change_pct >= 29.0:
        return "상한가"
    elif change_pct >= 10.0:
        return "급등(10%+)"
    elif change_pct >= 5.0:
        return "급등(5%+)"
    elif change_pct <= -29.0:
        return "하한가"
    elif change_pct <= -10.0:
        return "급락(10%+)"
    elif change_pct <= -5.0:
        return "급락(5%+)"
    return "일반"


def _event_threshold(event_type: str) -> tuple[float, float]:
    """이벤트 유형에 대한 등락률 범위."""
    thresholds = {
        "상한가": (29.0, 100.0),
        "급등(10%+)": (10.0, 29.0),
        "급등(5%+)": (5.0, 10.0),
        "하한가": (-100.0, -29.0),
        "급락(10%+)": (-29.0, -10.0),
        "급락(5%+)": (-10.0, -5.0),
    }
    return thresholds.get(event_type, (0.0, 0.0))


def compute_pattern_stats(
    df: pd.DataFrame,
    change_pct: float,
    min_samples: int = 5,
) -> PatternStats | None:
    """과거 유사 이벤트 후 수익률 통계를 계산한다.

    Args:
        df: OHLCV DataFrame (Close 컬럼 필수, 최소 2년 데이터 권장)
        change_pct: 오늘의 등락률
        min_samples: 최소 사례 수 (이보다 적으면 None)

    Returns:
        PatternStats 또는 None
    """
    if df is None or df.empty or len(df) < 30:
        return None

    event_type = _classify_event(change_pct)
    if event_type == "일반":
        return None

    # 일별 수익률 계산
    returns = df["Close"].pct_change() * 100

    # 유사 이벤트 필터링
    lo, hi = _event_threshold(event_type)
    if lo < 0:
        event_mask = (returns <= lo)
    else:
        event_mask = (returns >= lo)

    event_indices = returns.index[event_mask]
    if len(event_indices) < min_samples:
        return None

    # 이벤트 후 1일/5일 수익률 수집
    closes = df["Close"]
    returns_1d = []
    returns_5d = []
    for idx in event_indices:
        pos = closes.index.get_loc(idx)
        if pos + 1 < len(closes):
            ret_1d = (closes.iloc[pos + 1] - closes.iloc[pos]) / closes.iloc[pos] * 100
            returns_1d.append(ret_1d)
        if pos + 5 < len(closes):
            ret_5d = (closes.iloc[pos + 5] - closes.iloc[pos]) / closes.iloc[pos] * 100
            returns_5d.append(ret_5d)

    sample_count = len(event_indices)

    avg_1d = round(float(np.mean(returns_1d)), 2) if returns_1d else None
    avg_5d = round(float(np.mean(returns_5d)), 2) if returns_5d else None
    pos_rate_1d = round(sum(1 for r in returns_1d if r > 0) / len(returns_1d) * 100, 1) if returns_1d else None
    pos_rate_5d = round(sum(1 for r in returns_5d if r > 0) / len(returns_5d) * 100, 1) if returns_5d else None

    return PatternStats(
        event_type=event_type,
        sample_count=sample_count,
        avg_return_1d=avg_1d,
        avg_return_5d=avg_5d,
        positive_rate_1d=pos_rate_1d,
        positive_rate_5d=pos_rate_5d,
    )


def predict_direction(
    df: pd.DataFrame,
    min_rows: int = 500,
    confidence_threshold: float = 0.65,
) -> DirectionPrediction | None:
    """ML(LogisticRegression) 기반 익일 방향 예측.

    Args:
        df: OHLCV DataFrame (Close, Volume 컬럼 필수)
        min_rows: 최소 데이터 행 수
        confidence_threshold: 이 미만이면 "중립" 반환

    Returns:
        DirectionPrediction 또는 sklearn 미설치/데이터 부족 시 None
    """
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        logger.info("sklearn not installed — skipping ML prediction")
        return None

    if df is None or df.empty or len(df) < min_rows:
        logger.info("insufficient data for ML (%d rows, need %d)",
                     len(df) if df is not None else 0, min_rows)
        return None

    close_series = df["Close"]
    vol_series = df["Volume"]

    # 피처 계산 — technical.py의 공용 시리즈 함수 재사용
    features_list = []
    labels = []
    feature_names = ["RSI", "MACD_hist", "BB_pct", "Volume_ratio", "ChangePct"]

    rsi = rsi_series(close_series)
    _, _, macd_hist = macd_series(close_series)
    bb_pct = bb_pct_series(close_series)

    vol_ma20 = vol_series.rolling(20).mean()
    vol_ratio = vol_series / vol_ma20.replace(0, np.nan)

    change_pct = close_series.pct_change() * 100

    # 라벨: 익일 수익률 > 0 → 1, else → 0
    next_return = close_series.shift(-1) / close_series - 1

    # 유효한 행만 추출 (NaN 제거)
    n_rows = len(close_series)
    start_idx = 60  # MA60 이후부터
    for i in range(start_idx, n_rows - 1):
        if any(pd.isna([rsi.iloc[i], macd_hist.iloc[i], bb_pct.iloc[i],
                        vol_ratio.iloc[i], change_pct.iloc[i]])):
            continue
        features_list.append([
            rsi.iloc[i],
            macd_hist.iloc[i],
            bb_pct.iloc[i],
            vol_ratio.iloc[i],
            change_pct.iloc[i],
        ])
        labels.append(1 if next_return.iloc[i] > 0 else 0)

    if len(features_list) < 100:
        logger.info("insufficient valid features (%d rows)", len(features_list))
        return None

    X = np.array(features_list)
    y = np.array(labels)

    # 학습 (마지막 행 제외) / 예측 (마지막 행)
    X_train, y_train = X[:-1], y[:-1]

    # 마지막 유효 데이터로 현재 예측
    last_row_idx = n_rows - 1
    current_features = []
    if not any(pd.isna([rsi.iloc[last_row_idx], macd_hist.iloc[last_row_idx],
                        bb_pct.iloc[last_row_idx], vol_ratio.iloc[last_row_idx],
                        change_pct.iloc[last_row_idx]])):
        current_features = [[
            rsi.iloc[last_row_idx],
            macd_hist.iloc[last_row_idx],
            bb_pct.iloc[last_row_idx],
            vol_ratio.iloc[last_row_idx],
            change_pct.iloc[last_row_idx],
        ]]
    else:
        # 마지막 유효 행 사용
        current_features = [X[-1]]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_pred_scaled = scaler.transform(np.array(current_features))

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)

    # 교차검증 정확도
    cv_scores = cross_val_score(
        LogisticRegression(max_iter=1000, random_state=42),
        X_train_scaled, y_train, cv=min(5, len(y_train)), scoring="accuracy",
    )
    cv_accuracy = round(float(cv_scores.mean()), 4)

    proba = model.predict_proba(X_pred_scaled)[0]
    pred_class = model.predict(X_pred_scaled)[0]

    confidence = float(max(proba))

    if confidence < confidence_threshold:
        direction = "중립"
        grade = "판단불가"
    elif confidence >= 0.75:
        grade = "높음"
    elif confidence >= 0.65:
        grade = "보통"
    else:
        grade = "낮음"

    if confidence >= confidence_threshold:
        direction = "상승" if pred_class == 1 else "하락"

    return DirectionPrediction(
        direction=direction,
        confidence=round(confidence, 4),
        features_used=feature_names,
        model_type="LogisticRegression",
        confidence_grade=grade,
        cv_accuracy=cv_accuracy,
    )


def compute_outlook(mover: Mover, days: int = 500) -> OutlookData:
    """Mover에 대한 전망 데이터를 통합 계산한다.

    1회 FDR 호출로 technical + pattern + ML 공용 데이터 사용.

    Args:
        mover: 급등/급락 종목
        days: 히스토리 조회 일수 (기본 500일 = ~2년)

    Returns:
        OutlookData (각 필드는 실패 시 None)
    """
    outlook = OutlookData()

    try:
        end = datetime.now().date()
        start = end - timedelta(days=days)
        df = fdr.DataReader(mover.code, start, end)
    except Exception as exc:
        logger.warning("outlook data fetch failed for %s: %s", mover.code, exc)
        return outlook

    if df is None or df.empty:
        logger.warning("no data for outlook: %s", mover.code)
        return outlook

    # 1) 기술적 지표 (최근 90일이면 충분하지만 전체 데이터 사용)
    outlook.technical = compute_technical_from_df(df)

    # 2) 패턴 통계
    outlook.pattern = compute_pattern_stats(df, mover.change_pct)

    # 3) ML 예측
    outlook.prediction = predict_direction(df)

    return outlook


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    code = sys.argv[1] if len(sys.argv) > 1 else "005930"
    change = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0

    m = Mover(code, code, "KOSPI", "surge", 50000, change, 1000000, int(1e10), None)
    result = compute_outlook(m)

    print(f"\n{'='*50}")
    print(f"전망 데이터 — {code} ({change:+.1f}%)")
    print(f"{'='*50}")

    if result.technical:
        t = result.technical
        print(f"\n[기술적 지표]")
        print(f"  RSI(14): {t.rsi_14}")
        print(f"  MACD: {t.macd} / Signal: {t.macd_signal} / Hist: {t.macd_histogram}")
        print(f"  BB: {t.bb_lower} ~ {t.bb_middle} ~ {t.bb_upper} ({t.bb_position})")
        print(f"  MA: {t.ma_5}/{t.ma_20}/{t.ma_60} ({t.ma_trend})")
        print(f"  Volume ratio: {t.volume_ratio}x")

    if result.pattern:
        p = result.pattern
        print(f"\n[패턴 통계] {p.event_type} (과거 {p.sample_count}건)")
        print(f"  익일: 평균 {p.avg_return_1d}%, 상승 {p.positive_rate_1d}%")
        print(f"  5일:  평균 {p.avg_return_5d}%, 상승 {p.positive_rate_5d}%")

    if result.prediction:
        pr = result.prediction
        print(f"\n[ML 예측] {pr.direction} (신뢰도 {pr.confidence:.1%})")
        print(f"  피처: {', '.join(pr.features_used)}")
    else:
        print("\n[ML 예측] 계산 불가 (데이터 부족 또는 sklearn 미설치)")
