"""기술적 지표 계산 모듈.

OHLCV 히스토리(FinanceDataReader)로 순수 pandas/numpy 계산:
- RSI(14): Wilder 방식 (EMA 기반)
- MACD: EMA(12) - EMA(26), 시그널 EMA(9)
- 볼린저 밴드: MA(20) +- 2*sigma
- 이동평균: 5/20/60일 SMA
- 거래량 비율: 당일 / MA(20, Volume)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import FinanceDataReader as fdr
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TechnicalIndicators:
    """기술적 지표 결과."""

    rsi_14: float | None = None
    macd: float | None = None
    macd_signal: float | None = None
    macd_histogram: float | None = None
    bb_upper: float | None = None
    bb_middle: float | None = None
    bb_lower: float | None = None
    bb_position: str | None = None      # "상단돌파"|"상단근접"|"중립"|"하단근접"|"하단이탈"
    ma_5: float | None = None
    ma_20: float | None = None
    ma_60: float | None = None
    ma_trend: str | None = None         # "정배열"|"역배열"|"혼조"
    volume_ratio: float | None = None   # 당일거래량 / 20일평균거래량
    obv_trend: str | None = None        # "상승"|"하락"|"횡보"
    rsi_divergence: str | None = None   # "상승다이버전스"|"하락다이버전스"|None
    signal_summary: str | None = None   # 기술적 지표 종합 해석


def _calc_rsi(closes: pd.Series, period: int = 14) -> float | None:
    """Wilder RSI (EMA 기반)."""
    if len(closes) < period + 1:
        return None
    delta = closes.diff()
    gain = delta.clip(lower=0)
    loss = (-delta.clip(upper=0))

    # Wilder EMA: alpha = 1/period
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    last_gain = avg_gain.iloc[-1]
    last_loss = avg_loss.iloc[-1]
    if last_loss == 0:
        return 100.0
    rs = last_gain / last_loss
    return round(100 - (100 / (1 + rs)), 2)


def _calc_macd(
    closes: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[float | None, float | None, float | None]:
    """MACD 라인, 시그널, 히스토그램."""
    if len(closes) < slow + signal:
        return None, None, None
    ema_fast = closes.ewm(span=fast, adjust=False).mean()
    ema_slow = closes.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return (
        round(float(macd_line.iloc[-1]), 4),
        round(float(signal_line.iloc[-1]), 4),
        round(float(histogram.iloc[-1]), 4),
    )


def _calc_bollinger(
    closes: pd.Series, period: int = 20, num_std: float = 2.0,
) -> tuple[float | None, float | None, float | None, str | None]:
    """볼린저 밴드 (upper, middle, lower, position)."""
    if len(closes) < period:
        return None, None, None, None
    ma = closes.rolling(period).mean()
    std = closes.rolling(period).std(ddof=0)
    upper = ma + num_std * std
    lower = ma - num_std * std

    bb_upper = round(float(upper.iloc[-1]), 2)
    bb_middle = round(float(ma.iloc[-1]), 2)
    bb_lower = round(float(lower.iloc[-1]), 2)
    last_close = float(closes.iloc[-1])

    if last_close > bb_upper:
        pos = "상단돌파"
    elif last_close > bb_middle + (bb_upper - bb_middle) * 0.8:
        pos = "상단근접"
    elif last_close < bb_lower:
        pos = "하단이탈"
    elif last_close < bb_middle - (bb_middle - bb_lower) * 0.8:
        pos = "하단근접"
    else:
        pos = "중립"

    return bb_upper, bb_middle, bb_lower, pos


def _calc_ma(
    closes: pd.Series,
) -> tuple[float | None, float | None, float | None, str | None]:
    """5/20/60일 SMA + 배열 판단."""
    ma5 = round(float(closes.rolling(5).mean().iloc[-1]), 2) if len(closes) >= 5 else None
    ma20 = round(float(closes.rolling(20).mean().iloc[-1]), 2) if len(closes) >= 20 else None
    ma60 = round(float(closes.rolling(60).mean().iloc[-1]), 2) if len(closes) >= 60 else None

    if ma5 is not None and ma20 is not None and ma60 is not None:
        if ma5 > ma20 > ma60:
            trend = "정배열"
        elif ma5 < ma20 < ma60:
            trend = "역배열"
        else:
            trend = "혼조"
    else:
        trend = None

    return ma5, ma20, ma60, trend


def _calc_volume_ratio(volumes: pd.Series, period: int = 20) -> float | None:
    """당일 거래량 / 20일 평균 거래량."""
    if len(volumes) < period:
        return None
    avg = float(volumes.iloc[-period:].mean())
    if avg == 0:
        return None
    return round(float(volumes.iloc[-1]) / avg, 2)


def _calc_obv_trend(closes: pd.Series, volumes: pd.Series, window: int = 10) -> str | None:
    """OBV(On-Balance Volume) 추세 판단."""
    if len(closes) < window + 1:
        return None
    direction = (closes.diff() > 0).astype(int) * 2 - 1  # +1 or -1
    direction.iloc[0] = 0
    obv = (direction * volumes).cumsum()
    recent = obv.iloc[-window:]
    x = np.arange(len(recent))
    slope = np.polyfit(x, recent.values.astype(float), 1)[0]
    if slope > 0:
        return "상승"
    elif slope < 0:
        return "하락"
    return "횡보"


def _detect_rsi_divergence(
    closes: pd.Series, period: int = 14, lookback: int = 14,
) -> str | None:
    """RSI 다이버전스 탐지 (최근 lookback 일 범위)."""
    if len(closes) < period + lookback + 1:
        return None
    # RSI 시리즈 계산
    delta = closes.diff()
    gain = delta.clip(lower=0)
    loss = (-delta.clip(upper=0))
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_series = 100 - (100 / (1 + rs))

    recent_closes = closes.iloc[-lookback:]
    recent_rsi = rsi_series.iloc[-lookback:]
    mid = lookback // 2

    close_first = recent_closes.iloc[:mid]
    close_second = recent_closes.iloc[mid:]
    rsi_first = recent_rsi.iloc[:mid]
    rsi_second = recent_rsi.iloc[mid:]

    # 하락 다이버전스: 가격 higher high, RSI lower high
    if close_second.max() > close_first.max() and rsi_second.max() < rsi_first.max():
        return "하락다이버전스"
    # 상승 다이버전스: 가격 lower low, RSI higher low
    if close_second.min() < close_first.min() and rsi_second.min() > rsi_first.min():
        return "상승다이버전스"
    return None


def _generate_signal_summary(ti: TechnicalIndicators) -> str:
    """기술적 지표를 종합한 한 줄 해석."""
    signals: list[str] = []

    if ti.rsi_14 is not None:
        if ti.rsi_14 >= 80:
            signals.append("RSI 극과매수(단기 과열)")
        elif ti.rsi_14 >= 70:
            signals.append("RSI 과매수 진입")
        elif ti.rsi_14 <= 20:
            signals.append("RSI 극과매도(반등 가능성)")
        elif ti.rsi_14 <= 30:
            signals.append("RSI 과매도 구간")

    if ti.rsi_divergence:
        signals.append(f"RSI {ti.rsi_divergence}")

    if ti.macd is not None and ti.macd_signal is not None:
        if ti.macd > ti.macd_signal and ti.macd_histogram and ti.macd_histogram > 0:
            signals.append("MACD 골든크로스 진행")
        elif ti.macd < ti.macd_signal and ti.macd_histogram and ti.macd_histogram < 0:
            signals.append("MACD 데드크로스 진행")

    if ti.bb_position:
        bb_map = {
            "상단돌파": "볼린저 상단돌파(과열)",
            "하단이탈": "볼린저 하단이탈(급락 과잉)",
        }
        if ti.bb_position in bb_map:
            signals.append(bb_map[ti.bb_position])

    if ti.ma_trend:
        trend_map = {
            "정배열": "이평선 정배열(상승추세)",
            "역배열": "이평선 역배열(하락추세)",
        }
        if ti.ma_trend in trend_map:
            signals.append(trend_map[ti.ma_trend])

    if ti.volume_ratio and ti.volume_ratio >= 3.0:
        signals.append(f"거래량 {ti.volume_ratio:.1f}배 폭증")

    if ti.obv_trend:
        obv_map = {
            "상승": "OBV 상승(자금 유입)",
            "하락": "OBV 하락(자금 유출)",
        }
        if ti.obv_trend in obv_map:
            signals.append(obv_map[ti.obv_trend])

    return " / ".join(signals) if signals else "특이 시그널 없음"


def compute_technical(code: str, days: int = 90) -> TechnicalIndicators | None:
    """종목 코드에 대한 기술적 지표를 계산한다.

    Args:
        code: 종목 코드 (예: "005930")
        days: 조회할 일봉 수

    Returns:
        TechnicalIndicators 또는 데이터 부족 시 None
    """
    try:
        end = datetime.now().date()
        start = end - timedelta(days=days)
        df = fdr.DataReader(code, start, end)
    except Exception as exc:
        logger.warning("technical data fetch failed for %s: %s", code, exc)
        return None

    if df is None or df.empty or len(df) < 5:
        logger.warning("insufficient data for %s (%d rows)", code, len(df) if df is not None else 0)
        return None

    closes = df["Close"]
    volumes = df["Volume"]

    rsi = _calc_rsi(closes)
    macd_val, macd_sig, macd_hist = _calc_macd(closes)
    bb_upper, bb_middle, bb_lower, bb_pos = _calc_bollinger(closes)
    ma5, ma20, ma60, ma_trend = _calc_ma(closes)
    vol_ratio = _calc_volume_ratio(volumes)

    return TechnicalIndicators(
        rsi_14=rsi,
        macd=macd_val,
        macd_signal=macd_sig,
        macd_histogram=macd_hist,
        bb_upper=bb_upper,
        bb_middle=bb_middle,
        bb_lower=bb_lower,
        bb_position=bb_pos,
        ma_5=ma5,
        ma_20=ma20,
        ma_60=ma60,
        ma_trend=ma_trend,
        volume_ratio=vol_ratio,
    )


def compute_technical_from_df(df: pd.DataFrame) -> TechnicalIndicators | None:
    """이미 가져온 DataFrame으로 기술적 지표를 계산한다.

    Args:
        df: OHLCV DataFrame (Close, Volume 컬럼 필수)

    Returns:
        TechnicalIndicators 또는 데이터 부족 시 None
    """
    if df is None or df.empty or len(df) < 5:
        return None

    closes = df["Close"]
    volumes = df["Volume"]

    rsi = _calc_rsi(closes)
    macd_val, macd_sig, macd_hist = _calc_macd(closes)
    bb_upper, bb_middle, bb_lower, bb_pos = _calc_bollinger(closes)
    ma5, ma20, ma60, ma_trend = _calc_ma(closes)
    vol_ratio = _calc_volume_ratio(volumes)
    obv_trend = _calc_obv_trend(closes, volumes)
    rsi_div = _detect_rsi_divergence(closes)

    ti = TechnicalIndicators(
        rsi_14=rsi,
        macd=macd_val,
        macd_signal=macd_sig,
        macd_histogram=macd_hist,
        bb_upper=bb_upper,
        bb_middle=bb_middle,
        bb_lower=bb_lower,
        bb_position=bb_pos,
        ma_5=ma5,
        ma_20=ma20,
        ma_60=ma60,
        ma_trend=ma_trend,
        volume_ratio=vol_ratio,
        obv_trend=obv_trend,
        rsi_divergence=rsi_div,
    )
    ti.signal_summary = _generate_signal_summary(ti)
    return ti


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    target = sys.argv[1] if len(sys.argv) > 1 else "005930"
    result = compute_technical(target, days=90)
    if result is None:
        print(f"기술적 지표 계산 실패: {target}")
        sys.exit(1)

    print(f"\n{'='*50}")
    print(f"기술적 지표 — {target}")
    print(f"{'='*50}")
    print(f"  RSI(14):      {result.rsi_14}")
    print(f"  MACD:         {result.macd}")
    print(f"  MACD 시그널:  {result.macd_signal}")
    print(f"  MACD 히스토:  {result.macd_histogram}")
    print(f"  BB 상단:      {result.bb_upper}")
    print(f"  BB 중단:      {result.bb_middle}")
    print(f"  BB 하단:      {result.bb_lower}")
    print(f"  BB 위치:      {result.bb_position}")
    print(f"  MA(5):        {result.ma_5}")
    print(f"  MA(20):       {result.ma_20}")
    print(f"  MA(60):       {result.ma_60}")
    print(f"  MA 배열:      {result.ma_trend}")
    print(f"  거래량 비율:  {result.volume_ratio}배")
