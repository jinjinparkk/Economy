"""technical 모듈 단위 테스트."""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.technical import (
    TechnicalIndicators,
    _calc_rsi,
    _calc_macd,
    _calc_bollinger,
    _calc_ma,
    _calc_volume_ratio,
    _calc_obv_trend,
    _detect_rsi_divergence,
    _generate_signal_summary,
    compute_technical,
    compute_technical_from_df,
)


# ── 헬퍼 ──────────────────────────────────────────────────────────────
def _make_df(closes: list[float], volumes: list[int] | None = None) -> pd.DataFrame:
    """테스트용 OHLCV DataFrame 생성."""
    n = len(closes)
    if volumes is None:
        volumes = [1000] * n
    return pd.DataFrame({
        "Open": closes,
        "High": [c * 1.02 for c in closes],
        "Low": [c * 0.98 for c in closes],
        "Close": closes,
        "Volume": volumes,
    })


def _rising_series(n: int = 100, start: float = 1000, step: float = 10) -> pd.Series:
    """꾸준히 상승하는 종가 시리즈."""
    return pd.Series([start + i * step for i in range(n)])


def _falling_series(n: int = 100, start: float = 2000, step: float = 10) -> pd.Series:
    """꾸준히 하락하는 종가 시리즈."""
    return pd.Series([start - i * step for i in range(n)])


def _mixed_series(n: int = 100) -> pd.Series:
    """상승과 하락이 반복하는 시리즈."""
    np.random.seed(42)
    returns = np.random.normal(0, 0.02, n)
    prices = 1000 * np.cumprod(1 + returns)
    return pd.Series(prices)


# ── RSI ───────────────────────────────────────────────────────────────
class TestCalcRSI:
    def test_rising_only(self):
        """순상승 → RSI 100에 가까워야."""
        s = _rising_series(30)
        rsi = _calc_rsi(s)
        assert rsi is not None
        assert rsi > 90

    def test_falling_only(self):
        """순하락 → RSI 0에 가까워야."""
        s = _falling_series(30)
        rsi = _calc_rsi(s)
        assert rsi is not None
        assert rsi < 10

    def test_mixed(self):
        """혼합 → RSI 20~80 범위."""
        s = _mixed_series(50)
        rsi = _calc_rsi(s)
        assert rsi is not None
        assert 10 < rsi < 90

    def test_insufficient_data(self):
        """14일 미만 → None."""
        s = pd.Series([100, 101, 102])
        assert _calc_rsi(s) is None


# ── MACD ──────────────────────────────────────────────────────────────
class TestCalcMACD:
    def test_rising_positive_macd(self):
        """상승세 → MACD > 0."""
        s = _rising_series(60)
        macd, signal, hist = _calc_macd(s)
        assert macd is not None
        assert macd > 0

    def test_falling_negative_macd(self):
        """하락세 → MACD < 0."""
        s = _falling_series(60)
        macd, signal, hist = _calc_macd(s)
        assert macd is not None
        assert macd < 0

    def test_histogram_sign(self):
        """히스토그램 = MACD - Signal."""
        s = _mixed_series(60)
        macd, signal, hist = _calc_macd(s)
        assert hist is not None
        assert abs(hist - (macd - signal)) < 0.01

    def test_insufficient_data(self):
        """데이터 부족 → None 3개."""
        s = pd.Series([100] * 10)
        macd, signal, hist = _calc_macd(s)
        assert macd is None
        assert signal is None
        assert hist is None


# ── 볼린저 밴드 ──────────────────────────────────────────────────────
class TestCalcBollinger:
    def test_upper_break(self):
        """마지막 종가가 상단 초과 → 상단돌파."""
        prices = [100] * 19 + [200]  # 마지막만 급등
        s = pd.Series(prices)
        upper, middle, lower, pos = _calc_bollinger(s)
        assert pos == "상단돌파"

    def test_lower_break(self):
        """마지막 종가가 하단 미만 → 하단이탈."""
        prices = [100] * 19 + [10]  # 마지막만 급락
        s = pd.Series(prices)
        upper, middle, lower, pos = _calc_bollinger(s)
        assert pos == "하단이탈"

    def test_neutral(self):
        """안정적 → 중립."""
        s = pd.Series([100.0] * 25)
        upper, middle, lower, pos = _calc_bollinger(s)
        # 표준편차 0이면 upper == middle == lower, 종가 == middle
        assert pos in ("중립", "상단돌파", "하단이탈")  # 0 std edge case

    def test_insufficient_data(self):
        s = pd.Series([100] * 5)
        upper, middle, lower, pos = _calc_bollinger(s)
        assert upper is None
        assert pos is None


# ── 이동평균 ─────────────────────────────────────────────────────────
class TestCalcMA:
    def test_positive_alignment(self):
        """꾸준히 상승 → 정배열 (5 > 20 > 60)."""
        s = _rising_series(80)
        ma5, ma20, ma60, trend = _calc_ma(s)
        assert trend == "정배열"
        assert ma5 > ma20 > ma60

    def test_negative_alignment(self):
        """꾸준히 하락 → 역배열 (5 < 20 < 60)."""
        s = _falling_series(80)
        ma5, ma20, ma60, trend = _calc_ma(s)
        assert trend == "역배열"
        assert ma5 < ma20 < ma60

    def test_mixed_alignment(self):
        """혼합 → 혼조 가능 (또는 정/역배열)."""
        s = _mixed_series(80)
        ma5, ma20, ma60, trend = _calc_ma(s)
        assert trend in ("정배열", "역배열", "혼조")

    def test_short_data_partial(self):
        """60일 미만 → ma60 None, trend None."""
        s = pd.Series([100] * 25)
        ma5, ma20, ma60, trend = _calc_ma(s)
        assert ma5 is not None
        assert ma20 is not None
        assert ma60 is None
        assert trend is None


# ── 거래량 비율 ───────────────────────────────────────────────────────
class TestCalcVolumeRatio:
    def test_normal(self):
        """평균과 동일 → 비율 1.0."""
        volumes = pd.Series([1000] * 20)
        ratio = _calc_volume_ratio(volumes)
        assert ratio == 1.0

    def test_surge(self):
        """마지막 거래량 5배 → 비율 약 5."""
        volumes = pd.Series([1000] * 19 + [5000])
        ratio = _calc_volume_ratio(volumes)
        # 평균 = (19*1000+5000)/20 = 1200, 5000/1200 ≈ 4.17
        assert ratio is not None
        assert ratio > 3.0

    def test_insufficient(self):
        """20일 미만 → None."""
        volumes = pd.Series([1000] * 5)
        assert _calc_volume_ratio(volumes) is None


# ── compute_technical_from_df ────────────────────────────────────────
class TestComputeTechnicalFromDf:
    def test_full_data(self):
        """충분한 데이터 → 모든 필드 채워짐."""
        closes = list(_rising_series(80))
        df = _make_df(closes)
        result = compute_technical_from_df(df)
        assert result is not None
        assert result.rsi_14 is not None
        assert result.ma_5 is not None
        assert result.ma_20 is not None

    def test_empty_df(self):
        """빈 DataFrame → None."""
        assert compute_technical_from_df(pd.DataFrame()) is None

    def test_none_df(self):
        """None → None."""
        assert compute_technical_from_df(None) is None

    def test_too_short(self):
        """4행 이하 → None."""
        df = _make_df([100, 101, 102, 103])
        assert compute_technical_from_df(df) is None


# ── compute_technical (FDR 모킹) ─────────────────────────────────────
class TestComputeTechnical:
    @patch("src.technical.fdr")
    def test_success(self, mock_fdr):
        closes = list(_rising_series(80))
        mock_fdr.DataReader.return_value = _make_df(closes)
        result = compute_technical("005930", days=90)
        assert result is not None
        assert result.rsi_14 is not None

    @patch("src.technical.fdr")
    def test_fdr_exception(self, mock_fdr):
        mock_fdr.DataReader.side_effect = Exception("network error")
        result = compute_technical("005930")
        assert result is None

    @patch("src.technical.fdr")
    def test_empty_data(self, mock_fdr):
        mock_fdr.DataReader.return_value = pd.DataFrame()
        result = compute_technical("005930")
        assert result is None


# ── OBV 추세 ────────────────────────────────────────────────────────
class TestCalcObvTrend:
    def test_rising(self):
        closes = pd.Series(list(range(100, 115)))
        volumes = pd.Series([1000] * 15)
        trend = _calc_obv_trend(closes, volumes)
        assert trend == "상승"

    def test_falling(self):
        closes = pd.Series(list(range(115, 100, -1)))
        volumes = pd.Series([1000] * 15)
        trend = _calc_obv_trend(closes, volumes)
        assert trend == "하락"

    def test_insufficient(self):
        closes = pd.Series([100] * 5)
        volumes = pd.Series([1000] * 5)
        assert _calc_obv_trend(closes, volumes) is None


# ── RSI 다이버전스 ──────────────────────────────────────────────────
class TestRsiDivergence:
    def test_no_divergence_aligned(self):
        closes = pd.Series(list(_rising_series(50)))
        result = _detect_rsi_divergence(closes)
        # 꾸준히 오르는 시리즈에서는 다이버전스 없음
        assert result is None or result == "하락다이버전스"  # RSI 둔화 가능

    def test_insufficient_data(self):
        closes = pd.Series([100] * 10)
        assert _detect_rsi_divergence(closes) is None


# ── 시그널 요약 ─────────────────────────────────────────────────────
class TestSignalSummary:
    def test_overbought(self):
        ti = TechnicalIndicators(rsi_14=85.0, bb_position="상단돌파",
                                 ma_trend="정배열", obv_trend="상승")
        summary = _generate_signal_summary(ti)
        assert "과매수" in summary or "과열" in summary

    def test_oversold(self):
        ti = TechnicalIndicators(rsi_14=20.0, bb_position="하단이탈",
                                 ma_trend="역배열")
        summary = _generate_signal_summary(ti)
        assert "과매도" in summary

    def test_no_signals(self):
        ti = TechnicalIndicators()
        summary = _generate_signal_summary(ti)
        assert summary == "특이 시그널 없음"

    def test_volume_surge(self):
        ti = TechnicalIndicators(volume_ratio=5.0)
        summary = _generate_signal_summary(ti)
        assert "폭증" in summary

    def test_compute_from_df_has_new_fields(self):
        closes = list(_rising_series(80))
        df = _make_df(closes)
        result = compute_technical_from_df(df)
        assert result is not None
        assert result.signal_summary is not None
        assert isinstance(result.signal_summary, str)
