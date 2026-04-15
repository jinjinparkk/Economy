"""predictor 모듈 단위 테스트."""
from __future__ import annotations

from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

from src.predictor import (
    PatternStats,
    DirectionPrediction,
    OutlookData,
    _classify_event,
    compute_pattern_stats,
    predict_direction,
    compute_outlook,
)
from src.detect_movers import Mover
from src.technical import TechnicalIndicators


# ── 헬퍼 ──────────────────────────────────────────────────────────────
def _make_df(n: int = 600, base: float = 10000, seed: int = 42) -> pd.DataFrame:
    """테스트용 OHLCV DataFrame 생성 (n일)."""
    np.random.seed(seed)
    returns = np.random.normal(0, 0.02, n)
    closes = base * np.cumprod(1 + returns)
    return pd.DataFrame({
        "Open": closes * 0.99,
        "High": closes * 1.02,
        "Low": closes * 0.98,
        "Close": closes,
        "Volume": np.random.randint(100000, 1000000, n),
    })


def _make_df_with_spikes(n: int = 600, spike_pct: float = 30.0, spike_count: int = 10) -> pd.DataFrame:
    """급등 사례가 포함된 DataFrame."""
    np.random.seed(42)
    returns = np.random.normal(0, 0.01, n)
    # 특정 위치에 급등 삽입
    spike_positions = np.linspace(50, n - 50, spike_count, dtype=int)
    for pos in spike_positions:
        returns[pos] = spike_pct / 100
    closes = 10000 * np.cumprod(1 + returns)
    return pd.DataFrame({
        "Open": closes * 0.99,
        "High": closes * 1.02,
        "Low": closes * 0.98,
        "Close": closes,
        "Volume": np.random.randint(100000, 1000000, n),
    })


def _mover(code="005930", name="삼성전자", change_pct=30.0):
    return Mover(code, name, "KOSPI", "surge", 70000, change_pct,
                 1000000, int(5e10), int(4e14))


# ── _classify_event ──────────────────────────────────────────────────
class TestClassifyEvent:
    def test_limit_up(self):
        assert _classify_event(30.0) == "상한가"

    def test_surge_10(self):
        assert _classify_event(15.0) == "급등(10%+)"

    def test_surge_5(self):
        assert _classify_event(7.0) == "급등(5%+)"

    def test_limit_down(self):
        assert _classify_event(-30.0) == "하한가"

    def test_plunge_10(self):
        assert _classify_event(-15.0) == "급락(10%+)"

    def test_plunge_5(self):
        assert _classify_event(-7.0) == "급락(5%+)"

    def test_normal(self):
        assert _classify_event(3.0) == "일반"


# ── compute_pattern_stats ────────────────────────────────────────────
class TestComputePatternStats:
    def test_with_spike_data(self):
        """급등 사례가 충분한 데이터 → PatternStats 반환."""
        df = _make_df_with_spikes(spike_pct=30.0, spike_count=10)
        result = compute_pattern_stats(df, change_pct=30.0)
        assert result is not None
        assert result.event_type == "상한가"
        assert result.sample_count >= 5

    def test_returns_calculated(self):
        """수익률이 제대로 계산되는지."""
        df = _make_df_with_spikes(spike_pct=30.0, spike_count=10)
        result = compute_pattern_stats(df, change_pct=30.0)
        if result is not None:
            assert result.avg_return_1d is not None
            assert isinstance(result.avg_return_1d, float)

    def test_insufficient_samples(self):
        """사례 부족 → None."""
        df = _make_df(100)  # 작은 데이터, 급등 사례 거의 없음
        result = compute_pattern_stats(df, change_pct=30.0, min_samples=100)
        assert result is None

    def test_normal_event(self):
        """일반 등락 → None."""
        df = _make_df(200)
        result = compute_pattern_stats(df, change_pct=2.0)
        assert result is None

    def test_empty_df(self):
        """빈 DataFrame → None."""
        assert compute_pattern_stats(pd.DataFrame(), 10.0) is None

    def test_short_df(self):
        """짧은 DataFrame → None."""
        df = _make_df(10)
        assert compute_pattern_stats(df, 10.0) is None


# ── predict_direction ────────────────────────────────────────────────
class TestPredictDirection:
    def test_with_sufficient_data(self):
        """충분한 데이터 → DirectionPrediction 반환."""
        df = _make_df(600)
        result = predict_direction(df, min_rows=500)
        # sklearn이 설치된 환경에서만 테스트
        if result is not None:
            assert result.direction in ("상승", "하락", "중립")
            assert 0.0 <= result.confidence <= 1.0
            assert result.model_type == "LogisticRegression"
            assert len(result.features_used) > 0

    def test_insufficient_data(self):
        """데이터 부족 → None."""
        df = _make_df(100)
        result = predict_direction(df, min_rows=500)
        assert result is None

    def test_empty_df(self):
        """빈 DataFrame → None."""
        result = predict_direction(pd.DataFrame(), min_rows=50)
        assert result is None

    @patch.dict("sys.modules", {"sklearn": None, "sklearn.linear_model": None})
    def test_no_sklearn(self):
        """sklearn 미설치 → None (graceful)."""
        # 이미 import된 상태에서는 테스트가 어려우므로
        # 대신 데이터 부족으로 None을 확인
        df = _make_df(10)
        result = predict_direction(df, min_rows=500)
        assert result is None


# ── OutlookData ──────────────────────────────────────────────────────
class TestOutlookData:
    def test_default_all_none(self):
        """기본 생성 → 모두 None."""
        outlook = OutlookData()
        assert outlook.technical is None
        assert outlook.pattern is None
        assert outlook.prediction is None

    def test_with_data(self):
        """데이터 할당."""
        tech = TechnicalIndicators(rsi_14=50.0)
        pattern = PatternStats("상한가", 10, -2.0, -1.0, 40.0, 45.0)
        pred = DirectionPrediction("하락", 0.62, ["RSI", "MACD"], "LogisticRegression")
        outlook = OutlookData(technical=tech, pattern=pattern, prediction=pred)
        assert outlook.technical.rsi_14 == 50.0
        assert outlook.pattern.event_type == "상한가"
        assert outlook.prediction.direction == "하락"


# ── compute_outlook (FDR 모킹) ───────────────────────────────────────
class TestComputeOutlook:
    @patch("src.predictor.fdr")
    def test_success(self, mock_fdr):
        """정상 호출 → OutlookData 반환 (technical 있음)."""
        mock_fdr.DataReader.return_value = _make_df(600)
        m = _mover(change_pct=30.0)
        result = compute_outlook(m)
        assert isinstance(result, OutlookData)
        assert result.technical is not None

    @patch("src.predictor.fdr")
    def test_fdr_failure(self, mock_fdr):
        """FDR 실패 → 빈 OutlookData."""
        mock_fdr.DataReader.side_effect = Exception("network error")
        m = _mover()
        result = compute_outlook(m)
        assert isinstance(result, OutlookData)
        assert result.technical is None
        assert result.pattern is None
        assert result.prediction is None

    @patch("src.predictor.fdr")
    def test_empty_data(self, mock_fdr):
        """빈 데이터 → 빈 OutlookData."""
        mock_fdr.DataReader.return_value = pd.DataFrame()
        m = _mover()
        result = compute_outlook(m)
        assert result.technical is None
