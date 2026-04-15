"""fetch_macro 모듈 단위 테스트."""
from __future__ import annotations

from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

from src.fetch_macro import (
    MacroIndicator,
    MacroSnapshot,
    _fetch_indicator,
    fetch_macro_snapshot,
)


# ── MacroIndicator ───────────────────────────────────────────────────
class TestMacroIndicator:
    def test_str(self):
        ind = MacroIndicator("S&P 500", "US500", 5400.0, 5300.0, 100.0, 1.89, "2026-04-08")
        s = str(ind)
        assert "S&P 500" in s
        assert "+1.89%" in s

    def test_negative_change(self):
        ind = MacroIndicator("VIX", "VIX", 25.0, 30.0, -5.0, -16.67, "2026-04-08")
        assert "-16.67%" in str(ind)


# ── MacroSnapshot ────────────────────────────────────────────────────
class TestMacroSnapshot:
    def test_to_summary_dict(self):
        snap = MacroSnapshot()
        snap.us_indices["SP500"] = MacroIndicator("S&P 500", "US500", 5400, 5300, 100, 1.89, "2026-04-08")
        snap.fx["USDKRW"] = MacroIndicator("원/달러", "USD/KRW", 1380, 1370, 10, 0.73, "2026-04-08")

        d = snap.to_summary_dict()
        assert "S&P 500" in d
        assert "원/달러" in d
        assert d["S&P 500"]["Close"] == 5400
        assert d["S&P 500"]["ChangePct"] == 1.89

    def test_is_empty_true(self):
        assert MacroSnapshot().is_empty()

    def test_is_empty_false(self):
        snap = MacroSnapshot()
        snap.volatility["VIX"] = MacroIndicator("VIX", "VIX", 20, 22, -2, -9.09, "2026-04-08")
        assert not snap.is_empty()

    def test_all_indicators(self):
        snap = MacroSnapshot()
        snap.us_indices["A"] = MacroIndicator("A", "a", 1, 1, 0, 0, "d")
        snap.bonds["B"] = MacroIndicator("B", "b", 2, 2, 0, 0, "d")
        assert len(snap.all_indicators) == 2


# ── _fetch_indicator (mock) ──────────────────────────────────────────
class TestFetchIndicator:
    @patch("src.fetch_macro.fdr.DataReader")
    def test_success(self, mock_dr):
        dates = pd.date_range("2026-04-07", periods=2, freq="D")
        df = pd.DataFrame({"Close": [5300.0, 5400.0]}, index=dates)
        mock_dr.return_value = df

        ind = _fetch_indicator("SP500", "S&P 500", "US500")
        assert ind is not None
        assert ind.close == 5400.0
        assert ind.prev_close == 5300.0
        assert abs(ind.change_pct - 1.887) < 0.01

    @patch("src.fetch_macro.fdr.DataReader")
    def test_empty_returns_none(self, mock_dr):
        mock_dr.return_value = pd.DataFrame()
        ind = _fetch_indicator("SP500", "S&P 500", "US500")
        assert ind is None

    @patch("src.fetch_macro.fdr.DataReader")
    def test_error_returns_none(self, mock_dr):
        mock_dr.side_effect = Exception("network error")
        ind = _fetch_indicator("SP500", "S&P 500", "US500")
        assert ind is None


# ── fetch_macro_snapshot (mock) ──────────────────────────────────────
class TestFetchMacroSnapshot:
    @patch("src.fetch_macro._fetch_indicator")
    def test_returns_snapshot(self, mock_fetch):
        mock_fetch.return_value = MacroIndicator("T", "T", 100, 99, 1, 1.01, "2026-04-08")
        snap = fetch_macro_snapshot()
        assert not snap.is_empty()
        # 11개 지표 (SP500,NASDAQ,DOW,SOXX,USDKRW,DXY,WTI,GOLD,VIX,US10Y,US2Y)
        assert len(snap.all_indicators) == 11

    @patch("src.fetch_macro._fetch_indicator")
    def test_partial_failure(self, mock_fetch):
        call_count = [0]
        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] <= 3:  # 처음 3개만 성공
                return MacroIndicator("T", "T", 100, 99, 1, 1.01, "d")
            return None
        mock_fetch.side_effect = side_effect

        snap = fetch_macro_snapshot()
        assert len(snap.all_indicators) == 3


# ── yield_spread / market_regime / to_narrative ──────────────────
class TestMacroSnapshotDerived:
    def test_yield_spread_normal(self):
        snap = MacroSnapshot()
        snap.bonds["US10Y"] = MacroIndicator("미국채10Y", "US10YT", 4.3, 4.2, 0.1, 2.38, "d")
        snap.bonds["US2Y"] = MacroIndicator("미국채2Y", "US2YT", 3.8, 3.7, 0.1, 2.7, "d")
        assert snap.yield_spread == 0.5

    def test_yield_spread_inverted(self):
        snap = MacroSnapshot()
        snap.bonds["US10Y"] = MacroIndicator("미국채10Y", "US10YT", 3.5, 3.4, 0.1, 2.9, "d")
        snap.bonds["US2Y"] = MacroIndicator("미국채2Y", "US2YT", 4.0, 3.9, 0.1, 2.6, "d")
        assert snap.yield_spread == -0.5

    def test_yield_spread_missing_2y(self):
        snap = MacroSnapshot()
        snap.bonds["US10Y"] = MacroIndicator("미국채10Y", "US10YT", 4.3, 4.2, 0.1, 2.38, "d")
        assert snap.yield_spread is None

    def test_market_regime_calm(self):
        snap = MacroSnapshot()
        snap.volatility["VIX"] = MacroIndicator("VIX", "VIX", 12.0, 13.0, -1.0, -7.69, "d")
        assert snap.market_regime == "안정"

    def test_market_regime_normal(self):
        snap = MacroSnapshot()
        snap.volatility["VIX"] = MacroIndicator("VIX", "VIX", 18.0, 17.0, 1.0, 5.88, "d")
        assert snap.market_regime == "보통"

    def test_market_regime_anxious(self):
        snap = MacroSnapshot()
        snap.volatility["VIX"] = MacroIndicator("VIX", "VIX", 25.0, 22.0, 3.0, 13.64, "d")
        assert snap.market_regime == "불안"

    def test_market_regime_fear(self):
        snap = MacroSnapshot()
        snap.volatility["VIX"] = MacroIndicator("VIX", "VIX", 35.0, 30.0, 5.0, 16.67, "d")
        assert snap.market_regime == "공포"

    def test_market_regime_no_vix(self):
        snap = MacroSnapshot()
        assert snap.market_regime == "판단불가"

    def test_to_narrative_not_empty(self):
        snap = MacroSnapshot()
        snap.us_indices["SP500"] = MacroIndicator("S&P 500", "US500", 5400, 5300, 100, 1.89, "d")
        snap.us_indices["NASDAQ"] = MacroIndicator("NASDAQ", "IXIC", 17000, 16800, 200, 1.19, "d")
        snap.volatility["VIX"] = MacroIndicator("VIX", "VIX", 18.0, 19.0, -1.0, -5.26, "d")
        text = snap.to_narrative()
        assert "S&P500" in text
        assert "나스닥" in text
        assert "VIX" in text

    def test_to_summary_dict_includes_derived(self):
        snap = MacroSnapshot()
        snap.volatility["VIX"] = MacroIndicator("VIX", "VIX", 20.0, 22.0, -2.0, -9.09, "d")
        snap.bonds["US10Y"] = MacroIndicator("미국채10Y", "US10YT", 4.3, 4.2, 0.1, 2.38, "d")
        snap.bonds["US2Y"] = MacroIndicator("미국채2Y", "US2YT", 3.8, 3.7, 0.1, 2.7, "d")
        d = snap.to_summary_dict()
        assert "_yield_spread" in d
        assert "_market_regime" in d
        assert d["_market_regime"] == "불안"
