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
        # 41개 지표 (기존 11 + 섹터11 + Mag7 + 스타일3 + 아시아3 + 유럽2 + 추가원자재2 + 신용2)
        assert len(snap.all_indicators) == 41

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


# ── 신규 파생 프로퍼티 테스트 ─────────────────────────────────────
class TestMacroSnapshotNewProperties:
    def test_sector_top_bottom(self):
        snap = MacroSnapshot()
        snap.sectors["XLK"] = MacroIndicator("기술", "XLK", 200, 196, 4, 2.04, "d")
        snap.sectors["XLE"] = MacroIndicator("에너지", "XLE", 90, 92, -2, -2.17, "d")
        result = snap.sector_top_bottom
        assert result is not None
        assert "기술" in result[0]
        assert "+2.04%" in result[0]
        assert "에너지" in result[1]
        assert "-2.17%" in result[1]

    def test_sector_top_bottom_empty(self):
        snap = MacroSnapshot()
        assert snap.sector_top_bottom is None

    def test_growth_value_ratio_positive(self):
        snap = MacroSnapshot()
        snap.style["QQQ"] = MacroIndicator("나스닥100ETF", "QQQ", 500, 492, 8, 1.63, "d")
        snap.style["VTV"] = MacroIndicator("가치ETF", "VTV", 160, 159.5, 0.5, 0.31, "d")
        assert snap.growth_value_ratio == 1.32

    def test_growth_value_ratio_negative(self):
        snap = MacroSnapshot()
        snap.style["QQQ"] = MacroIndicator("나스닥100ETF", "QQQ", 490, 492, -2, -0.41, "d")
        snap.style["VTV"] = MacroIndicator("가치ETF", "VTV", 161, 159.5, 1.5, 0.94, "d")
        assert snap.growth_value_ratio == -1.35

    def test_growth_value_ratio_missing(self):
        snap = MacroSnapshot()
        assert snap.growth_value_ratio is None

    def test_credit_stress_warning(self):
        snap = MacroSnapshot()
        snap.credit["HYG"] = MacroIndicator("하이일드채권", "HYG", 75, 76.2, -1.2, -1.57, "d")
        assert snap.credit_stress == "경고 (하이일드 급락)"

    def test_credit_stress_caution(self):
        snap = MacroSnapshot()
        snap.credit["HYG"] = MacroIndicator("하이일드채권", "HYG", 76, 76.3, -0.3, -0.39, "d")
        assert snap.credit_stress == "주의"

    def test_credit_stress_stable(self):
        snap = MacroSnapshot()
        snap.credit["HYG"] = MacroIndicator("하이일드채권", "HYG", 76, 75.9, 0.1, 0.13, "d")
        assert snap.credit_stress == "안정"

    def test_credit_stress_none(self):
        snap = MacroSnapshot()
        assert snap.credit_stress is None

    def test_mag7_avg(self):
        snap = MacroSnapshot()
        snap.mega_caps["NVDA"] = MacroIndicator("엔비디아", "NVDA", 950, 920, 30, 3.26, "d")
        snap.mega_caps["AAPL"] = MacroIndicator("애플", "AAPL", 200, 198, 2, 1.01, "d")
        snap.mega_caps["TSLA"] = MacroIndicator("테슬라", "TSLA", 245, 250, -5, -2.0, "d")
        expected = round((3.26 + 1.01 + (-2.0)) / 3, 2)
        assert snap.mag7_avg == expected

    def test_mag7_avg_empty(self):
        snap = MacroSnapshot()
        assert snap.mag7_avg is None

    def test_all_indicators_includes_new(self):
        snap = MacroSnapshot()
        snap.sectors["XLK"] = MacroIndicator("기술", "XLK", 200, 196, 4, 2.04, "d")
        snap.mega_caps["NVDA"] = MacroIndicator("엔비디아", "NVDA", 950, 920, 30, 3.26, "d")
        snap.style["QQQ"] = MacroIndicator("나스닥100ETF", "QQQ", 500, 492, 8, 1.63, "d")
        snap.asia["NIKKEI"] = MacroIndicator("닛케이225", "^N225", 38500, 38300, 200, 0.52, "d")
        snap.europe["DAX"] = MacroIndicator("독일DAX", "^GDAXI", 18200, 18160, 40, 0.22, "d")
        snap.credit["HYG"] = MacroIndicator("하이일드채권", "HYG", 76, 75.9, 0.1, 0.13, "d")
        assert len(snap.all_indicators) == 6

    def test_is_empty_with_new_fields(self):
        snap = MacroSnapshot()
        snap.sectors["XLK"] = MacroIndicator("기술", "XLK", 200, 196, 4, 2.04, "d")
        assert not snap.is_empty()

    def test_narrative_includes_sectors(self):
        snap = MacroSnapshot()
        snap.sectors["XLK"] = MacroIndicator("기술", "XLK", 200, 196, 4, 2.04, "d")
        snap.sectors["XLE"] = MacroIndicator("에너지", "XLE", 90, 92, -2, -2.17, "d")
        text = snap.to_narrative()
        assert "섹터" in text
        assert "기술" in text

    def test_narrative_includes_asia(self):
        snap = MacroSnapshot()
        snap.asia["NIKKEI"] = MacroIndicator("닛케이225", "^N225", 38500, 38300, 200, 0.52, "d")
        text = snap.to_narrative()
        assert "아시아" in text
        assert "닛케이" in text

    def test_narrative_includes_mag7(self):
        snap = MacroSnapshot()
        snap.mega_caps["NVDA"] = MacroIndicator("엔비디아", "NVDA", 950, 920, 30, 3.26, "d")
        text = snap.to_narrative()
        assert "Mag7" in text

    def test_to_summary_dict_includes_new_derived(self):
        snap = MacroSnapshot()
        snap.sectors["XLK"] = MacroIndicator("기술", "XLK", 200, 196, 4, 2.04, "d")
        snap.sectors["XLE"] = MacroIndicator("에너지", "XLE", 90, 92, -2, -2.17, "d")
        snap.credit["HYG"] = MacroIndicator("하이일드채권", "HYG", 76, 75.9, 0.1, 0.13, "d")
        d = snap.to_summary_dict()
        assert "_sector_top_bottom" in d
        assert "_credit_stress" in d
        assert d["_credit_stress"] == "안정"
