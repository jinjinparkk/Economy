"""fetch_flow 모듈 단위 테스트."""
from __future__ import annotations

from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

from src.fetch_flow import (
    InvestorFlow,
    TopFlowStock,
    FlowSnapshot,
    _count_consecutive,
    _fetch_investor_flow,
    _fetch_top_stocks,
    fetch_flow_snapshot,
)


# ── _count_consecutive ────────────────────────────────────────────────

class TestCountConsecutive:
    def test_empty(self):
        assert _count_consecutive([]) == 0

    def test_single_positive(self):
        assert _count_consecutive([100]) == 1

    def test_single_negative(self):
        assert _count_consecutive([-100]) == -1

    def test_single_zero(self):
        assert _count_consecutive([0]) == 0

    def test_all_positive(self):
        assert _count_consecutive([100, 200, 300]) == 3

    def test_all_negative(self):
        assert _count_consecutive([-100, -200, -300]) == -3

    def test_mixed_ends_positive(self):
        assert _count_consecutive([-100, -200, 300, 400]) == 2

    def test_mixed_ends_negative(self):
        assert _count_consecutive([100, 200, -300, -400]) == -2

    def test_alternating(self):
        assert _count_consecutive([100, -200, 300]) == 1

    def test_last_zero(self):
        assert _count_consecutive([100, 200, 0]) == 0


# ── InvestorFlow ─────────────────────────────────────────────────────

class TestInvestorFlow:
    def test_creation(self):
        flow = InvestorFlow("20260420", 1000000000, -500000000, -500000000, 3)
        assert flow.foreign_net == 1000000000
        assert flow.foreign_consecutive == 3


# ── TopFlowStock ─────────────────────────────────────────────────────

class TestTopFlowStock:
    def test_creation(self):
        stock = TopFlowStock("005930", "삼성전자", 50000000000, "외국인")
        assert stock.name == "삼성전자"
        assert stock.investor_type == "외국인"


# ── FlowSnapshot ─────────────────────────────────────────────────────

class TestFlowSnapshot:
    def _sample(self):
        kf = InvestorFlow("20260420", 100_000_000_000, -50_000_000_000, -50_000_000_000, 5)
        qf = InvestorFlow("20260420", 20_000_000_000, -10_000_000_000, -10_000_000_000, 2)
        return FlowSnapshot(
            trade_date="2026-04-20",
            kospi_flow=kf,
            kosdaq_flow=qf,
            foreign_top_buy=[TopFlowStock("005930", "삼성전자", 50_000_000_000, "외국인")],
            foreign_top_sell=[TopFlowStock("000660", "SK하이닉스", -30_000_000_000, "외국인")],
            institution_top_buy=[TopFlowStock("005380", "현대차", 20_000_000_000, "기관")],
        )

    def test_to_dict(self):
        snap = self._sample()
        d = snap.to_dict()
        assert d["trade_date"] == "2026-04-20"
        assert d["kospi_flow"]["foreign_net"] == 100_000_000_000
        assert d["kospi_flow"]["foreign_consecutive"] == 5
        assert len(d["foreign_top_buy"]) == 1
        assert d["foreign_top_buy"][0]["name"] == "삼성전자"

    def test_to_narrative(self):
        snap = self._sample()
        narrative = snap.to_narrative()
        assert "외국인" in narrative
        assert "기관" in narrative
        assert "삼성전자" in narrative
        assert "5일 연속 매수" in narrative

    def test_to_narrative_sell_consecutive(self):
        kf = InvestorFlow("20260420", -100_000_000_000, 50_000_000_000, 50_000_000_000, -3)
        snap = FlowSnapshot("2026-04-20", kf,
                            InvestorFlow("20260420", 0, 0, 0))
        narrative = snap.to_narrative()
        assert "3일 연속 매도" in narrative


# ── _fetch_investor_flow (mocked) ────────────────────────────────────

class TestFetchInvestorFlow:
    @patch("src.fetch_flow.krx_stock")
    @patch("src.fetch_flow._HAS_PYKRX", True)
    def test_basic_flow(self, mock_krx):
        df = pd.DataFrame({
            "외국인합계": [100, 200, 300],
            "기관합계": [-50, -100, -150],
            "개인": [-50, -100, -150],
        })
        mock_krx.get_market_trading_value_by_date.return_value = df
        result = _fetch_investor_flow("KOSPI", "20260420")
        assert result is not None
        assert result.foreign_net == 300
        assert result.institution_net == -150
        assert result.foreign_consecutive == 3  # 3일 연속 양수

    @patch("src.fetch_flow.krx_stock")
    @patch("src.fetch_flow._HAS_PYKRX", True)
    def test_empty_dataframe(self, mock_krx):
        mock_krx.get_market_trading_value_by_date.return_value = pd.DataFrame()
        result = _fetch_investor_flow("KOSPI", "20260420")
        assert result is None

    @patch("src.fetch_flow._HAS_PYKRX", False)
    def test_no_pykrx(self):
        result = _fetch_investor_flow("KOSPI", "20260420")
        assert result is None

    @patch("src.fetch_flow.krx_stock")
    @patch("src.fetch_flow._HAS_PYKRX", True)
    def test_exception_graceful(self, mock_krx):
        mock_krx.get_market_trading_value_by_date.side_effect = RuntimeError("network")
        result = _fetch_investor_flow("KOSPI", "20260420")
        assert result is None


# ── _fetch_top_stocks (mocked) ───────────────────────────────────────

class TestFetchTopStocks:
    @patch("src.fetch_flow.krx_stock")
    @patch("src.fetch_flow._HAS_PYKRX", True)
    def test_basic_top(self, mock_krx):
        df = pd.DataFrame(
            {
                "종목명": ["삼성전자", "SK하이닉스", "LG에너지솔루션"],
                "순매수거래대금": [500, -300, 200],
            },
            index=["005930", "000660", "373220"],
        )
        mock_krx.get_market_net_purchases_of_equities_by_ticker.return_value = df
        buy, sell = _fetch_top_stocks("20260420", "외국인", "KOSPI", top_n=2)
        assert len(buy) == 2
        assert buy[0].name == "삼성전자"
        assert len(sell) == 2
        assert sell[0].net_amount == -300  # 가장 작은 순매수

    @patch("src.fetch_flow._HAS_PYKRX", False)
    def test_no_pykrx(self):
        buy, sell = _fetch_top_stocks("20260420", "외국인", "KOSPI")
        assert buy == []
        assert sell == []


# ── fetch_flow_snapshot (mocked) ─────────────────────────────────────

class TestFetchFlowSnapshot:
    @patch("src.fetch_flow._fetch_top_stocks")
    @patch("src.fetch_flow._fetch_investor_flow")
    @patch("src.fetch_flow._HAS_PYKRX", True)
    def test_basic_snapshot(self, mock_flow, mock_top):
        mock_flow.return_value = InvestorFlow("20260420", 100, -50, -50, 2)
        mock_top.return_value = (
            [TopFlowStock("005930", "삼성전자", 500, "외국인")],
            [TopFlowStock("000660", "SK하이닉스", -300, "외국인")],
        )
        result = fetch_flow_snapshot("2026-04-20")
        assert result is not None
        assert result.trade_date == "2026-04-20"
        assert result.kospi_flow.foreign_net == 100

    @patch("src.fetch_flow._HAS_PYKRX", False)
    def test_no_pykrx_returns_none(self):
        result = fetch_flow_snapshot("2026-04-20")
        assert result is None

    @patch("src.fetch_flow._fetch_top_stocks")
    @patch("src.fetch_flow._fetch_investor_flow")
    @patch("src.fetch_flow._HAS_PYKRX", True)
    def test_flow_failure_graceful(self, mock_flow, mock_top):
        mock_flow.return_value = None
        mock_top.return_value = ([], [])
        result = fetch_flow_snapshot("2026-04-20")
        assert result is not None
        # flow가 None이면 기본값(0) 사용
        assert result.kospi_flow.foreign_net == 0
