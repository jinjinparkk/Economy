"""detect_movers 모듈 단위 테스트."""
from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from src.detect_movers import Mover, MoverReport, detect_movers, _enrich_with_hints
from src.fetch_market import MarketSnapshot


# ── 헬퍼: 가짜 MarketSnapshot 생성 ─────────────────────────────────────
def _make_snapshot(rows: list[dict], trade_date: str = "2026-04-09") -> MarketSnapshot:
    """rows 에서 KOSPI/KOSDAQ DataFrame을 만들어 MarketSnapshot 리턴."""
    if not rows:
        empty = pd.DataFrame(columns=["Code", "Name", "Close", "ChangeRatio",
                                       "Volume", "Amount", "Marcap", "Industry"])
        return MarketSnapshot(
            trade_date=date.fromisoformat(trade_date),
            kospi=empty, kosdaq=empty, indices={},
        )
    df = pd.DataFrame(rows)
    kospi = df[df["Market"] == "KOSPI"].drop(columns=["Market"]).reset_index(drop=True)
    kosdaq = df[df["Market"] == "KOSDAQ"].drop(columns=["Market"]).reset_index(drop=True)
    return MarketSnapshot(
        trade_date=date.fromisoformat(trade_date),
        kospi=kospi,
        kosdaq=kosdaq,
        indices={},
    )


def _row(code, name, market, change, close=5000, volume=100000, amount=2e9, marcap=1e11, industry="테스트업"):
    return {
        "Code": code, "Name": name, "Market": market,
        "Close": close, "ChangeRatio": change, "Volume": volume,
        "Amount": amount, "Marcap": marcap, "Industry": industry,
    }


# ── Mover 데이터클래스 ─────────────────────────────────────────────────
class TestMover:
    def test_is_limit_positive(self):
        m = Mover("000000", "T", "KOSPI", "surge", 10000, 30.0, 0, 0, None)
        assert m.is_limit

    def test_is_limit_negative(self):
        m = Mover("000000", "T", "KOSPI", "plunge", 10000, -29.5, 0, 0, None)
        assert m.is_limit

    def test_not_limit(self):
        m = Mover("000000", "T", "KOSPI", "surge", 10000, 20.0, 0, 0, None)
        assert not m.is_limit

    def test_str_surge(self):
        m = Mover("005930", "삼성전자", "KOSPI", "surge", 70000, 5.5, 1000000, 5e10, 4e14)
        s = str(m)
        assert "↑" in s
        assert "삼성전자" in s

    def test_str_plunge(self):
        m = Mover("005930", "삼성전자", "KOSPI", "plunge", 65000, -5.5, 1000000, 5e10, 4e14)
        s = str(m)
        assert "↓" in s


# ── MoverReport ────────────────────────────────────────────────────────
class TestMoverReport:
    def test_all_movers(self):
        m1 = Mover("A", "A", "KOSPI", "surge", 100, 10.0, 0, 0, None)
        m2 = Mover("B", "B", "KOSPI", "plunge", 100, -10.0, 0, 0, None)
        rpt = MoverReport("2026-04-09", [m1], [m2])
        assert len(rpt.all_movers) == 2


# ── _enrich_with_hints ────────────────────────────────────────────────
class TestEnrichWithHints:
    def test_limit_up(self):
        row = pd.Series({"ChangeRatio": 30.0, "Volume": 100, "Amount": 1e8})
        hints = _enrich_with_hints(row, "surge", 50)
        assert "상한가" in hints

    def test_limit_down(self):
        row = pd.Series({"ChangeRatio": -30.0, "Volume": 100, "Amount": 1e8})
        hints = _enrich_with_hints(row, "plunge", 50)
        assert "하한가" in hints

    def test_volume_surge(self):
        row = pd.Series({"ChangeRatio": 10.0, "Volume": 600, "Amount": 1e8})
        hints = _enrich_with_hints(row, "surge", 100)
        assert "거래량급증" in hints

    def test_volume_normal(self):
        row = pd.Series({"ChangeRatio": 10.0, "Volume": 200, "Amount": 1e8})
        hints = _enrich_with_hints(row, "surge", 100)
        assert "거래량급증" not in hints

    def test_amount_high(self):
        row = pd.Series({"ChangeRatio": 10.0, "Volume": 100, "Amount": 2e10})
        hints = _enrich_with_hints(row, "surge", 100)
        assert "거래대금상위" in hints


# ── detect_movers 통합 ────────────────────────────────────────────────
class TestDetectMovers:
    def test_basic_detection(self):
        rows = [
            _row("001", "급등A", "KOSPI", 15.0),
            _row("002", "급등B", "KOSDAQ", 10.0),
            _row("003", "보합C", "KOSPI", 2.0),
            _row("004", "급락D", "KOSPI", -12.0),
            _row("005", "급락E", "KOSDAQ", -7.0),
        ]
        snap = _make_snapshot(rows)
        rpt = detect_movers(snap, threshold_pct=5.0, top_n=5)
        assert len(rpt.surges) == 2
        assert len(rpt.plunges) == 2
        assert rpt.surges[0].change_pct >= rpt.surges[1].change_pct
        assert rpt.plunges[0].change_pct <= rpt.plunges[1].change_pct

    def test_top_n_limits(self):
        rows = [_row(f"0{i}", f"S{i}", "KOSPI", 10.0 + i) for i in range(10)]
        snap = _make_snapshot(rows)
        rpt = detect_movers(snap, threshold_pct=5.0, top_n=3)
        assert len(rpt.surges) == 3

    def test_min_amount_filter(self):
        rows = [
            _row("001", "소량", "KOSPI", 15.0, amount=5e8),   # 거래대금 5억 < 10억
            _row("002", "충분", "KOSPI", 12.0, amount=2e9),   # 거래대금 20억
        ]
        snap = _make_snapshot(rows)
        rpt = detect_movers(snap, threshold_pct=5.0, top_n=5, min_amount=1e9)
        assert len(rpt.surges) == 1
        assert rpt.surges[0].name == "충분"

    def test_empty_market(self):
        snap = _make_snapshot([])
        rpt = detect_movers(snap, threshold_pct=5.0, top_n=5)
        assert len(rpt.surges) == 0
        assert len(rpt.plunges) == 0

    def test_threshold_filters_correctly(self):
        rows = [
            _row("001", "A", "KOSPI", 4.9),
            _row("002", "B", "KOSPI", 5.1),
        ]
        snap = _make_snapshot(rows)
        rpt = detect_movers(snap, threshold_pct=5.0, top_n=5)
        assert len(rpt.surges) == 1
        assert rpt.surges[0].name == "B"

    def test_industry_included(self):
        rows = [_row("001", "A", "KOSPI", 10.0, industry="반도체 제조업")]
        snap = _make_snapshot(rows)
        rpt = detect_movers(snap, threshold_pct=5.0, top_n=5)
        assert rpt.surges[0].industry == "반도체 제조업"


# ── relative_strength + 시장역행 ─────────────────────────────────────
class TestRelativeStrength:
    def test_rs_with_index(self):
        """종목 +10%, 지수 +2% → RS +8.0"""
        rows = [_row("001", "A", "KOSPI", 10.0)]
        snap = _make_snapshot(rows)
        snap.indices["KOSPI"] = pd.Series({"Close": 2800, "ChangePct": 2.0, "Volume": 0, "Date": "2026-04-13"})
        rpt = detect_movers(snap, threshold_pct=5.0, top_n=5)
        assert rpt.surges[0].relative_strength == 8.0

    def test_rs_no_index(self):
        """지수 데이터 없으면 RS = 종목 등락률 그대로"""
        rows = [_row("001", "A", "KOSPI", 10.0)]
        snap = _make_snapshot(rows)
        rpt = detect_movers(snap, threshold_pct=5.0, top_n=5)
        assert rpt.surges[0].relative_strength == 10.0

    def test_contrarian_surge_in_down_market(self):
        """시장 -2%, 종목 +10% → 시장역행 hint"""
        rows = [_row("001", "A", "KOSPI", 10.0)]
        snap = _make_snapshot(rows)
        snap.indices["KOSPI"] = pd.Series({"Close": 2700, "ChangePct": -2.0, "Volume": 0, "Date": "2026-04-13"})
        rpt = detect_movers(snap, threshold_pct=5.0, top_n=5)
        assert "시장역행" in rpt.surges[0].reason_hints

    def test_contrarian_plunge_in_up_market(self):
        """시장 +3%, 종목 -10% → 시장역행 hint"""
        rows = [_row("001", "A", "KOSPI", -10.0)]
        snap = _make_snapshot(rows)
        snap.indices["KOSPI"] = pd.Series({"Close": 2900, "ChangePct": 3.0, "Volume": 0, "Date": "2026-04-13"})
        rpt = detect_movers(snap, threshold_pct=5.0, top_n=5)
        assert "시장역행" in rpt.plunges[0].reason_hints

    def test_no_contrarian_normal(self):
        """시장 +1%, 종목 +10% → 시장역행 없음"""
        rows = [_row("001", "A", "KOSPI", 10.0)]
        snap = _make_snapshot(rows)
        snap.indices["KOSPI"] = pd.Series({"Close": 2800, "ChangePct": 1.0, "Volume": 0, "Date": "2026-04-13"})
        rpt = detect_movers(snap, threshold_pct=5.0, top_n=5)
        assert "시장역행" not in rpt.surges[0].reason_hints
