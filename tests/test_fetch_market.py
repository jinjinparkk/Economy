"""fetch_market 모듈 단위 테스트 — market_breadth / sector_breadth."""
from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from src.fetch_market import MarketSnapshot


def _make_snapshot(kospi_rows, kosdaq_rows=None, indices=None):
    """테스트용 MarketSnapshot 생성."""
    cols = ["Code", "Name", "Close", "ChangeRatio", "Volume", "Amount", "Marcap", "Industry"]
    kospi = pd.DataFrame(kospi_rows, columns=cols) if kospi_rows else pd.DataFrame(columns=cols)
    kosdaq = pd.DataFrame(kosdaq_rows, columns=cols) if kosdaq_rows else pd.DataFrame(columns=cols)
    return MarketSnapshot(
        trade_date=date(2026, 4, 13),
        kospi=kospi,
        kosdaq=kosdaq,
        indices=indices or {},
    )


class TestMarketBreadth:
    def test_basic_counts(self):
        rows = [
            ["001", "A", 1000, 5.0, 100, 1e9, 1e10, "반도체"],
            ["002", "B", 2000, -3.0, 100, 1e9, 1e10, "반도체"],
            ["003", "C", 3000, 0.0, 100, 1e9, 1e10, "화학"],
        ]
        snap = _make_snapshot(rows)
        b = snap.market_breadth()
        assert b["total_up"] == 1
        assert b["total_down"] == 1
        assert b["total_unchanged"] == 1
        assert b["total"] == 3

    def test_all_up(self):
        rows = [
            ["001", "A", 1000, 5.0, 100, 1e9, 1e10, "반도체"],
            ["002", "B", 2000, 3.0, 100, 1e9, 1e10, "반도체"],
        ]
        snap = _make_snapshot(rows)
        b = snap.market_breadth()
        assert b["total_up"] == 2
        assert b["up_ratio"] == 100.0

    def test_empty(self):
        snap = _make_snapshot(None)
        b = snap.market_breadth()
        assert b["total_up"] == 0
        assert b["up_ratio"] == 0.0

    def test_combined_kospi_kosdaq(self):
        kospi = [["001", "A", 1000, 5.0, 100, 1e9, 1e10, "반도체"]]
        kosdaq = [["002", "B", 2000, -3.0, 100, 1e9, 1e10, "화학"]]
        snap = _make_snapshot(kospi, kosdaq)
        b = snap.market_breadth()
        assert b["total"] == 2
        assert b["total_up"] == 1
        assert b["total_down"] == 1


class TestSectorBreadth:
    def test_groups_by_industry(self):
        rows = [
            ["001", "A", 1000, 5.0, 100, 1e9, 1e10, "반도체"],
            ["002", "B", 2000, -3.0, 100, 1e9, 1e10, "반도체"],
            ["003", "C", 3000, 2.0, 100, 1e9, 1e10, "화학"],
        ]
        snap = _make_snapshot(rows)
        sec = snap.sector_breadth()
        assert "반도체" in sec
        assert sec["반도체"]["up_count"] == 1
        assert sec["반도체"]["down_count"] == 1
        assert sec["반도체"]["total"] == 2
        assert "화학" in sec
        assert sec["화학"]["up_count"] == 1

    def test_empty_industry_excluded(self):
        rows = [
            ["001", "A", 1000, 5.0, 100, 1e9, 1e10, ""],
        ]
        snap = _make_snapshot(rows)
        sec = snap.sector_breadth()
        assert len(sec) == 0

    def test_avg_change_pct(self):
        rows = [
            ["001", "A", 1000, 10.0, 100, 1e9, 1e10, "반도체"],
            ["002", "B", 2000, -2.0, 100, 1e9, 1e10, "반도체"],
        ]
        snap = _make_snapshot(rows)
        sec = snap.sector_breadth()
        assert sec["반도체"]["avg_change_pct"] == 4.0  # (10 + -2) / 2

    def test_empty_snapshot(self):
        snap = _make_snapshot(None)
        sec = snap.sector_breadth()
        assert len(sec) == 0
