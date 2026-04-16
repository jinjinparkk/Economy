"""fetch_history 모듈 단위 테스트."""
from __future__ import annotations

from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

from src.fetch_history import (
    MacroPeriodReturn,
    StockPeriodReturn,
    SectorPeriodReturn,
    PeriodSnapshot,
    _PERIOD_DAYS,
    _PERIOD_CALENDAR,
    _PERIOD_LABELS,
    _PERIOD_NEWS_QUERIES,
    _df_to_macro_return,
    _fetch_macro_period_returns,
    _fetch_us_sector_period_returns,
    _fetch_stock_period_returns,
    _fetch_kr_sector_period_returns,
    _fetch_period_news,
    _compute_stock_period_return,
    fetch_period_snapshot,
)


def _price_df(closes: list[float], volumes: list[int] | None = None) -> pd.DataFrame:
    """테스트용 일봉 DataFrame."""
    dates = pd.date_range("2026-04-01", periods=len(closes), freq="D")
    df = pd.DataFrame({"Close": closes}, index=dates)
    if volumes is not None:
        df["Volume"] = volumes
    else:
        df["Volume"] = [1_000_000] * len(closes)
    return df


# ═══════════════════════════════════════════════════════════════════════
# Dataclass 테스트
# ═══════════════════════════════════════════════════════════════════════

class TestMacroPeriodReturn:
    def test_fields(self):
        mr = MacroPeriodReturn(
            name="S&P 500", code="US500",
            start_close=5000.0, end_close=5100.0,
            cumulative_return_pct=2.0, high=5120.0, low=4980.0,
            volatility=1.5, start_date="2026-04-01", end_date="2026-04-08",
        )
        assert mr.name == "S&P 500"
        assert mr.cumulative_return_pct == 2.0
        assert mr.volatility == 1.5

    def test_str_repr(self):
        mr = MacroPeriodReturn(
            "나스닥", "IXIC", 1000, 1050, 5.0, 1060, 980, 2.1,
            "2026-04-01", "2026-04-08",
        )
        s = str(mr)
        assert "나스닥" in s
        assert "+5.00%" in s
        assert "σ 2.10" in s


class TestStockPeriodReturn:
    def test_fields(self):
        s = StockPeriodReturn(
            code="005930", name="삼성전자", market="KOSPI", industry="반도체",
            start_close=60000, end_close=63000,
            cumulative_return_pct=5.0, avg_amount_eok=1500.0,
            start_date="2026-04-01", end_date="2026-04-08",
        )
        assert s.code == "005930"
        assert s.market == "KOSPI"
        assert s.industry == "반도체"

    def test_str_repr(self):
        s = StockPeriodReturn(
            "005930", "삼성전자", "KOSPI", "반도체",
            60000, 63000, 5.0, 1500.0, "2026-04-01", "2026-04-08",
        )
        text = str(s)
        assert "삼성전자" in text
        assert "KOSPI" in text
        assert "반도체" in text
        assert "+5.00%" in text

    def test_str_empty_industry(self):
        s = StockPeriodReturn(
            "123456", "테스트", "KOSDAQ", "",
            1000, 1010, 1.0, 100.0, "2026-04-01", "2026-04-08",
        )
        assert "미분류" in str(s)


class TestSectorPeriodReturn:
    def test_fields(self):
        s = SectorPeriodReturn(
            name="기술", code="XLK", kind="us_etf",
            cumulative_return_pct=3.5, rank=1,
        )
        assert s.kind == "us_etf"
        assert s.rank == 1

    def test_str_repr(self):
        s = SectorPeriodReturn("반도체", "sem", "kr_industry", -2.0, 5)
        text = str(s)
        assert "반도체" in text
        assert "kr_industry" in text
        assert "-2.00%" in text
        assert "rank 5" in text


class TestPeriodSnapshot:
    def test_label(self):
        for p, lbl in _PERIOD_LABELS.items():
            snap = PeriodSnapshot(
                period=p, trading_days=_PERIOD_DAYS[p],
                start_date="2026-04-01", end_date="2026-04-08",
            )
            assert snap.label == lbl

    def test_is_empty_true(self):
        snap = PeriodSnapshot(
            period="weekly", trading_days=5,
            start_date="", end_date="",
        )
        assert snap.is_empty() is True

    def test_is_empty_false_with_macro(self):
        mr = MacroPeriodReturn("S&P", "US500", 1, 2, 100, 2, 1, 0, "a", "b")
        snap = PeriodSnapshot(
            period="weekly", trading_days=5,
            start_date="a", end_date="b",
            macro_returns={"SP500": mr},
        )
        assert snap.is_empty() is False

    def test_is_empty_false_with_kospi(self):
        s = StockPeriodReturn("005930", "삼성", "KOSPI", "반도체",
                              1000, 1100, 10, 100, "a", "b")
        snap = PeriodSnapshot(
            period="monthly", trading_days=21,
            start_date="a", end_date="b",
            kospi_top=[s],
        )
        assert snap.is_empty() is False

    def test_to_summary_dict(self):
        snap = PeriodSnapshot(
            period="weekly", trading_days=5,
            start_date="2026-04-01", end_date="2026-04-08",
        )
        d = snap.to_summary_dict()
        assert d["period"] == "weekly"
        assert d["label"] == "주간"
        assert d["trading_days"] == 5
        assert d["macro_count"] == 0
        assert d["kospi_top_count"] == 0

    def test_period_constants(self):
        assert _PERIOD_DAYS["weekly"] == 5
        assert _PERIOD_DAYS["monthly"] == 21
        assert _PERIOD_DAYS["yearly"] == 252
        assert _PERIOD_CALENDAR["weekly"] == 10
        assert _PERIOD_CALENDAR["monthly"] == 35
        assert _PERIOD_CALENDAR["yearly"] == 380


# ═══════════════════════════════════════════════════════════════════════
# _df_to_macro_return
# ═══════════════════════════════════════════════════════════════════════

class TestDfToMacroReturn:
    def test_normal(self):
        df = _price_df([100.0, 101.0, 102.0, 103.0, 105.0])
        mr = _df_to_macro_return("테스트", "TST", df)
        assert mr is not None
        assert mr.name == "테스트"
        assert mr.start_close == 100.0
        assert mr.end_close == 105.0
        assert mr.cumulative_return_pct == 5.0
        assert mr.high == 105.0
        assert mr.low == 100.0
        assert mr.volatility > 0

    def test_empty_df(self):
        assert _df_to_macro_return("X", "X", pd.DataFrame()) is None

    def test_single_row(self):
        df = _price_df([100.0])
        assert _df_to_macro_return("X", "X", df) is None

    def test_zero_start(self):
        df = _price_df([0.0, 100.0, 200.0])
        assert _df_to_macro_return("X", "X", df) is None


# ═══════════════════════════════════════════════════════════════════════
# _fetch_macro_period_returns (mock _load_history)
# ═══════════════════════════════════════════════════════════════════════

class TestFetchMacroPeriodReturns:
    @patch("src.fetch_history._load_history")
    def test_returns_dict(self, mock_load):
        mock_load.return_value = _price_df([100.0, 102.0, 103.0])
        result = _fetch_macro_period_returns(10)
        assert isinstance(result, dict)
        # 카탈로그에 있는 여러 지표가 수집됨
        assert len(result) > 0

    @patch("src.fetch_history._load_history")
    def test_all_failures_returns_empty(self, mock_load):
        mock_load.return_value = None
        result = _fetch_macro_period_returns(10)
        assert result == {}


# ═══════════════════════════════════════════════════════════════════════
# _fetch_us_sector_period_returns
# ═══════════════════════════════════════════════════════════════════════

class TestFetchUsSectorPeriodReturns:
    def test_rank_assignment(self):
        from src.fetch_macro import _SECTORS
        macro_returns = {}
        for i, (key, (name, code)) in enumerate(_SECTORS.items()):
            macro_returns[key] = MacroPeriodReturn(
                name=name, code=code,
                start_close=100, end_close=100 + i,
                cumulative_return_pct=float(i),
                high=100 + i, low=100,
                volatility=1.0,
                start_date="a", end_date="b",
            )
        sectors = _fetch_us_sector_period_returns(macro_returns)
        assert len(sectors) == len(_SECTORS)
        # 수익률 내림차순 정렬
        assert sectors[0].rank == 1
        for i in range(len(sectors) - 1):
            assert sectors[i].cumulative_return_pct >= sectors[i + 1].cumulative_return_pct
        # 모두 us_etf
        assert all(s.kind == "us_etf" for s in sectors)

    def test_empty_input(self):
        assert _fetch_us_sector_period_returns({}) == []


# ═══════════════════════════════════════════════════════════════════════
# _fetch_stock_period_returns
# ═══════════════════════════════════════════════════════════════════════

class TestFetchStockPeriodReturns:
    def test_empty_market_df(self):
        top, bottom, all_ret = _fetch_stock_period_returns("KOSPI", 10, pd.DataFrame())
        assert top == []
        assert bottom == []
        assert all_ret == []

    @patch("src.fetch_history._load_history")
    def test_filters_low_amount(self, mock_load):
        # Close=1, Volume=100 → 평균 거래대금 = 100원 → 5억 미만
        mock_load.return_value = _price_df([1.0, 1.1, 1.2], volumes=[100, 100, 100])
        market_df = pd.DataFrame([
            {"Code": "000001", "Name": "저거래", "Industry": "테스트", "Marcap": 1e11},
        ])
        top, bottom, all_ret = _fetch_stock_period_returns(
            "KOSPI", 10, market_df, min_avg_amount_eok=5.0,
        )
        assert all_ret == []

    @patch("src.fetch_history._load_history")
    def test_top_bottom_sort(self, mock_load):
        # 3개 종목, 거래대금 충분, 수익률 상이
        returns_map = {
            "A": _price_df([100.0, 110.0, 120.0], volumes=[10_000_000] * 3),
            "B": _price_df([100.0, 95.0, 90.0], volumes=[10_000_000] * 3),
            "C": _price_df([100.0, 102.0, 105.0], volumes=[10_000_000] * 3),
        }
        mock_load.side_effect = lambda code, days: returns_map.get(code)
        market_df = pd.DataFrame([
            {"Code": "A", "Name": "알파", "Industry": "IT", "Marcap": 1e12},
            {"Code": "B", "Name": "베타", "Industry": "자동차", "Marcap": 5e11},
            {"Code": "C", "Name": "감마", "Industry": "IT", "Marcap": 3e11},
        ])
        top, bottom, all_ret = _fetch_stock_period_returns(
            "KOSPI", 10, market_df, top_n=2, min_avg_amount_eok=1.0,
        )
        assert len(all_ret) == 3
        # 수익률 내림차순
        assert all_ret[0].name == "알파"
        assert all_ret[-1].name == "베타"
        # top_n=2
        assert len(top) == 2

    @patch("src.fetch_history._load_history")
    def test_max_candidates_limit(self, mock_load):
        mock_load.return_value = _price_df([100.0, 101.0, 102.0], volumes=[10_000_000] * 3)
        # 250개 종목
        rows = [
            {"Code": f"{i:06d}", "Name": f"종목{i}", "Industry": "IT", "Marcap": 1e12 - i}
            for i in range(250)
        ]
        market_df = pd.DataFrame(rows)
        top, bottom, all_ret = _fetch_stock_period_returns(
            "KOSPI", 10, market_df, max_candidates=50, min_avg_amount_eok=1.0,
        )
        # 최대 50개만 시도
        assert len(all_ret) <= 50


# ═══════════════════════════════════════════════════════════════════════
# _fetch_kr_sector_period_returns
# ═══════════════════════════════════════════════════════════════════════

class TestFetchKrSectorPeriodReturns:
    def test_empty(self):
        assert _fetch_kr_sector_period_returns([]) == []

    def test_min_members_filter(self):
        # 업종 A: 2종목 → 제외 (min=3), 업종 B: 3종목 → 포함
        stocks = [
            StockPeriodReturn("1", "n1", "KOSPI", "A", 1, 2, 10, 100, "x", "y"),
            StockPeriodReturn("2", "n2", "KOSPI", "A", 1, 2, 20, 100, "x", "y"),
            StockPeriodReturn("3", "n3", "KOSPI", "B", 1, 2, 1, 100, "x", "y"),
            StockPeriodReturn("4", "n4", "KOSPI", "B", 1, 2, 2, 100, "x", "y"),
            StockPeriodReturn("5", "n5", "KOSPI", "B", 1, 2, 3, 100, "x", "y"),
        ]
        sectors = _fetch_kr_sector_period_returns(stocks, min_members=3)
        assert len(sectors) == 1
        assert sectors[0].name == "B"
        assert sectors[0].kind == "kr_industry"
        # 평균 (1+2+3)/3 = 2.0
        assert sectors[0].cumulative_return_pct == 2.0

    def test_sort_by_return(self):
        stocks = [
            StockPeriodReturn("1", "n1", "KOSPI", "저조", 1, 2, -5, 100, "x", "y"),
            StockPeriodReturn("2", "n2", "KOSPI", "저조", 1, 2, -3, 100, "x", "y"),
            StockPeriodReturn("3", "n3", "KOSPI", "저조", 1, 2, -4, 100, "x", "y"),
            StockPeriodReturn("4", "n4", "KOSPI", "상승", 1, 2, 10, 100, "x", "y"),
            StockPeriodReturn("5", "n5", "KOSPI", "상승", 1, 2, 12, 100, "x", "y"),
            StockPeriodReturn("6", "n6", "KOSPI", "상승", 1, 2, 14, 100, "x", "y"),
        ]
        sectors = _fetch_kr_sector_period_returns(stocks, min_members=3)
        assert len(sectors) == 2
        assert sectors[0].name == "상승"
        assert sectors[0].rank == 1
        assert sectors[1].name == "저조"
        assert sectors[1].rank == 2

    def test_skips_empty_industry(self):
        stocks = [
            StockPeriodReturn("1", "n1", "KOSPI", "", 1, 2, 10, 100, "x", "y"),
            StockPeriodReturn("2", "n2", "KOSPI", "", 1, 2, 20, 100, "x", "y"),
            StockPeriodReturn("3", "n3", "KOSPI", "", 1, 2, 30, 100, "x", "y"),
        ]
        assert _fetch_kr_sector_period_returns(stocks, min_members=3) == []


# ═══════════════════════════════════════════════════════════════════════
# _fetch_period_news
# ═══════════════════════════════════════════════════════════════════════

class TestFetchPeriodNews:
    @patch("src.fetch_news._fetch_via_google_rss")
    def test_weekly_queries(self, mock_rss):
        mock_rss.return_value = []
        _fetch_period_news("weekly")
        called_queries = [c.args[0] for c in mock_rss.call_args_list]
        assert any("금주" in q or "이번 주" in q or "주간" in q for q in called_queries)

    @patch("src.fetch_news._fetch_via_google_rss")
    def test_monthly_queries(self, mock_rss):
        mock_rss.return_value = []
        _fetch_period_news("monthly")
        called_queries = [c.args[0] for c in mock_rss.call_args_list]
        assert any("이번 달" in q or "월간" in q or "한달" in q for q in called_queries)

    @patch("src.fetch_news._fetch_via_google_rss")
    def test_yearly_queries(self, mock_rss):
        mock_rss.return_value = []
        _fetch_period_news("yearly")
        called_queries = [c.args[0] for c in mock_rss.call_args_list]
        assert any("올해" in q or "연간" in q or "연말" in q for q in called_queries)

    @patch("src.fetch_news._fetch_via_google_rss")
    def test_dedup(self, mock_rss):
        from src.fetch_news import NewsItem
        # 동일 제목 반복 → 1개로 축소
        mock_rss.return_value = [NewsItem("같은제목", "", "", "KBS")] * 5
        headlines = _fetch_period_news("weekly")
        assert len(headlines) == 1


# ═══════════════════════════════════════════════════════════════════════
# fetch_period_snapshot (통합)
# ═══════════════════════════════════════════════════════════════════════

class TestFetchPeriodSnapshot:
    @patch("src.fetch_history._fetch_period_news")
    @patch("src.fetch_history._fetch_mag7_period_returns")
    @patch("src.fetch_history._fetch_stock_period_returns")
    @patch("src.fetch_history._fetch_macro_period_returns")
    def test_weekly(self, mock_macro, mock_stock, mock_mag7, mock_news):
        mock_macro.return_value = {
            "SP500": MacroPeriodReturn("S&P", "US500", 5000, 5100, 2.0, 5120, 4950, 1.5,
                                       "2026-04-01", "2026-04-08"),
        }
        mock_stock.return_value = ([], [], [])
        mock_mag7.return_value = []
        mock_news.return_value = ["[KBS] 주간 헤드라인"]

        market_snap = MagicMock()
        market_snap.kospi = pd.DataFrame()
        market_snap.kosdaq = pd.DataFrame()

        snap = fetch_period_snapshot("weekly", market_snapshot=market_snap)
        assert snap.period == "weekly"
        assert snap.trading_days == 5
        assert len(snap.macro_returns) == 1
        # 주간은 mag7 호출 안 함
        mock_mag7.assert_not_called()

    @patch("src.fetch_history._fetch_period_news")
    @patch("src.fetch_history._fetch_mag7_period_returns")
    @patch("src.fetch_history._fetch_stock_period_returns")
    @patch("src.fetch_history._fetch_macro_period_returns")
    def test_monthly_includes_mag7(self, mock_macro, mock_stock, mock_mag7, mock_news):
        mock_macro.return_value = {}
        mock_stock.return_value = ([], [], [])
        mock_mag7.return_value = [
            StockPeriodReturn("NVDA", "엔비디아", "US", "Mag7",
                              900, 1000, 11.11, 0, "a", "b"),
        ]
        mock_news.return_value = []

        market_snap = MagicMock()
        market_snap.kospi = pd.DataFrame()
        market_snap.kosdaq = pd.DataFrame()

        snap = fetch_period_snapshot("monthly", market_snapshot=market_snap)
        assert snap.trading_days == 21
        assert len(snap.mag7_returns) == 1
        mock_mag7.assert_called_once()

    @patch("src.fetch_history._fetch_period_news")
    @patch("src.fetch_history._fetch_mag7_period_returns")
    @patch("src.fetch_history._fetch_stock_period_returns")
    @patch("src.fetch_history._fetch_macro_period_returns")
    def test_yearly(self, mock_macro, mock_stock, mock_mag7, mock_news):
        mock_macro.return_value = {}
        mock_stock.return_value = ([], [], [])
        mock_mag7.return_value = []
        mock_news.return_value = []

        market_snap = MagicMock()
        market_snap.kospi = pd.DataFrame()
        market_snap.kosdaq = pd.DataFrame()

        snap = fetch_period_snapshot("yearly", market_snapshot=market_snap)
        assert snap.trading_days == 252
        assert snap.period == "yearly"

    def test_unknown_period_raises(self):
        with pytest.raises(ValueError, match="unknown period"):
            fetch_period_snapshot("daily")  # type: ignore[arg-type]

    @patch("src.fetch_history._fetch_period_news")
    @patch("src.fetch_history._fetch_mag7_period_returns")
    @patch("src.fetch_history._fetch_stock_period_returns")
    @patch("src.fetch_history._fetch_macro_period_returns")
    def test_include_news_false_skips(self, mock_macro, mock_stock, mock_mag7, mock_news):
        mock_macro.return_value = {}
        mock_stock.return_value = ([], [], [])
        mock_mag7.return_value = []

        market_snap = MagicMock()
        market_snap.kospi = pd.DataFrame()
        market_snap.kosdaq = pd.DataFrame()

        snap = fetch_period_snapshot("weekly", market_snapshot=market_snap, include_news=False)
        assert snap.news_headlines == []
        mock_news.assert_not_called()
