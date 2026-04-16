"""기간(주간/월간/연간) 집계 데이터 수집 모듈.

Rolling N거래일 기준으로 매크로·종목·섹터·뉴스 스냅샷을 생성한다.
- weekly : 최근 5거래일
- monthly: 최근 21거래일
- yearly : 최근 252거래일

기존 fetch_macro / fetch_market / fetch_news 모듈의 자산·인벤토리를 재사용한다.
"""
from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Literal, Optional

import pandas as pd

from src.fetch_macro import (
    _ASIA,
    _BONDS,
    _BONDS_OPTIONAL,
    _COMMODITIES,
    _COMMODITIES_EXT,
    _CREDIT,
    _EUROPE,
    _FDR_TO_YF,
    _FX,
    _FX_OPTIONAL,
    _MEGA_CAPS,
    _SECTORS,
    _SEMI,
    _STYLE,
    _US_INDICES,
    _VOLATILITY,
)

# FDR 우선, 없으면 yfinance
try:
    import FinanceDataReader as fdr
    _USE_FDR = True
except ImportError:
    _USE_FDR = False
    try:
        import yfinance as yf
    except ImportError:  # pragma: no cover
        yf = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

Period = Literal["weekly", "monthly", "yearly"]

# 표시용 거래일 수 (실제 집계 trading_days에 사용됨 — 실데이터 기반 재계산)
_PERIOD_DAYS: dict[str, int] = {
    "weekly": 5,
    "monthly": 21,
    "yearly": 252,
}

# FDR 입력용 캘린더 일수 (주말·공휴일 여유 포함)
_PERIOD_CALENDAR: dict[str, int] = {
    "weekly": 10,
    "monthly": 35,
    "yearly": 380,
}

_PERIOD_LABELS: dict[str, str] = {
    "weekly": "주간",
    "monthly": "월간",
    "yearly": "연간",
}

# 뉴스 쿼리 세트
_PERIOD_NEWS_QUERIES: dict[str, list[str]] = {
    "weekly": ["금주 증시", "이번 주 KOSPI", "주간 코스피 코스닥"],
    "monthly": ["이번 달 증시", "월간 리포트 KOSPI", "한달 코스닥"],
    "yearly": ["올해 증시 결산", "연간 리포트 KOSPI", "연말 증시 전망"],
}


# ═══════════════════════════════════════════════════════════════════════
# Dataclasses
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class MacroPeriodReturn:
    """거시 지표의 기간 수익률."""

    name: str
    code: str
    start_close: float
    end_close: float
    cumulative_return_pct: float    # (end-start)/start*100
    high: float
    low: float
    volatility: float               # 일간 수익률 표준편차(%)
    start_date: str
    end_date: str

    def __str__(self) -> str:
        return (f"{self.name}: {self.cumulative_return_pct:+.2f}% "
                f"(σ {self.volatility:.2f}%)")


@dataclass
class StockPeriodReturn:
    """개별 종목의 기간 수익률."""

    code: str
    name: str
    market: str                     # "KOSPI" | "KOSDAQ"
    industry: str
    start_close: float
    end_close: float
    cumulative_return_pct: float
    avg_amount_eok: float           # 기간 평균 거래대금(억)
    start_date: str
    end_date: str

    def __str__(self) -> str:
        return (f"{self.name}({self.code}, {self.market}): "
                f"{self.cumulative_return_pct:+.2f}% "
                f"[{self.industry or '미분류'}]")


@dataclass
class SectorPeriodReturn:
    """섹터의 기간 수익률."""

    name: str
    code: str
    kind: str                       # "us_etf" | "kr_industry"
    cumulative_return_pct: float
    rank: int                       # 전체 대비 순위 (1=최상위)

    def __str__(self) -> str:
        return f"[{self.kind}] {self.name}: {self.cumulative_return_pct:+.2f}% (rank {self.rank})"


@dataclass
class PeriodSnapshot:
    """기간 집계 스냅샷."""

    period: Period
    trading_days: int
    start_date: str
    end_date: str
    macro_returns: dict[str, MacroPeriodReturn] = field(default_factory=dict)
    us_sectors: list[SectorPeriodReturn] = field(default_factory=list)
    kr_sectors: list[SectorPeriodReturn] = field(default_factory=list)
    kospi_top: list[StockPeriodReturn] = field(default_factory=list)
    kospi_bottom: list[StockPeriodReturn] = field(default_factory=list)
    kosdaq_top: list[StockPeriodReturn] = field(default_factory=list)
    kosdaq_bottom: list[StockPeriodReturn] = field(default_factory=list)
    mag7_returns: list[StockPeriodReturn] = field(default_factory=list)
    news_headlines: list[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        return _PERIOD_LABELS[self.period]

    def is_empty(self) -> bool:
        return not self.macro_returns and not self.kospi_top

    def to_summary_dict(self) -> dict:
        """LLM 프롬프트 보조용 요약 딕셔너리."""
        return {
            "period": self.period,
            "label": self.label,
            "trading_days": self.trading_days,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "macro_count": len(self.macro_returns),
            "us_sector_count": len(self.us_sectors),
            "kr_sector_count": len(self.kr_sectors),
            "kospi_top_count": len(self.kospi_top),
            "kospi_bottom_count": len(self.kospi_bottom),
            "kosdaq_top_count": len(self.kosdaq_top),
            "kosdaq_bottom_count": len(self.kosdaq_bottom),
            "mag7_count": len(self.mag7_returns),
            "news_count": len(self.news_headlines),
        }


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════


def _load_history(code: str, days_calendar: int) -> Optional[pd.DataFrame]:
    """FDR 우선 + yfinance fallback으로 OHLCV 히스토리를 로드한다.

    실패 시 None (graceful)을 반환한다.
    """
    end = datetime.now().date()
    start = end - timedelta(days=days_calendar)

    if _USE_FDR:
        try:
            df = fdr.DataReader(code, start, end)
            if df is None or df.empty:
                return None
            return df
        except Exception as exc:
            logger.warning("history %s FDR failed: %s", code, exc)
            return None

    if yf is None:  # pragma: no cover
        logger.error("history %s: neither FinanceDataReader nor yfinance available", code)
        return None

    yf_code = _FDR_TO_YF.get(code, code)
    try:
        df = yf.download(yf_code, start=str(start), end=str(end),
                         progress=False, auto_adjust=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if df is None or df.empty:
            return None
        return df
    except Exception as exc:
        logger.warning("history %s (yf:%s) failed: %s", code, yf_code, exc)
        return None


def _df_to_macro_return(
    name: str, code: str, df: pd.DataFrame,
) -> Optional[MacroPeriodReturn]:
    """DataFrame → MacroPeriodReturn (누적 수익률·변동성 계산)."""
    if df is None or df.empty or len(df) < 2:
        return None

    closes = df["Close"].astype(float)
    # NaN 제거
    closes = closes.dropna()
    if len(closes) < 2:
        return None

    start_close = float(closes.iloc[0])
    end_close = float(closes.iloc[-1])
    if start_close <= 0:
        return None

    cumulative_return_pct = round((end_close - start_close) / start_close * 100, 2)

    high = float(closes.max())
    low = float(closes.min())

    # 일간 수익률 표준편차(%)
    daily_returns = closes.pct_change().dropna() * 100.0
    if len(daily_returns) >= 2:
        volatility = round(float(daily_returns.std()), 3)
    else:
        volatility = 0.0

    start_date = str(closes.index[0].date()) if hasattr(closes.index[0], "date") else str(closes.index[0])
    end_date = str(closes.index[-1].date()) if hasattr(closes.index[-1], "date") else str(closes.index[-1])

    return MacroPeriodReturn(
        name=name,
        code=code,
        start_close=round(start_close, 4),
        end_close=round(end_close, 4),
        cumulative_return_pct=cumulative_return_pct,
        high=round(high, 4),
        low=round(low, 4),
        volatility=volatility,
        start_date=start_date,
        end_date=end_date,
    )


# ═══════════════════════════════════════════════════════════════════════
# 거시 지표 수집
# ═══════════════════════════════════════════════════════════════════════


def _iter_macro_catalog():
    """(name, code) 페어를 카탈로그 전체에서 순회한다."""
    for group in (_US_INDICES, _SEMI, _FX, _FX_OPTIONAL, _COMMODITIES,
                  _COMMODITIES_EXT, _VOLATILITY, _BONDS, _BONDS_OPTIONAL,
                  _SECTORS, _MEGA_CAPS, _STYLE, _ASIA, _EUROPE, _CREDIT):
        for key, (name, code) in group.items():
            yield key, name, code


def _fetch_macro_period_returns(days_calendar: int) -> dict[str, MacroPeriodReturn]:
    """모든 거시 지표의 기간 수익률을 수집한다."""
    result: dict[str, MacroPeriodReturn] = {}
    for key, name, code in _iter_macro_catalog():
        df = _load_history(code, days_calendar)
        if df is None:
            continue
        mr = _df_to_macro_return(name, code, df)
        if mr is not None:
            result[key] = mr
    logger.info("macro period returns: %d indicators", len(result))
    return result


def _fetch_us_sector_period_returns(
    macro_returns: dict[str, MacroPeriodReturn],
) -> list[SectorPeriodReturn]:
    """_SECTORS 11개 ETF를 수익률 내림차순으로 정렬하여 반환."""
    items: list[SectorPeriodReturn] = []
    sector_keys = set(_SECTORS.keys())
    pairs = [
        (key, mr) for key, mr in macro_returns.items()
        if key in sector_keys
    ]
    # 수익률 내림차순
    pairs.sort(key=lambda x: x[1].cumulative_return_pct, reverse=True)
    for rank, (key, mr) in enumerate(pairs, start=1):
        items.append(SectorPeriodReturn(
            name=mr.name,
            code=mr.code,
            kind="us_etf",
            cumulative_return_pct=mr.cumulative_return_pct,
            rank=rank,
        ))
    return items


# ═══════════════════════════════════════════════════════════════════════
# 개별 종목 수집
# ═══════════════════════════════════════════════════════════════════════


def _compute_stock_period_return(
    code: str, name: str, market: str, industry: str,
    days_calendar: int, *, min_avg_amount_eok: float = 5.0,
) -> Optional[StockPeriodReturn]:
    """단일 종목의 기간 수익률 계산. 거래대금 필터 미달 시 None."""
    df = _load_history(code, days_calendar)
    if df is None or df.empty or len(df) < 2:
        return None

    closes = df["Close"].astype(float).dropna()
    if len(closes) < 2:
        return None

    start_close = float(closes.iloc[0])
    end_close = float(closes.iloc[-1])
    if start_close <= 0:
        return None

    cumulative_return_pct = round((end_close - start_close) / start_close * 100, 2)

    # 거래대금(기간 평균, 억 원)
    if "Volume" in df.columns:
        # amount는 없을 수 있으므로 Close * Volume로 근사
        approx_amount = (df["Close"].astype(float) * df["Volume"].astype(float)).dropna()
        avg_amount_eok = round(float(approx_amount.mean()) / 1e8, 2) if len(approx_amount) else 0.0
    else:
        avg_amount_eok = 0.0

    if avg_amount_eok < min_avg_amount_eok:
        return None

    start_date = str(closes.index[0].date()) if hasattr(closes.index[0], "date") else str(closes.index[0])
    end_date = str(closes.index[-1].date()) if hasattr(closes.index[-1], "date") else str(closes.index[-1])

    return StockPeriodReturn(
        code=code,
        name=name,
        market=market,
        industry=industry or "",
        start_close=round(start_close, 2),
        end_close=round(end_close, 2),
        cumulative_return_pct=cumulative_return_pct,
        avg_amount_eok=avg_amount_eok,
        start_date=start_date,
        end_date=end_date,
    )


def _fetch_stock_period_returns(
    market: str,
    days_calendar: int,
    market_df: pd.DataFrame,
    *,
    top_n: int = 10,
    min_avg_amount_eok: float = 5.0,
    max_candidates: int = 200,
) -> tuple[list[StockPeriodReturn], list[StockPeriodReturn], list[StockPeriodReturn]]:
    """시총 상위 종목 중 기간 수익률을 계산해 TOP/BOTTOM/전체를 반환한다.

    Returns:
        (top_n_list, bottom_n_list, all_list)
    """
    if market_df is None or market_df.empty:
        return [], [], []

    df = market_df.copy()
    # 시가총액 내림차순 상위 N개
    if "Marcap" in df.columns:
        df = df.sort_values("Marcap", ascending=False, na_position="last")
    candidates = df.head(max_candidates)

    all_returns: list[StockPeriodReturn] = []
    for _, row in candidates.iterrows():
        code = str(row.get("Code", "")).strip()
        name = str(row.get("Name", "")).strip()
        industry = str(row.get("Industry", "")).strip()
        if not code or not name:
            continue
        try:
            result = _compute_stock_period_return(
                code, name, market, industry, days_calendar,
                min_avg_amount_eok=min_avg_amount_eok,
            )
        except Exception as exc:  # pragma: no cover
            logger.debug("stock %s(%s) failed: %s", name, code, str(exc)[:80])
            continue
        if result is not None:
            all_returns.append(result)

    # 수익률 내림차순 정렬
    all_returns.sort(key=lambda x: x.cumulative_return_pct, reverse=True)
    top = all_returns[:top_n]
    bottom = list(reversed(all_returns[-top_n:])) if len(all_returns) >= top_n else []
    logger.info("%s period returns: %d total, top=%d, bottom=%d",
                market, len(all_returns), len(top), len(bottom))
    return top, bottom, all_returns


def _fetch_kr_sector_period_returns(
    all_stock_returns: list[StockPeriodReturn],
    *,
    min_members: int = 3,
) -> list[SectorPeriodReturn]:
    """업종별 평균 수익률을 계산해 정렬된 리스트 반환."""
    if not all_stock_returns:
        return []

    by_industry: dict[str, list[float]] = {}
    for s in all_stock_returns:
        ind = (s.industry or "").strip()
        if not ind:
            continue
        by_industry.setdefault(ind, []).append(s.cumulative_return_pct)

    pairs: list[tuple[str, float]] = []
    for ind, rets in by_industry.items():
        if len(rets) < min_members:
            continue
        avg = round(sum(rets) / len(rets), 2)
        pairs.append((ind, avg))

    pairs.sort(key=lambda x: x[1], reverse=True)
    items: list[SectorPeriodReturn] = []
    for rank, (ind, avg) in enumerate(pairs, start=1):
        items.append(SectorPeriodReturn(
            name=ind,
            code=ind,
            kind="kr_industry",
            cumulative_return_pct=avg,
            rank=rank,
        ))
    logger.info("kr sector period returns: %d industries", len(items))
    return items


def _fetch_mag7_period_returns(days_calendar: int) -> list[StockPeriodReturn]:
    """Magnificent 7 개별 종목의 기간 수익률."""
    results: list[StockPeriodReturn] = []
    for key, (name, code) in _MEGA_CAPS.items():
        df = _load_history(code, days_calendar)
        if df is None or df.empty or len(df) < 2:
            continue
        closes = df["Close"].astype(float).dropna()
        if len(closes) < 2:
            continue
        start_close = float(closes.iloc[0])
        end_close = float(closes.iloc[-1])
        if start_close <= 0:
            continue
        cum = round((end_close - start_close) / start_close * 100, 2)
        start_date = str(closes.index[0].date()) if hasattr(closes.index[0], "date") else str(closes.index[0])
        end_date = str(closes.index[-1].date()) if hasattr(closes.index[-1], "date") else str(closes.index[-1])
        results.append(StockPeriodReturn(
            code=code,
            name=name,
            market="US",
            industry="Mag7",
            start_close=round(start_close, 2),
            end_close=round(end_close, 2),
            cumulative_return_pct=cum,
            avg_amount_eok=0.0,
            start_date=start_date,
            end_date=end_date,
        ))
    results.sort(key=lambda x: x.cumulative_return_pct, reverse=True)
    return results


# ═══════════════════════════════════════════════════════════════════════
# 뉴스 수집
# ═══════════════════════════════════════════════════════════════════════


def _fetch_period_news(period: Period, *, per_query: int = 5, max_total: int = 15) -> list[str]:
    """기간별 주요 뉴스 헤드라인 리스트."""
    from src.fetch_news import _fetch_via_google_rss

    queries = _PERIOD_NEWS_QUERIES.get(period, [])
    seen_titles: set[str] = set()
    headlines: list[str] = []
    for q in queries:
        items = _fetch_via_google_rss(q, display=per_query)
        for item in items:
            if item.title in seen_titles:
                continue
            seen_titles.add(item.title)
            tag = f"[{item.press}]" if item.press and item.press != "unknown" else ""
            headlines.append(f"{tag} {item.title}".strip())
            if len(headlines) >= max_total:
                return headlines
    return headlines


# ═══════════════════════════════════════════════════════════════════════
# 진입점
# ═══════════════════════════════════════════════════════════════════════


def fetch_period_snapshot(
    period: Period,
    *,
    market_snapshot: Optional[object] = None,
    include_news: bool = True,
) -> PeriodSnapshot:
    """기간 스냅샷을 수집한다.

    Args:
        period: "weekly" | "monthly" | "yearly"
        market_snapshot: 주입된 MarketSnapshot (테스트/재사용용). None이면 fetch.
        include_news: Google News 호출 여부.
    """
    if period not in _PERIOD_DAYS:
        raise ValueError(f"unknown period: {period!r}")

    days_calendar = _PERIOD_CALENDAR[period]
    logger.info("fetching %s snapshot (calendar_days=%d)", period, days_calendar)

    # 1) 거시 지표
    macro_returns = _fetch_macro_period_returns(days_calendar)

    # 2) US 섹터
    us_sectors = _fetch_us_sector_period_returns(macro_returns)

    # 3) 한국 시장 스냅샷 확보
    if market_snapshot is None:
        from src.fetch_market import fetch_market_snapshot
        market_snapshot = fetch_market_snapshot()

    kospi_df = getattr(market_snapshot, "kospi", pd.DataFrame())
    kosdaq_df = getattr(market_snapshot, "kosdaq", pd.DataFrame())

    # 4) KOSPI / KOSDAQ 종목별 수익률
    kospi_top, kospi_bottom, kospi_all = _fetch_stock_period_returns(
        "KOSPI", days_calendar, kospi_df,
    )
    kosdaq_top, kosdaq_bottom, kosdaq_all = _fetch_stock_period_returns(
        "KOSDAQ", days_calendar, kosdaq_df,
    )

    # 5) 한국 업종 평균 (KOSPI+KOSDAQ 합산 결과 기반)
    kr_sectors = _fetch_kr_sector_period_returns(kospi_all + kosdaq_all)

    # 6) Mag7 수익률 (주간은 생략 — 빌더 측에서 표시 여부 제어)
    mag7_returns = _fetch_mag7_period_returns(days_calendar) if period != "weekly" else []

    # 7) 뉴스
    news = _fetch_period_news(period) if include_news else []

    # 8) 기간 실데이터 기반 집계 정보 (거시 지표의 첫 지표 기준 사용)
    trading_days = _PERIOD_DAYS[period]
    start_date = ""
    end_date = ""
    if macro_returns:
        any_mr = next(iter(macro_returns.values()))
        start_date = any_mr.start_date
        end_date = any_mr.end_date

    snap = PeriodSnapshot(
        period=period,
        trading_days=trading_days,
        start_date=start_date,
        end_date=end_date,
        macro_returns=macro_returns,
        us_sectors=us_sectors,
        kr_sectors=kr_sectors,
        kospi_top=kospi_top,
        kospi_bottom=kospi_bottom,
        kosdaq_top=kosdaq_top,
        kosdaq_bottom=kosdaq_bottom,
        mag7_returns=mag7_returns,
        news_headlines=news,
    )
    logger.info("period snapshot ready: %s (%d→%d, macro=%d, kr_sec=%d, top=%d+%d)",
                period, len(kospi_all), len(kosdaq_all),
                len(macro_returns), len(kr_sectors),
                len(kospi_top), len(kosdaq_top))
    return snap
