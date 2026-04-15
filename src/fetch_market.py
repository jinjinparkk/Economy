"""시장 데이터 수집 모듈 (FinanceDataReader 기반).

전 종목 시세, 지수, 개별 종목 히스토리를 가져온다.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Literal

import FinanceDataReader as fdr
import pandas as pd

logger = logging.getLogger(__name__)

Market = Literal["KOSPI", "KOSDAQ"]

INDEX_CODES = {
    "KOSPI": "KS11",
    "KOSDAQ": "KQ11",
    "KOSPI200": "KS200",
}


@dataclass
class MarketSnapshot:
    """장 마감 시점의 전 종목 스냅샷."""

    trade_date: date
    kospi: pd.DataFrame  # Code, Name, Close, Changes, ChagesRatio, Volume, Amount, Marcap
    kosdaq: pd.DataFrame
    indices: dict[str, pd.Series]  # {"KOSPI": latest row, ...}

    def all_stocks(self) -> pd.DataFrame:
        """KOSPI + KOSDAQ 전 종목 합친 DataFrame."""
        k1 = self.kospi.copy()
        k1["Market"] = "KOSPI"
        k2 = self.kosdaq.copy()
        k2["Market"] = "KOSDAQ"
        return pd.concat([k1, k2], ignore_index=True)

    def market_breadth(self) -> dict:
        """전체 시장 등락 종목 수 + 상승 비율."""
        df = self.all_stocks()
        if df.empty:
            return {"total_up": 0, "total_down": 0, "total_unchanged": 0,
                    "total": 0, "up_ratio": 0.0}
        total_up = int((df["ChangeRatio"] > 0).sum())
        total_down = int((df["ChangeRatio"] < 0).sum())
        total_unchanged = int((df["ChangeRatio"] == 0).sum())
        total = len(df)
        up_ratio = round(total_up / total * 100, 1) if total > 0 else 0.0
        return {
            "total_up": total_up,
            "total_down": total_down,
            "total_unchanged": total_unchanged,
            "total": total,
            "up_ratio": up_ratio,
        }

    def sector_breadth(self) -> dict[str, dict]:
        """업종별 등락 종목 수 + 평균 등락률."""
        df = self.all_stocks()
        if df.empty or "Industry" not in df.columns:
            return {}
        df_ind = df[df["Industry"] != ""].copy()
        if df_ind.empty:
            return {}
        result: dict[str, dict] = {}
        for industry, group in df_ind.groupby("Industry"):
            up_count = int((group["ChangeRatio"] > 0).sum())
            down_count = int((group["ChangeRatio"] < 0).sum())
            avg_change = round(float(group["ChangeRatio"].mean()), 2)
            result[str(industry)] = {
                "up_count": up_count,
                "down_count": down_count,
                "total": len(group),
                "avg_change_pct": avg_change,
            }
        return result


def _normalize_listing(df: pd.DataFrame) -> pd.DataFrame:
    """FDR StockListing 응답에서 필요한 컬럼만 추려 표준화한다."""
    keep = ["Code", "Name", "Close", "Open", "High", "Low",
            "Changes", "ChagesRatio", "Volume", "Amount", "Marcap"]
    available = [c for c in keep if c in df.columns]
    out = df[available].copy()
    # 오타 교정: ChagesRatio → ChangeRatio
    if "ChagesRatio" in out.columns:
        out = out.rename(columns={"ChagesRatio": "ChangeRatio"})
    # 빈 값 제거 (상장폐지/거래정지 종목)
    out = out.dropna(subset=["Close"])
    out = out[out["Close"] > 0]
    return out.reset_index(drop=True)


def fetch_stock_listing(market: Market) -> pd.DataFrame:
    """KOSPI 또는 KOSDAQ 전 종목 현재 시세를 가져온다.

    Returns:
        DataFrame with columns: Code, Name, Close, Open, High, Low,
        Changes, ChangeRatio, Volume, Amount, Marcap
    """
    logger.info("fetching stock listing: market=%s", market)
    raw = fdr.StockListing(market)
    df = _normalize_listing(raw)
    logger.info("fetched %d stocks from %s", len(df), market)
    return df


def fetch_index_snapshot(days: int = 5) -> dict[str, pd.Series]:
    """주요 지수의 최근 N일 종가를 가져와 최신 Series만 리턴.

    Returns:
        {"KOSPI": pd.Series(Close, Change, ...), ...}
    """
    end = datetime.now().date()
    start = end - timedelta(days=days + 10)  # 주말/공휴일 여유

    result: dict[str, pd.Series] = {}
    for name, code in INDEX_CODES.items():
        try:
            df = fdr.DataReader(code, start, end)
            if df.empty:
                logger.warning("index %s returned empty", name)
                continue
            latest = df.iloc[-1]
            prev = df.iloc[-2] if len(df) >= 2 else None
            # 전일 대비 변화 계산
            close = round(float(latest["Close"]), 2)
            prev_close = round(float(prev["Close"]), 2) if prev is not None else close
            change = round(close - prev_close, 2)
            change_pct = round((change / prev_close * 100), 2) if prev_close else 0.0
            result[name] = pd.Series({
                "Close": close,
                "PrevClose": prev_close,
                "Change": change,
                "ChangePct": change_pct,
                "Volume": float(latest.get("Volume", 0)),
                "Date": df.index[-1].date(),
            })
            logger.info("index %s: %.2f (%+.2f%%)", name, close, change_pct)
        except Exception as exc:
            logger.error("failed to fetch index %s: %s", name, exc)
    return result


def fetch_ohlcv_history(code: str, days: int = 30) -> pd.DataFrame:
    """개별 종목의 OHLCV 히스토리.

    Args:
        code: 6자리 종목코드 (예: "005930")
        days: 캘린더 일수 (주말 포함)
    """
    end = datetime.now().date()
    start = end - timedelta(days=days)
    df = fdr.DataReader(code, start, end)
    return df


def _fetch_industry_map() -> dict[str, str]:
    """KRX-DESC에서 종목코드 → 업종 매핑을 가져온다."""
    try:
        desc = fdr.StockListing("KRX-DESC")
        mapping = {}
        for _, row in desc.iterrows():
            code = str(row.get("Code", ""))
            industry = str(row.get("Industry", ""))
            if code and industry and industry != "nan":
                mapping[code] = industry
        logger.info("industry map: %d entries loaded", len(mapping))
        return mapping
    except Exception as exc:
        logger.warning("failed to load KRX-DESC: %s", exc)
        return {}


def fetch_market_snapshot() -> MarketSnapshot:
    """장 마감 시점의 시장 전체 스냅샷."""
    kospi = fetch_stock_listing("KOSPI")
    kosdaq = fetch_stock_listing("KOSDAQ")
    indices = fetch_index_snapshot()

    # 업종 정보 조인
    industry_map = _fetch_industry_map()
    for df in (kospi, kosdaq):
        df["Industry"] = df["Code"].map(industry_map).fillna("")

    trade_date = datetime.now().date()
    if "KOSPI" in indices:
        trade_date = indices["KOSPI"]["Date"]

    return MarketSnapshot(
        trade_date=trade_date,
        kospi=kospi,
        kosdaq=kosdaq,
        indices=indices,
    )


if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    snapshot = fetch_market_snapshot()
    print(f"\n{'='*60}")
    print(f"장 마감 스냅샷 — {snapshot.trade_date}")
    print(f"{'='*60}")
    print(f"KOSPI 종목: {len(snapshot.kospi)}")
    print(f"KOSDAQ 종목: {len(snapshot.kosdaq)}")
    print(f"\n[주요 지수]")
    for name, s in snapshot.indices.items():
        print(f"  {name}: {s['Close']:,.2f} ({s['ChangePct']:+.2f}%)")
    print(f"\n[KOSPI 상위 5 등락률]")
    top = snapshot.kospi.nlargest(5, "ChangeRatio")[["Code", "Name", "Close", "ChangeRatio", "Volume"]]
    print(top.to_string(index=False))
