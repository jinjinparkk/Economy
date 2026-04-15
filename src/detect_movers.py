"""급등/급락 종목 탐지 모듈.

MarketSnapshot에서 의미있는 급등/급락 종목을 추려낸다.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

from src.fetch_market import MarketSnapshot

logger = logging.getLogger(__name__)

MoveType = Literal["surge", "plunge"]


@dataclass
class Mover:
    """급등 또는 급락 종목."""

    code: str
    name: str
    market: str  # "KOSPI" | "KOSDAQ"
    move_type: MoveType
    close: float
    change_pct: float
    volume: int
    amount: int  # 거래대금
    marcap: int | None  # 시가총액
    industry: str = ""  # 업종 (예: "반도체 제조업")
    reason_hints: list[str] = field(default_factory=list)  # "상한가", "거래량급증", ...
    relative_strength: float = 0.0  # 종목 등락률 - 시장 지수 등락률

    @property
    def is_limit(self) -> bool:
        """상한가/하한가 여부 (±29% 이상)."""
        return abs(self.change_pct) >= 29.0

    def __str__(self) -> str:
        arrow = "↑" if self.move_type == "surge" else "↓"
        rs_str = f" RS{self.relative_strength:+.1f}%p" if self.relative_strength else ""
        return (f"[{self.market}] {self.name}({self.code}) "
                f"{arrow}{self.change_pct:+.2f}%{rs_str} "
                f"종가 {self.close:,.0f}원 "
                f"거래대금 {self.amount/1e8:.0f}억")


@dataclass
class MoverReport:
    """탐지 결과 종합 리포트."""

    trade_date: str
    surges: list[Mover]
    plunges: list[Mover]

    @property
    def all_movers(self) -> list[Mover]:
        return self.surges + self.plunges


def _enrich_with_hints(row: pd.Series, move_type: MoveType,
                        volume_median: float) -> list[str]:
    """탐지된 mover에 추가 특징을 부여한다."""
    hints: list[str] = []
    change_pct = float(row["ChangeRatio"])
    volume = float(row["Volume"])

    # 상한가/하한가
    if change_pct >= 29.0:
        hints.append("상한가")
    elif change_pct <= -29.0:
        hints.append("하한가")

    # 거래량 급증 (median의 3배 이상)
    if volume_median > 0 and volume >= volume_median * 3:
        hints.append("거래량급증")

    # 거래대금 100억 이상 = 실질 거래
    if float(row.get("Amount", 0)) >= 1e10:
        hints.append("거래대금상위")

    return hints


def detect_movers(
    snapshot: MarketSnapshot,
    threshold_pct: float = 5.0,
    top_n: int = 5,
    min_amount: float = 1e9,  # 최소 거래대금 10억 (너무 한산한 종목 제외)
) -> MoverReport:
    """장 마감 스냅샷에서 급등/급락 종목 TOP N을 탐지한다.

    Args:
        snapshot: 시장 스냅샷
        threshold_pct: 최소 등락률 (%, 절댓값)
        top_n: 방향별 최대 개수
        min_amount: 최소 거래대금 (원) — 이보다 적으면 제외

    Returns:
        MoverReport — 급등 N개 + 급락 N개
    """
    df = snapshot.all_stocks()

    if df.empty:
        logger.info("detected 0 surges, 0 plunges (empty market)")
        return MoverReport(trade_date=str(snapshot.trade_date), surges=[], plunges=[])

    # 거래대금 필터
    df = df[df["Amount"] >= min_amount]

    if df.empty:
        logger.info("detected 0 surges, 0 plunges (all below min_amount)")
        return MoverReport(trade_date=str(snapshot.trade_date), surges=[], plunges=[])

    # 거래량 중앙값 (거래량 급증 판단용)
    vol_median = float(df["Volume"].median())

    # 시장 지수 등락률 (상대강도 계산용)
    index_change: dict[str, float] = {}
    for name in ("KOSPI", "KOSDAQ"):
        idx = snapshot.indices.get(name)
        if idx is not None:
            index_change[name] = float(idx["ChangePct"])
        else:
            index_change[name] = 0.0

    surges: list[Mover] = []
    plunges: list[Mover] = []

    def _to_mover(row: pd.Series, move_type: MoveType) -> Mover:
        market = str(row["Market"])
        change_pct = float(row["ChangeRatio"])
        mkt_change = index_change.get(market, 0.0)
        rel_strength = round(change_pct - mkt_change, 2)

        hints = _enrich_with_hints(row, move_type, vol_median)

        # 시장역행: 시장 하락인데 급등 or 시장 상승인데 급락
        if move_type == "surge" and mkt_change < -1.0:
            hints.append("시장역행")
        elif move_type == "plunge" and mkt_change > 1.0:
            hints.append("시장역행")

        return Mover(
            code=str(row["Code"]),
            name=str(row["Name"]),
            market=market,
            move_type=move_type,
            close=float(row["Close"]),
            change_pct=change_pct,
            volume=int(row["Volume"]),
            amount=int(row["Amount"]),
            marcap=int(row["Marcap"]) if pd.notna(row.get("Marcap")) else None,
            industry=str(row.get("Industry", "")) or "",
            reason_hints=hints,
            relative_strength=rel_strength,
        )

    # 급등
    up = df[df["ChangeRatio"] >= threshold_pct].nlargest(top_n, "ChangeRatio")
    for _, row in up.iterrows():
        surges.append(_to_mover(row, "surge"))

    # 급락
    down = df[df["ChangeRatio"] <= -threshold_pct].nsmallest(top_n, "ChangeRatio")
    for _, row in down.iterrows():
        plunges.append(_to_mover(row, "plunge"))

    logger.info("detected %d surges, %d plunges (threshold=%.1f%%)",
                len(surges), len(plunges), threshold_pct)

    return MoverReport(
        trade_date=str(snapshot.trade_date),
        surges=surges,
        plunges=plunges,
    )


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    from src.fetch_market import fetch_market_snapshot

    snap = fetch_market_snapshot()
    report = detect_movers(snap, threshold_pct=5.0, top_n=5)

    print(f"\n{'='*70}")
    print(f"급등/급락 리포트 — {report.trade_date}")
    print(f"{'='*70}\n")

    print(f"[급등 TOP {len(report.surges)}]")
    for m in report.surges:
        hints = f" ({', '.join(m.reason_hints)})" if m.reason_hints else ""
        print(f"  {m}{hints}")

    print(f"\n[급락 TOP {len(report.plunges)}]")
    for m in report.plunges:
        hints = f" ({', '.join(m.reason_hints)})" if m.reason_hints else ""
        print(f"  {m}{hints}")
