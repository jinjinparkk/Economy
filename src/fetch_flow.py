"""투자자별 수급 데이터 수집 모듈 (pykrx 기반).

외국인/기관/개인 순매수 금액 + 종목별 순매수 상위를 조회한다.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

try:
    from pykrx import stock as krx_stock
    _HAS_PYKRX = True
except ImportError:
    krx_stock = None  # type: ignore[assignment]
    _HAS_PYKRX = False


@dataclass
class InvestorFlow:
    """시장 전체 투자자별 순매수 금액 (원)."""
    date: str
    foreign_net: int
    institution_net: int
    individual_net: int
    foreign_consecutive: int = 0   # +매수연속/-매도연속 일수


@dataclass
class TopFlowStock:
    """종목별 순매수 상위."""
    code: str
    name: str
    net_amount: int          # 순매수 금액 (원)
    investor_type: str       # "외국인" | "기관"


@dataclass
class FlowSnapshot:
    """수급 스냅샷."""
    trade_date: str
    kospi_flow: InvestorFlow
    kosdaq_flow: InvestorFlow
    foreign_top_buy: list[TopFlowStock] = field(default_factory=list)
    foreign_top_sell: list[TopFlowStock] = field(default_factory=list)
    institution_top_buy: list[TopFlowStock] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "trade_date": self.trade_date,
            "kospi_flow": {
                "foreign_net": self.kospi_flow.foreign_net,
                "institution_net": self.kospi_flow.institution_net,
                "individual_net": self.kospi_flow.individual_net,
                "foreign_consecutive": self.kospi_flow.foreign_consecutive,
            },
            "kosdaq_flow": {
                "foreign_net": self.kosdaq_flow.foreign_net,
                "institution_net": self.kosdaq_flow.institution_net,
                "individual_net": self.kosdaq_flow.individual_net,
                "foreign_consecutive": self.kosdaq_flow.foreign_consecutive,
            },
            "foreign_top_buy": [
                {"code": s.code, "name": s.name, "net_amount": s.net_amount}
                for s in self.foreign_top_buy
            ],
            "foreign_top_sell": [
                {"code": s.code, "name": s.name, "net_amount": s.net_amount}
                for s in self.foreign_top_sell
            ],
            "institution_top_buy": [
                {"code": s.code, "name": s.name, "net_amount": s.net_amount}
                for s in self.institution_top_buy
            ],
        }

    def to_narrative(self) -> str:
        parts: list[str] = []
        kf = self.kospi_flow
        parts.append(
            f"KOSPI 외국인 {kf.foreign_net/1e8:+,.0f}억, "
            f"기관 {kf.institution_net/1e8:+,.0f}억, "
            f"개인 {kf.individual_net/1e8:+,.0f}억"
        )
        if kf.foreign_consecutive != 0:
            direction = "매수" if kf.foreign_consecutive > 0 else "매도"
            parts.append(f"(외국인 {abs(kf.foreign_consecutive)}일 연속 {direction})")

        if self.foreign_top_buy:
            names = ", ".join(s.name for s in self.foreign_top_buy[:3])
            parts.append(f"외국인 순매수 TOP: {names}")

        return ". ".join(parts)


def _count_consecutive(daily_values: list[int]) -> int:
    """최근부터 부호가 같은 연속 일수를 센다.

    양수 연속이면 +N, 음수 연속이면 -N.
    """
    if not daily_values:
        return 0
    latest = daily_values[-1]
    if latest == 0:
        return 0
    sign = 1 if latest > 0 else -1
    count = 0
    for v in reversed(daily_values):
        if (v > 0 and sign > 0) or (v < 0 and sign < 0):
            count += 1
        else:
            break
    return count * sign


def _fetch_investor_flow(
    market: str, trade_date: str,
) -> InvestorFlow | None:
    """pykrx를 사용해 투자자별 순매수 금액을 조회한다.

    Args:
        market: "KOSPI" | "KOSDAQ"
        trade_date: "YYYYMMDD"
    """
    if not _HAS_PYKRX:
        logger.warning("pykrx not installed — skipping investor flow")
        return None

    try:
        # 최근 20일 조회 (연속 매수/매도 카운트용)
        end_dt = datetime.strptime(trade_date, "%Y%m%d")
        start_dt = end_dt - timedelta(days=35)  # 주말 포함 여유
        start_str = start_dt.strftime("%Y%m%d")

        df = krx_stock.get_market_trading_value_by_date(
            start_str, trade_date, market,
        )

        if df.empty:
            logger.warning("investor flow empty for %s %s", market, trade_date)
            return None

        # 컬럼: 기관합계, 기타법인, 개인, 외국인합계, 전체
        # 금액 단위: 원
        latest = df.iloc[-1]
        foreign_net = int(latest.get("외국인합계", 0))
        institution_net = int(latest.get("기관합계", 0))
        individual_net = int(latest.get("개인", 0))

        # 연속 매수/매도 카운트
        foreign_daily = [int(row.get("외국인합계", 0)) for _, row in df.iterrows()]
        consecutive = _count_consecutive(foreign_daily)

        return InvestorFlow(
            date=trade_date,
            foreign_net=foreign_net,
            institution_net=institution_net,
            individual_net=individual_net,
            foreign_consecutive=consecutive,
        )
    except Exception as exc:
        logger.warning("investor flow fetch failed for %s: %s", market, str(exc)[:200])
        return None


def _fetch_top_stocks(
    trade_date: str, investor: str, market: str, top_n: int = 5,
) -> tuple[list[TopFlowStock], list[TopFlowStock]]:
    """종목별 순매수 상위/하위를 조회한다.

    Returns:
        (top_buy, top_sell)
    """
    if not _HAS_PYKRX:
        return [], []

    try:
        df = krx_stock.get_market_net_purchases_of_equities_by_ticker(
            trade_date, trade_date, market, investor,
        )
        if df.empty:
            return [], []

        # 컬럼: 종목명, 매도거래량, 매수거래량, 순매수거래량, 매도거래대금, 매수거래대금, 순매수거래대금
        col = "순매수거래대금"
        if col not in df.columns:
            # 컬럼명이 다를 수 있음
            for c in df.columns:
                if "순매수" in c and "대금" in c:
                    col = c
                    break

        if col not in df.columns:
            return [], []

        top_buy_df = df.nlargest(top_n, col)
        top_sell_df = df.nsmallest(top_n, col)

        name_col = "종목명" if "종목명" in df.columns else None

        def _to_flow_stock(row, code) -> TopFlowStock:
            name = str(row[name_col]) if name_col else str(code)
            return TopFlowStock(
                code=str(code),
                name=name,
                net_amount=int(row[col]),
                investor_type=investor,
            )

        top_buy = [_to_flow_stock(row, code) for code, row in top_buy_df.iterrows()]
        top_sell = [_to_flow_stock(row, code) for code, row in top_sell_df.iterrows()]
        return top_buy, top_sell

    except Exception as exc:
        logger.warning("top stocks fetch failed for %s %s: %s",
                        investor, market, str(exc)[:200])
        return [], []


def fetch_flow_snapshot(trade_date: str) -> FlowSnapshot | None:
    """투자자 수급 스냅샷을 수집한다.

    Args:
        trade_date: "YYYY-MM-DD" 형식

    Returns:
        FlowSnapshot 또는 None (실패 시)
    """
    if not _HAS_PYKRX:
        logger.warning("pykrx not installed — skipping flow snapshot")
        return None

    try:
        dt_str = trade_date.replace("-", "")

        kospi_flow = _fetch_investor_flow("KOSPI", dt_str)
        kosdaq_flow = _fetch_investor_flow("KOSDAQ", dt_str)

        if kospi_flow is None:
            kospi_flow = InvestorFlow(dt_str, 0, 0, 0)
        if kosdaq_flow is None:
            kosdaq_flow = InvestorFlow(dt_str, 0, 0, 0)

        # 외국인 종목별 순매수 상위 (KOSPI 기준)
        foreign_buy, foreign_sell = _fetch_top_stocks(dt_str, "외국인", "KOSPI")

        # 기관 종목별 순매수 상위 (KOSPI 기준)
        inst_buy, _ = _fetch_top_stocks(dt_str, "기관합계", "KOSPI")

        return FlowSnapshot(
            trade_date=trade_date,
            kospi_flow=kospi_flow,
            kosdaq_flow=kosdaq_flow,
            foreign_top_buy=foreign_buy,
            foreign_top_sell=foreign_sell,
            institution_top_buy=inst_buy,
        )
    except Exception as exc:
        logger.error("flow snapshot failed: %s", str(exc)[:300])
        return None
