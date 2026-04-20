"""실적 시즌 트래커 모듈 (DART OpenAPI 기반).

주요 대형주 10개의 실적 발표 일정과 재무데이터를 추적한다.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import requests

from src.config import Config

logger = logging.getLogger(__name__)

# ── 추적 대상 종목 (종목명 → DART corp_code) ─────────────────────────
# corp_code는 DART 고유 8자리 코드
_TRACKED_CORPS: dict[str, str] = {
    "삼성전자": "00126380",
    "SK하이닉스": "00164779",
    "LG에너지솔루션": "01634089",
    "현대차": "00164742",
    "기아": "00270052",
    "POSCO홀딩스": "00367624",
    "삼성바이오로직스": "00751684",
    "NAVER": "00266961",
    "카카오": "00447073",
    "셀트리온": "00421045",
}

_DART_BASE = "https://opendart.fss.or.kr"


@dataclass
class EarningsEvent:
    """개별 실적 이벤트."""
    corp_code: str
    corp_name: str
    report_date: str             # 공시일 YYYY-MM-DD
    report_type: str             # "1Q" | "2Q" | "3Q" | "4Q"
    revenue: int | None = None
    operating_profit: int | None = None
    net_income: int | None = None
    revenue_yoy: float | None = None
    op_yoy: float | None = None
    d_day: int = 0               # 공시일까지 남은 일수 (0=당일, 음수=이미 발표)
    surprise: str | None = None  # "어닝서프라이즈" | "어닝쇼크" | "부합" | None


@dataclass
class EarningsSnapshot:
    """실적 스냅샷."""
    upcoming: list[EarningsEvent] = field(default_factory=list)  # 향후 2주 예정
    recent: list[EarningsEvent] = field(default_factory=list)    # 최근 1주 발표

    def to_dict(self) -> dict:
        def _event_dict(e: EarningsEvent) -> dict:
            return {
                "corp_name": e.corp_name,
                "report_date": e.report_date,
                "report_type": e.report_type,
                "revenue": e.revenue,
                "operating_profit": e.operating_profit,
                "net_income": e.net_income,
                "revenue_yoy": e.revenue_yoy,
                "op_yoy": e.op_yoy,
                "d_day": e.d_day,
                "surprise": e.surprise,
            }
        return {
            "upcoming": [_event_dict(e) for e in self.upcoming],
            "recent": [_event_dict(e) for e in self.recent],
        }

    def to_narrative(self) -> str:
        parts: list[str] = []
        if self.recent:
            names = ", ".join(e.corp_name for e in self.recent[:3])
            parts.append(f"최근 실적 발표: {names}")
            for e in self.recent[:3]:
                if e.surprise:
                    parts.append(f"  {e.corp_name}: {e.surprise}")
                if e.op_yoy is not None:
                    parts.append(f"  영업이익 YoY {e.op_yoy:+.1f}%")
        if self.upcoming:
            names = ", ".join(f"{e.corp_name}(D-{e.d_day})" for e in self.upcoming[:3])
            parts.append(f"실적 발표 예정: {names}")
        return ". ".join(parts) if parts else "추적 대상 실적 이벤트 없음"


def _determine_report_type(report_nm: str) -> str:
    """공시 보고서명에서 분기를 추출한다."""
    if "1분기" in report_nm or "Q1" in report_nm.upper():
        return "1Q"
    elif "반기" in report_nm or "2분기" in report_nm or "Q2" in report_nm.upper():
        return "2Q"
    elif "3분기" in report_nm or "Q3" in report_nm.upper():
        return "3Q"
    elif "사업" in report_nm or "4분기" in report_nm or "Q4" in report_nm.upper():
        return "4Q"
    return "4Q"


def _determine_surprise(op_yoy: float | None) -> str | None:
    """영업이익 YoY 기준 서프라이즈/쇼크 판정."""
    if op_yoy is None:
        return None
    if op_yoy >= 20:
        return "어닝서프라이즈"
    elif op_yoy <= -20:
        return "어닝쇼크"
    else:
        return "부합"


def _fetch_disclosure_list(
    api_key: str,
    corp_code: str,
    bgn_de: str,
    end_de: str,
) -> list[dict]:
    """DART 공시 목록 조회."""
    url = f"{_DART_BASE}/api/list.json"
    params = {
        "crtfc_key": api_key,
        "corp_code": corp_code,
        "bgn_de": bgn_de,
        "end_de": end_de,
        "pblntf_detail_ty": "A001",  # 사업보고서
        "page_count": "10",
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") == "000":
            return data.get("list", [])
        return []
    except Exception as exc:
        logger.warning("DART list API failed for %s: %s", corp_code, str(exc)[:100])
        return []


def _fetch_financials(
    api_key: str,
    corp_code: str,
    bsns_year: str,
    reprt_code: str,
) -> dict | None:
    """DART 재무제표 조회 (매출/영업이익/순이익)."""
    url = f"{_DART_BASE}/api/fnlttSinglAcntAll.json"
    params = {
        "crtfc_key": api_key,
        "corp_code": corp_code,
        "bsns_year": bsns_year,
        "reprt_code": reprt_code,
        "fs_div": "CFS",  # 연결재무제표
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") != "000":
            return None

        result = {}
        for item in data.get("list", []):
            acnt_nm = item.get("account_nm", "")
            thstrm_amount = item.get("thstrm_amount", "")
            if not thstrm_amount:
                continue
            try:
                amount = int(thstrm_amount.replace(",", ""))
            except (ValueError, AttributeError):
                continue

            if "매출" in acnt_nm and "revenue" not in result:
                result["revenue"] = amount
            elif "영업이익" in acnt_nm and "operating_profit" not in result:
                result["operating_profit"] = amount
            elif "당기순이익" in acnt_nm and "net_income" not in result:
                result["net_income"] = amount

        return result if result else None
    except Exception as exc:
        logger.warning("DART financials failed for %s: %s", corp_code, str(exc)[:100])
        return None


def _reprt_code_from_type(report_type: str) -> str:
    """분기 → DART reprt_code."""
    return {
        "1Q": "11013",
        "2Q": "11012",
        "3Q": "11014",
        "4Q": "11011",
    }.get(report_type, "11011")


def fetch_earnings_snapshot(
    config: Config,
    trade_date: str,
) -> EarningsSnapshot | None:
    """실적 스냅샷을 수집한다.

    Args:
        config: Config (dart_api_key 사용)
        trade_date: "YYYY-MM-DD"

    Returns:
        EarningsSnapshot 또는 None (API key 없으면)
    """
    if not config.dart_api_key:
        logger.warning("DART_API_KEY not set — skipping earnings snapshot")
        return None

    try:
        today = datetime.strptime(trade_date, "%Y-%m-%d")
        recent_start = (today - timedelta(days=7)).strftime("%Y%m%d")
        upcoming_end = (today + timedelta(days=14)).strftime("%Y%m%d")
        today_str = today.strftime("%Y%m%d")

        snapshot = EarningsSnapshot()

        for corp_name, corp_code in _TRACKED_CORPS.items():
            # 최근 1주 + 향후 2주 범위 조회
            disclosures = _fetch_disclosure_list(
                config.dart_api_key,
                corp_code,
                recent_start,
                upcoming_end,
            )

            for disc in disclosures:
                rcept_dt = disc.get("rcept_dt", "")
                report_nm = disc.get("report_nm", "")

                if not rcept_dt:
                    continue

                # 날짜 파싱
                try:
                    disc_date = datetime.strptime(rcept_dt, "%Y%m%d")
                except ValueError:
                    continue

                report_date = disc_date.strftime("%Y-%m-%d")
                report_type = _determine_report_type(report_nm)
                d_day = (disc_date - today).days

                event = EarningsEvent(
                    corp_code=corp_code,
                    corp_name=corp_name,
                    report_date=report_date,
                    report_type=report_type,
                    d_day=d_day,
                )

                # 이미 발표된 경우 재무데이터 조회
                if d_day <= 0:
                    bsns_year = str(disc_date.year)
                    reprt_code = _reprt_code_from_type(report_type)
                    financials = _fetch_financials(
                        config.dart_api_key, corp_code, bsns_year, reprt_code,
                    )
                    if financials:
                        event.revenue = financials.get("revenue")
                        event.operating_profit = financials.get("operating_profit")
                        event.net_income = financials.get("net_income")

                    # TODO: YoY 계산 (전년 동기 재무데이터 조회 필요, 현재는 생략)
                    event.surprise = _determine_surprise(event.op_yoy)
                    snapshot.recent.append(event)
                else:
                    snapshot.upcoming.append(event)

        # 정렬
        snapshot.upcoming.sort(key=lambda e: e.d_day)
        snapshot.recent.sort(key=lambda e: e.d_day, reverse=True)

        logger.info("earnings: %d recent, %d upcoming",
                     len(snapshot.recent), len(snapshot.upcoming))
        return snapshot

    except Exception as exc:
        logger.error("earnings snapshot failed: %s", str(exc)[:300])
        return None
