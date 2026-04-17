"""프리마켓 브리핑 인사이트 모듈.

3가지 인사이트를 수집한다:
1. 글로벌 이슈 트래커 — 시장 영향 큰 이슈 자동 추적 (Google RSS)
2. 테마주 연결 분석 — 미국 매크로 변동 → 한국 수혜/피해주 매핑
3. 경제 캘린더 브리핑 — 오늘/이번주 경제지표 발표 일정 구조화
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# 1. 글로벌 이슈 트래커
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class GlobalIssue:
    """추적 중인 글로벌 이슈."""

    topic: str           # "SpaceX IPO"
    category: str        # "IPO" | "통화정책" | "무역" | "지정학" | ...
    keywords: list[str]  # 검색 키워드
    kr_impact: str       # 한국 시장 영향 요약
    kr_stocks_positive: list[str] = field(default_factory=list)   # 수혜 예상 종목
    kr_stocks_negative: list[str] = field(default_factory=list)   # 피해 예상 종목
    headlines: list[str] = field(default_factory=list)            # 수집된 최신 뉴스 헤드라인

    def to_dict(self) -> dict:
        return {
            "topic": self.topic,
            "category": self.category,
            "kr_impact": self.kr_impact,
            "kr_stocks_positive": self.kr_stocks_positive,
            "kr_stocks_negative": self.kr_stocks_negative,
            "headlines": self.headlines,
        }


# 추적 이슈 목록 — 코드 내 dict로 관리
# kr_stocks_positive: 해당 이슈 활성 시 상승 예상 종목
# kr_stocks_negative: 해당 이슈 활성 시 하락 예상 종목
_TRACKED_ISSUES: list[dict] = [
    {
        "topic": "SpaceX IPO",
        "category": "IPO",
        "keywords": ["SpaceX 상장", "SpaceX IPO"],
        "kr_impact": "SpaceX 지분 보유 VC·우주항공 부품주 직접 수혜",
        "kr_stocks_positive": ["미래에셋벤처투자", "아주IB투자", "한화에어로스페이스", "쎄트렉아이", "AP위성"],
        "kr_stocks_negative": [],
    },
    {
        "topic": "Fed 금리 인하",
        "category": "통화정책",
        "keywords": ["Fed 금리 인하", "FOMC 비둘기"],
        "kr_impact": "금리 인하 → 성장주·부동산·리츠 수혜, 은행 순이자마진 축소",
        "kr_stocks_positive": ["카카오", "네이버", "NAVER", "맥쿼리인프라", "SK리츠"],
        "kr_stocks_negative": ["KB금융", "신한지주", "하나금융지주"],
    },
    {
        "topic": "Fed 금리 동결/인상",
        "category": "통화정책",
        "keywords": ["Fed 금리 동결", "FOMC 매파"],
        "kr_impact": "금리 인상·동결 장기화 → 은행주 수혜, 성장주·부동산 약세",
        "kr_stocks_positive": ["KB금융", "신한지주", "하나금융지주", "우리금융지주"],
        "kr_stocks_negative": ["카카오", "네이버", "맥쿼리인프라"],
    },
    {
        "topic": "미중 무역갈등",
        "category": "무역",
        "keywords": ["미중 관세", "미중 무역전쟁"],
        "kr_impact": "반도체·디스플레이 수출주 타격, 내수주 상대 수혜",
        "kr_stocks_positive": ["이마트", "CJ제일제당", "오리온"],
        "kr_stocks_negative": ["삼성전자", "SK하이닉스", "LG디스플레이"],
    },
    {
        "topic": "AI 반도체 투자",
        "category": "기술",
        "keywords": ["AI 반도체 투자", "엔비디아 실적"],
        "kr_impact": "HBM·AI 가속기 수요 확대 → 반도체·장비주 직접 수혜",
        "kr_stocks_positive": ["SK하이닉스", "삼성전자", "한미반도체", "리노공업", "ISC"],
        "kr_stocks_negative": [],
    },
    {
        "topic": "일본 엔화 정책",
        "category": "통화정책",
        "keywords": ["일본 금리 인상", "엔화 강세"],
        "kr_impact": "엔캐리 청산 → 외국인 매도 압력, 원화 약세 시 수출주 반사 수혜",
        "kr_stocks_positive": ["현대차", "기아", "삼성전자"],
        "kr_stocks_negative": ["대한항공", "신세계", "호텔신라"],
    },
    {
        "topic": "중국 경기부양",
        "category": "재정정책",
        "keywords": ["중국 경기부양", "중국 부양책"],
        "kr_impact": "중국 소비·인프라 확대 → 화학·철강·화장품 수혜",
        "kr_stocks_positive": ["아모레퍼시픽", "코스맥스", "포스코홀딩스", "LG화학", "효성첨단소재"],
        "kr_stocks_negative": [],
    },
    {
        "topic": "유가 지정학 리스크",
        "category": "지정학",
        "keywords": ["중동 긴장", "이란 이스라엘"],
        "kr_impact": "유가 급등 → 정유·방산 수혜, 항공·운송 피해",
        "kr_stocks_positive": ["SK이노베이션", "S-Oil", "한화에어로스페이스", "LIG넥스원"],
        "kr_stocks_negative": ["대한항공", "제주항공", "CJ대한통운"],
    },
    {
        "topic": "미국 관세/IRA 정책",
        "category": "정치",
        "keywords": ["트럼프 관세", "IRA 보조금"],
        "kr_impact": "IRA 수혜 축소 시 2차전지 피해, 관세 확대 시 방산주 수혜",
        "kr_stocks_positive": ["한화에어로스페이스", "현대로템", "LIG넥스원"],
        "kr_stocks_negative": ["LG에너지솔루션", "삼성SDI", "에코프로비엠"],
    },
    {
        "topic": "글로벌 은행 리스크",
        "category": "금융",
        "keywords": ["은행 위기", "SVB 뱅크런"],
        "kr_impact": "금융주 동반 하락, 안전자산(금·채권 ETF) 수혜",
        "kr_stocks_positive": ["KODEX 골드선물", "TIGER 미국채10년선물"],
        "kr_stocks_negative": ["KB금융", "신한지주", "하나금융지주", "우리금융지주"],
    },
    {
        "topic": "2차전지 공급망",
        "category": "산업",
        "keywords": ["2차전지 수주", "전기차 배터리"],
        "kr_impact": "글로벌 EV 확산 → 배터리·소재·장비주 직접 수혜",
        "kr_stocks_positive": ["LG에너지솔루션", "삼성SDI", "에코프로비엠", "포스코퓨처엠", "엘앤에프"],
        "kr_stocks_negative": [],
    },
    {
        "topic": "로봇/자율주행",
        "category": "기술",
        "keywords": ["테슬라 로봇", "자율주행 상용화"],
        "kr_impact": "로봇·자율주행 부품주 수혜",
        "kr_stocks_positive": ["삼성전자", "레인보우로보틱스", "두산로보틱스", "현대차", "모트렉스"],
        "kr_stocks_negative": [],
    },
]


def fetch_global_issues() -> list[GlobalIssue]:
    """추적 이슈 중 최근 24시간 뉴스가 있는 '활성 이슈'만 반환한다.

    각 이슈의 keywords로 Google RSS 검색 → 최근 24시간 뉴스 1건 이상이면 포함.
    이슈당 최대 3개 헤드라인.
    """
    from src.fetch_news import _fetch_via_google_rss

    cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
    active_issues: list[GlobalIssue] = []

    for issue_cfg in _TRACKED_ISSUES:
        headlines: list[str] = []
        seen: set[str] = set()

        for keyword in issue_cfg["keywords"]:
            items = _fetch_via_google_rss(keyword, display=3)
            for item in items:
                # 24시간 필터
                if item.pub_date and item.pub_date < cutoff:
                    continue
                if item.title not in seen:
                    seen.add(item.title)
                    headlines.append(item.title)

        if not headlines:
            continue

        active_issues.append(GlobalIssue(
            topic=issue_cfg["topic"],
            category=issue_cfg["category"],
            keywords=issue_cfg["keywords"],
            kr_impact=issue_cfg["kr_impact"],
            kr_stocks_positive=issue_cfg.get("kr_stocks_positive", []),
            kr_stocks_negative=issue_cfg.get("kr_stocks_negative", []),
            headlines=headlines[:3],
        ))

    logger.info("global issues: %d active out of %d tracked",
                len(active_issues), len(_TRACKED_ISSUES))
    return active_issues


# ═══════════════════════════════════════════════════════════════════════
# 2. 테마주 연결 분석
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class ThemeConnection:
    """매크로 변동 → 한국 테마주 연결."""

    trigger: str          # "WTI +3.72%"
    theme: str            # "유가 급등"
    kr_sectors: list[str] # ["정유/화학", "해운"]
    kr_stocks: list[str]  # ["SK이노베이션", "S-Oil", "HMM"]
    direction: str        # "수혜" | "피해"

    def to_dict(self) -> dict:
        return {
            "trigger": self.trigger,
            "theme": self.theme,
            "kr_sectors": self.kr_sectors,
            "kr_stocks": self.kr_stocks,
            "direction": self.direction,
        }


# 매핑 테이블 — 코드 내 dict
_THEME_MAP: list[dict] = [
    {
        "trigger_key": "WTI",
        "threshold_pct": 2.0,
        "theme": "유가 급등",
        "direction": "수혜",
        "kr_sectors": ["정유/화학", "해운"],
        "kr_stocks": ["SK이노베이션", "S-Oil", "HMM"],
    },
    {
        "trigger_key": "WTI",
        "threshold_pct": -2.0,
        "theme": "유가 급락",
        "direction": "수혜",
        "kr_sectors": ["항공", "화학 원가 하락"],
        "kr_stocks": ["대한항공", "아시아나항공", "제주항공"],
    },
    {
        "trigger_key": "SOXX",
        "threshold_pct": 1.5,
        "theme": "반도체 강세",
        "direction": "수혜",
        "kr_sectors": ["반도체"],
        "kr_stocks": ["삼성전자", "SK하이닉스", "한미반도체"],
    },
    {
        "trigger_key": "SOXX",
        "threshold_pct": -1.5,
        "theme": "반도체 약세",
        "direction": "피해",
        "kr_sectors": ["반도체"],
        "kr_stocks": ["삼성전자", "SK하이닉스", "한미반도체"],
    },
    {
        "trigger_key": "GOLD",
        "threshold_pct": 1.0,
        "theme": "금 가격 상승",
        "direction": "수혜",
        "kr_sectors": ["귀금속"],
        "kr_stocks": ["고려아연", "풍산"],
    },
    {
        "trigger_key": "GOLD",
        "threshold_pct": -1.0,
        "theme": "금 가격 하락",
        "direction": "피해",
        "kr_sectors": ["귀금속"],
        "kr_stocks": ["고려아연", "풍산"],
    },
    {
        "trigger_key": "XLK",
        "threshold_pct": 1.5,
        "theme": "미국 기술주 강세",
        "direction": "수혜",
        "kr_sectors": ["IT/소프트웨어"],
        "kr_stocks": ["네이버", "카카오", "삼성SDS"],
    },
    {
        "trigger_key": "XLK",
        "threshold_pct": -1.5,
        "theme": "미국 기술주 약세",
        "direction": "피해",
        "kr_sectors": ["IT/소프트웨어"],
        "kr_stocks": ["네이버", "카카오", "삼성SDS"],
    },
    {
        "trigger_key": "VIX",
        "threshold_pct": 10.0,
        "theme": "공포 지수 급등",
        "direction": "수혜",
        "kr_sectors": ["인버스ETF", "안전자산"],
        "kr_stocks": ["KODEX 인버스", "TIGER 금은선물"],
    },
    {
        "trigger_key": "VIX",
        "threshold_pct": -10.0,
        "theme": "공포 지수 급락",
        "direction": "수혜",
        "kr_sectors": ["레버리지ETF", "성장주"],
        "kr_stocks": ["KODEX 레버리지", "TIGER 나스닥100"],
    },
    {
        "trigger_key": "XLE",
        "threshold_pct": 2.0,
        "theme": "에너지 섹터 강세",
        "direction": "수혜",
        "kr_sectors": ["에너지", "정유"],
        "kr_stocks": ["SK이노베이션", "GS", "S-Oil"],
    },
    {
        "trigger_key": "XLV",
        "threshold_pct": 1.5,
        "theme": "헬스케어 강세",
        "direction": "수혜",
        "kr_sectors": ["바이오", "제약"],
        "kr_stocks": ["삼성바이오로직스", "셀트리온", "SK바이오팜"],
    },
    {
        "trigger_key": "USDKRW",
        "threshold_pct": 0.5,
        "theme": "원화 약세 (달러 강세)",
        "direction": "수혜",
        "kr_sectors": ["수출주"],
        "kr_stocks": ["현대차", "삼성전자", "포스코홀딩스"],
    },
    {
        "trigger_key": "USDKRW",
        "threshold_pct": -0.5,
        "theme": "원화 강세 (달러 약세)",
        "direction": "수혜",
        "kr_sectors": ["내수주", "항공"],
        "kr_stocks": ["대한항공", "신세계", "이마트"],
    },
    {
        "trigger_key": "TSLA",
        "threshold_pct": 3.0,
        "theme": "테슬라 급등",
        "direction": "수혜",
        "kr_sectors": ["2차전지", "전기차부품"],
        "kr_stocks": ["LG에너지솔루션", "삼성SDI", "에코프로비엠"],
    },
    {
        "trigger_key": "TSLA",
        "threshold_pct": -3.0,
        "theme": "테슬라 급락",
        "direction": "피해",
        "kr_sectors": ["2차전지", "전기차부품"],
        "kr_stocks": ["LG에너지솔루션", "삼성SDI", "에코프로비엠"],
    },
    {
        "trigger_key": "COPPER",
        "threshold_pct": 2.0,
        "theme": "구리 가격 상승",
        "direction": "수혜",
        "kr_sectors": ["비철금속"],
        "kr_stocks": ["고려아연", "LS", "풍산"],
    },
    {
        "trigger_key": "XLF",
        "threshold_pct": 1.5,
        "theme": "미국 금융주 강세",
        "direction": "수혜",
        "kr_sectors": ["금융", "은행"],
        "kr_stocks": ["KB금융", "신한지주", "하나금융지주"],
    },
]


def detect_theme_connections(macro) -> list[ThemeConnection]:
    """매크로 스냅샷에서 threshold 초과 변동을 감지하여 테마 연결을 반환한다.

    Args:
        macro: MacroSnapshot 객체 (fetch_macro.py)
    """
    all_indicators = macro.all_indicators
    connections: list[ThemeConnection] = []

    for theme_cfg in _THEME_MAP:
        trigger_key = theme_cfg["trigger_key"]
        threshold = theme_cfg["threshold_pct"]

        indicator = all_indicators.get(trigger_key)
        if indicator is None:
            continue

        change_pct = indicator.change_pct

        # threshold가 양수: change_pct >= threshold 일 때 활성화
        # threshold가 음수: change_pct <= threshold 일 때 활성화
        if threshold >= 0 and change_pct >= threshold:
            activated = True
        elif threshold < 0 and change_pct <= threshold:
            activated = True
        else:
            activated = False

        if not activated:
            continue

        connections.append(ThemeConnection(
            trigger=f"{indicator.name} {change_pct:+.2f}%",
            theme=theme_cfg["theme"],
            kr_sectors=theme_cfg["kr_sectors"],
            kr_stocks=theme_cfg["kr_stocks"],
            direction=theme_cfg["direction"],
        ))

    logger.info("theme connections: %d activated", len(connections))
    return connections


# ═══════════════════════════════════════════════════════════════════════
# 3. 경제 캘린더 브리핑
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class EconEvent:
    """경제 캘린더 이벤트."""

    date: str             # "2026-04-17"
    time: str             # "21:30" (KST) 또는 ""
    event: str            # "미국 소매판매"
    importance: str       # "상" | "중" | "하"
    previous: str         # "0.2%" 또는 ""
    consensus: str        # "0.3%" 또는 ""
    headlines: list[str] = field(default_factory=list)  # 관련 뉴스

    def to_dict(self) -> dict:
        return {
            "date": self.date,
            "time": self.time,
            "event": self.event,
            "importance": self.importance,
            "previous": self.previous,
            "consensus": self.consensus,
            "headlines": self.headlines,
        }


# 중요도 매핑 — 키워드 기반
_IMPORTANCE_HIGH = ["FOMC", "CPI", "고용", "비농업", "금리결정", "PCE"]
_IMPORTANCE_MED = ["GDP", "소매판매", "ISM", "PMI", "실업률", "실업수당", "소비자심리"]


def _classify_importance(text: str) -> str:
    """뉴스 제목/이벤트명에서 중요도를 판별한다."""
    for kw in _IMPORTANCE_HIGH:
        if kw in text:
            return "상"
    for kw in _IMPORTANCE_MED:
        if kw in text:
            return "중"
    return "하"


def fetch_econ_calendar() -> list[EconEvent]:
    """경제 캘린더 이벤트를 Google RSS 키워드 검색으로 수집한다.

    기존 _fetch_econ_calendar_news() 대비:
    - 검색 키워드 6개로 확대
    - 이벤트별 중요도 태깅
    - EconEvent 구조체로 구조화
    """
    from src.fetch_news import _fetch_via_google_rss

    queries = [
        "오늘 경제지표 발표",
        "FOMC 금리 결정",
        "미국 CPI 고용",
        "미국 소매판매 GDP",
        "미국 실업수당 청구",
        "한국 금리 한국은행",
    ]

    today_str = datetime.now().strftime("%Y-%m-%d")
    seen_titles: set[str] = set()
    events: list[EconEvent] = []

    for q in queries:
        items = _fetch_via_google_rss(q, display=3)
        for item in items:
            if item.title in seen_titles:
                continue
            seen_titles.add(item.title)

            importance = _classify_importance(item.title)
            events.append(EconEvent(
                date=today_str,
                time="",
                event=item.title,
                importance=importance,
                previous="",
                consensus="",
                headlines=[f"[{item.press}] {item.title}" if item.press != "unknown" else item.title],
            ))

    # 중요도 순 정렬: 상 > 중 > 하
    importance_order = {"상": 0, "중": 1, "하": 2}
    events.sort(key=lambda e: importance_order.get(e.importance, 3))

    # 최대 10건
    events = events[:10]

    logger.info("econ calendar: %d events collected", len(events))
    return events


if __name__ == "__main__":
    import sys

    for stream in (sys.stdout, sys.stderr):
        if stream.encoding and stream.encoding.lower() != "utf-8":
            try:
                stream.reconfigure(encoding="utf-8")
            except Exception:
                pass
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    print("\n[1] 글로벌 이슈 트래커")
    print("=" * 60)
    issues = fetch_global_issues()
    for issue in issues:
        print(f"  [{issue.category}] {issue.topic}")
        for h in issue.headlines:
            print(f"    - {h}")
        print(f"    → 한국: {issue.kr_impact}")

    print("\n[2] 경제 캘린더")
    print("=" * 60)
    events = fetch_econ_calendar()
    for ev in events:
        print(f"  [{ev.importance}] {ev.event}")
        for h in ev.headlines:
            print(f"    - {h}")

    print("\n[3] 테마주 연결 분석 (매크로 데이터 필요)")
    print("  → py -3 -c \"from src.fetch_macro import fetch_macro_snapshot; "
          "from src.fetch_insight import detect_theme_connections; "
          "macro = fetch_macro_snapshot(); "
          "conns = detect_theme_connections(macro); "
          "[print(c) for c in conns]\"")
