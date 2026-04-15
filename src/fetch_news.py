"""종목별 뉴스 수집 모듈.

Google News RSS를 기본 소스로 사용 (키 없이 안정적).
네이버 뉴스 API도 지원하지만 현재는 Google RSS 우선.
"""
from __future__ import annotations

import html
import logging
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import quote

import requests

from src.config import Config

logger = logging.getLogger(__name__)

_TAG_RE = re.compile(r"<[^>]+>")
_USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
               "AppleWebKit/537.36 (KHTML, like Gecko) "
               "Chrome/120.0.0.0 Safari/537.36")


@dataclass
class NewsItem:
    title: str
    description: str
    link: str
    press: str  # 언론사
    pub_date: Optional[datetime] = None
    credibility: float = 0.5    # 언론사 신뢰도 (0.0~1.0)
    sentiment: str = "중립"     # "긍정"|"부정"|"중립"

    def __str__(self) -> str:
        date_str = self.pub_date.strftime("%m/%d %H:%M") if self.pub_date else ""
        return f"[{self.press}] {self.title} ({date_str}) [{self.sentiment}]"

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "description": self.description,
            "link": self.link,
            "press": self.press,
            "pub_date": self.pub_date.isoformat() if self.pub_date else None,
            "credibility": self.credibility,
            "sentiment": self.sentiment,
        }


# ── 언론사 신뢰도 ────────────────────────────────────────────────
_PRESS_TIERS: dict[str, float] = {
    # Tier 1 (0.9): 주요 경제지
    "한국경제": 0.9, "한경": 0.9, "매일경제": 0.9, "서울경제": 0.9,
    "아시아경제": 0.9, "헤럴드경제": 0.9, "파이낸셜뉴스": 0.9,
    "이데일리": 0.9, "머니투데이": 0.9,
    # Tier 2 (0.8): 종합지/통신사
    "조선일보": 0.8, "중앙일보": 0.8, "동아일보": 0.8, "한겨레": 0.8,
    "경향신문": 0.8, "연합뉴스": 0.8,
    # Tier 3 (0.7): 전문지/방송
    "조선비즈": 0.7, "전자신문": 0.7, "ZDNet Korea": 0.7,
    "인베스트조선": 0.7, "뉴스1": 0.7, "뉴시스": 0.7,
    "SBS": 0.7, "MBC": 0.7, "KBS": 0.7, "YTN": 0.7, "JTBC": 0.7,
    # Tier 4 (0.4): 데이터 자동생성
    "핀포인트뉴스": 0.5, "서울데이터랩": 0.3,
}


def _score_credibility(press: str) -> float:
    """언론사명으로 신뢰도 점수 반환."""
    for name, score in _PRESS_TIERS.items():
        if name in press:
            return score
    return 0.5


# ── 뉴스 감성 분석 ──────────────────────────────────────────────
_POSITIVE_KEYWORDS = [
    "급등", "상한가", "호재", "수주", "계약", "흑자", "실적 개선",
    "신고가", "최고가", "상승", "호조", "수혜", "성장", "확대",
    "공급계약", "기술 수출", "특허", "승인", "순매수",
]
_NEGATIVE_KEYWORDS = [
    "급락", "하한가", "악재", "적자", "손실", "하락", "부진",
    "상장폐지", "상폐", "감자", "횡령", "배임", "불성실공시",
    "거래정지", "자본잠식", "소송", "리콜", "제재", "순매도",
]


def _analyze_sentiment(title: str, description: str = "") -> str:
    """제목+설명 키워드 기반 감성 분류."""
    text = f"{title} {description}"
    pos = sum(1 for kw in _POSITIVE_KEYWORDS if kw in text)
    neg = sum(1 for kw in _NEGATIVE_KEYWORDS if kw in text)
    if pos > neg:
        return "긍정"
    elif neg > pos:
        return "부정"
    return "중립"


# ── 뉴스 중복 제거 ──────────────────────────────────────────────
def _deduplicate_news(items: list[NewsItem], threshold: float = 0.6) -> list[NewsItem]:
    """유사 제목 뉴스 제거 (자카드 유사도)."""
    if len(items) <= 1:
        return items

    def _jaccard(a: str, b: str) -> float:
        set_a = set(a.split())
        set_b = set(b.split())
        if not set_a or not set_b:
            return 0.0
        return len(set_a & set_b) / len(set_a | set_b)

    result: list[NewsItem] = [items[0]]
    for item in items[1:]:
        is_dup = any(_jaccard(item.title, kept.title) >= threshold for kept in result)
        if not is_dup:
            result.append(item)
    return result


def _clean(text: str) -> str:
    return html.unescape(_TAG_RE.sub("", text or "")).strip()


def _parse_rss_date(raw: str) -> Optional[datetime]:
    """RSS 날짜 포맷 파싱 (RFC 2822)."""
    if not raw:
        return None
    fmts = [
        "%a, %d %b %Y %H:%M:%S %Z",
        "%a, %d %b %Y %H:%M:%S %z",
        "%a, %d %b %Y %H:%M:%S GMT",
    ]
    for fmt in fmts:
        try:
            dt = datetime.strptime(raw, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            continue
    return None


def _strip_source_suffix(title: str, press: str) -> str:
    """'제목 - 언론사' 형태의 suffix 제거."""
    suffix = f" - {press}"
    if title.endswith(suffix):
        return title[: -len(suffix)].rstrip()
    # 이중 suffix: '제목 - 언론사 - 언론사'
    parts = title.rsplit(" - ", 1)
    if len(parts) == 2 and len(parts[1]) < 40:
        return parts[0].rstrip()
    return title


def _fetch_via_google_rss(query: str, display: int = 10) -> list[NewsItem]:
    """Google News RSS에서 한국어 뉴스 검색."""
    url = f"https://news.google.com/rss/search?q={quote(query)}&hl=ko&gl=KR&ceid=KR:ko"
    try:
        resp = requests.get(url, headers={"User-Agent": _USER_AGENT}, timeout=10)
        resp.raise_for_status()
    except Exception as exc:
        logger.error("google news rss failed for '%s': %s", query, exc)
        return []

    try:
        root = ET.fromstring(resp.text)
    except ET.ParseError as exc:
        logger.error("rss parse error: %s", exc)
        return []

    items: list[NewsItem] = []
    for item in root.findall(".//item")[:display]:
        title = _clean(item.findtext("title", ""))
        link = item.findtext("link", "") or ""
        desc = _clean(item.findtext("description", ""))
        pub_raw = item.findtext("pubDate", "") or ""
        source_el = item.find("source")
        press = source_el.text.strip() if (source_el is not None and source_el.text) else "unknown"

        title = _strip_source_suffix(title, press)

        items.append(NewsItem(
            title=title,
            description=desc,
            link=link,
            press=press,
            pub_date=_parse_rss_date(pub_raw),
            credibility=_score_credibility(press),
            sentiment=_analyze_sentiment(title, desc),
        ))

    items = _deduplicate_news(items)
    logger.info("google rss: %d items for '%s'", len(items), query)
    return items


def _fetch_via_naver_api(
    query: str, client_id: str, client_secret: str, display: int = 10
) -> list[NewsItem]:
    """네이버 뉴스 검색 API (선택적)."""
    url = "https://openapi.naver.com/v1/search/news.json"
    headers = {
        "X-Naver-Client-Id": client_id,
        "X-Naver-Client-Secret": client_secret,
    }
    params = {"query": query, "display": display, "sort": "date"}
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.error("naver api failed for '%s': %s", query, exc)
        return []

    items: list[NewsItem] = []
    for it in data.get("items", []):
        try:
            pub = datetime.strptime(it["pubDate"], "%a, %d %b %Y %H:%M:%S %z")
        except (KeyError, ValueError):
            pub = None
        orig = it.get("originallink", "") or it.get("link", "")
        m = re.search(r"https?://(?:www\.)?([^/]+)", orig)
        press = m.group(1).split(".")[-2] if m else "unknown"
        clean_title = _clean(it.get("title", ""))
        clean_desc = _clean(it.get("description", ""))
        items.append(NewsItem(
            title=clean_title,
            description=clean_desc,
            link=it.get("link", ""),
            press=press,
            pub_date=pub,
            credibility=_score_credibility(press),
            sentiment=_analyze_sentiment(clean_title, clean_desc),
        ))
    items = _deduplicate_news(items)
    logger.info("naver api: %d items for '%s'", len(items), query)
    return items


def fetch_news_for_stock(
    name: str,
    config: Optional[Config] = None,
    display: int = 5,
    prefer: str = "google",  # "google" | "naver"
) -> list[NewsItem]:
    """종목명으로 최근 뉴스 N개를 가져온다.

    기본은 Google News RSS (키 불필요). `prefer="naver"`이면서 키가 있으면 네이버 API 사용.
    """
    if config is None:
        config = Config.load()

    has_naver = bool(config.naver_client_id and config.naver_client_secret)

    if prefer == "naver" and has_naver:
        items = _fetch_via_naver_api(
            name, config.naver_client_id, config.naver_client_secret, display
        )
        if items:
            return items
        logger.warning("naver empty, falling back to google rss")

    return _fetch_via_google_rss(name, display)


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    query = sys.argv[1] if len(sys.argv) > 1 else "대우건설"
    cfg = Config.load()
    news = fetch_news_for_stock(query, cfg, display=5)

    print(f"\n[{query}] 최근 뉴스 {len(news)}건")
    print("=" * 60)
    for i, n in enumerate(news, 1):
        print(f"{i}. {n}")
        if n.description:
            print(f"   → {n.description[:90]}")
