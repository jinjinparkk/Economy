"""fetch_news 모듈 단위 테스트."""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import patch, MagicMock

import pytest

from src.fetch_news import (
    NewsItem,
    _clean,
    _parse_rss_date,
    _strip_source_suffix,
    _fetch_via_google_rss,
    fetch_news_for_stock,
    _score_credibility,
    _analyze_sentiment,
    _deduplicate_news,
)


# ── NewsItem ──────────────────────────────────────────────────────────
class TestNewsItem:
    def test_str_with_date(self):
        dt = datetime(2026, 4, 9, 10, 30, tzinfo=timezone.utc)
        n = NewsItem("제목", "설명", "http://x", "조선비즈", dt)
        s = str(n)
        assert "조선비즈" in s
        assert "04/09" in s

    def test_str_no_date(self):
        n = NewsItem("제목", "설명", "http://x", "한경", None)
        assert "한경" in str(n)

    def test_to_dict_roundtrip(self):
        dt = datetime(2026, 4, 9, 10, 0, tzinfo=timezone.utc)
        n = NewsItem("T", "D", "http://x", "P", dt)
        d = n.to_dict()
        assert d["title"] == "T"
        assert d["press"] == "P"
        assert d["pub_date"] == dt.isoformat()

    def test_to_dict_no_date(self):
        n = NewsItem("T", "D", "http://x", "P", None)
        assert n.to_dict()["pub_date"] is None


# ── _clean ────────────────────────────────────────────────────────────
class TestClean:
    def test_removes_html_tags(self):
        assert _clean("<b>bold</b>") == "bold"

    def test_unescapes_entities(self):
        assert _clean("A &amp; B") == "A & B"

    def test_empty(self):
        assert _clean("") == ""

    def test_none_safe(self):
        assert _clean(None) == ""


# ── _parse_rss_date ──────────────────────────────────────────────────
class TestParseRssDate:
    def test_rfc2822_gmt(self):
        raw = "Wed, 09 Apr 2026 12:00:00 GMT"
        dt = _parse_rss_date(raw)
        assert dt is not None
        assert dt.day == 9
        assert dt.month == 4

    def test_rfc2822_with_tz(self):
        raw = "Wed, 09 Apr 2026 12:00:00 +0900"
        dt = _parse_rss_date(raw)
        assert dt is not None

    def test_empty(self):
        assert _parse_rss_date("") is None

    def test_none(self):
        assert _parse_rss_date(None) is None

    def test_invalid_format(self):
        assert _parse_rss_date("invalid date") is None


# ── _strip_source_suffix ─────────────────────────────────────────────
class TestStripSourceSuffix:
    def test_exact_suffix(self):
        assert _strip_source_suffix("기사 제목 - 조선비즈", "조선비즈") == "기사 제목"

    def test_no_suffix(self):
        assert _strip_source_suffix("기사 제목", "조선비즈") == "기사 제목"

    def test_double_suffix(self):
        result = _strip_source_suffix("기사 제목 - 뉴스데스크 - 한경", "조선비즈")
        assert result == "기사 제목 - 뉴스데스크"


# ── _fetch_via_google_rss (mock) ─────────────────────────────────────
SAMPLE_RSS = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
<channel>
<item>
  <title>삼성전자 실적 호조 - 한경</title>
  <link>https://news.google.com/articles/xxx</link>
  <description>삼성전자가 분기 실적...</description>
  <pubDate>Wed, 09 Apr 2026 03:00:00 GMT</pubDate>
  <source url="https://www.hankyung.com">한경</source>
</item>
<item>
  <title>반도체 수출 증가</title>
  <link>https://news.google.com/articles/yyy</link>
  <description>반도체 수출이...</description>
  <pubDate>Tue, 08 Apr 2026 10:00:00 GMT</pubDate>
  <source url="https://www.mk.co.kr">매일경제</source>
</item>
</channel>
</rss>"""


class TestFetchViaGoogleRss:
    @patch("src.fetch_news.requests.get")
    def test_parses_rss_correctly(self, mock_get):
        resp = MagicMock()
        resp.status_code = 200
        resp.text = SAMPLE_RSS
        resp.raise_for_status = MagicMock()
        mock_get.return_value = resp

        items = _fetch_via_google_rss("삼성전자", display=5)
        assert len(items) == 2
        assert items[0].press == "한경"
        assert "삼성전자 실적 호조" in items[0].title
        assert items[0].pub_date is not None

    @patch("src.fetch_news.requests.get")
    def test_display_limit(self, mock_get):
        resp = MagicMock()
        resp.status_code = 200
        resp.text = SAMPLE_RSS
        resp.raise_for_status = MagicMock()
        mock_get.return_value = resp

        items = _fetch_via_google_rss("삼성전자", display=1)
        assert len(items) == 1

    @patch("src.fetch_news.requests.get")
    def test_network_error_returns_empty(self, mock_get):
        mock_get.side_effect = Exception("timeout")
        items = _fetch_via_google_rss("삼성전자")
        assert items == []

    @patch("src.fetch_news.requests.get")
    def test_invalid_xml_returns_empty(self, mock_get):
        resp = MagicMock()
        resp.text = "not xml at all"
        resp.raise_for_status = MagicMock()
        mock_get.return_value = resp

        items = _fetch_via_google_rss("삼성전자")
        assert items == []


# ── fetch_news_for_stock (통합) ──────────────────────────────────────
class TestFetchNewsForStock:
    @patch("src.fetch_news._fetch_via_google_rss")
    def test_defaults_to_google(self, mock_google):
        mock_google.return_value = [NewsItem("T", "D", "L", "P")]
        from src.config import Config
        cfg = Config(
            llm_provider="gemini", anthropic_api_key="", gemini_api_key="",
            naver_client_id="", naver_client_secret="", dart_api_key="",
            output_dir=__import__("pathlib").Path("."), timezone="Asia/Seoul",
            mover_threshold_pct=5.0, volume_surge_multiplier=3.0, top_n_movers=5,
            wp_access_token="", wp_site_id="", wp_auto_publish=False,
        )
        items = fetch_news_for_stock("삼성전자", cfg)
        mock_google.assert_called_once()
        assert len(items) == 1


# ── 신뢰도 / 감성 / 중복제거 ────────────────────────────────────────
class TestCredibility:
    def test_tier1(self):
        assert _score_credibility("한국경제") == 0.9

    def test_tier2(self):
        assert _score_credibility("연합뉴스") == 0.8

    def test_tier3(self):
        assert _score_credibility("조선비즈") == 0.7

    def test_unknown(self):
        assert _score_credibility("개인블로그") == 0.5

    def test_partial_match(self):
        assert _score_credibility("매일경제 마켓") == 0.9


class TestSentiment:
    def test_positive(self):
        assert _analyze_sentiment("삼성전자 수주 호재") == "긍정"

    def test_negative(self):
        assert _analyze_sentiment("상장폐지 위기 급락") == "부정"

    def test_neutral(self):
        assert _analyze_sentiment("삼성전자 주가 동향") == "중립"

    def test_mixed_more_negative(self):
        assert _analyze_sentiment("상한가 이후 급락 자본잠식") == "부정"


class TestDeduplicate:
    def test_removes_similar(self):
        items = [
            NewsItem("삼성전자 주가 급등 상한가", "", "", "A"),
            NewsItem("삼성전자 주가 급등 상한가 기록", "", "", "B"),
        ]
        result = _deduplicate_news(items)
        assert len(result) == 1

    def test_keeps_distinct(self):
        items = [
            NewsItem("삼성전자 수주 계약 체결", "", "", "A"),
            NewsItem("현대차 실적 발표 임박", "", "", "B"),
        ]
        result = _deduplicate_news(items)
        assert len(result) == 2

    def test_single_item(self):
        items = [NewsItem("제목", "", "", "A")]
        result = _deduplicate_news(items)
        assert len(result) == 1

    def test_empty(self):
        result = _deduplicate_news([])
        assert len(result) == 0

    def test_to_dict_includes_new_fields(self):
        n = NewsItem("제목", "설명", "http://x", "한경", credibility=0.9, sentiment="긍정")
        d = n.to_dict()
        assert d["credibility"] == 0.9
        assert d["sentiment"] == "긍정"
