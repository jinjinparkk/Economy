"""wordpress_publisher 모듈 단위 테스트."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from src.analyzer import Article
from src.detect_movers import Mover
from src.content_post import ContentPost
from src.wordpress_publisher import (
    PublishResult,
    _md_to_html,
    _build_tags,
    _build_categories,
    publish_to_wordpress,
    publish_articles,
    publish_content_post,
    publish_content_posts,
)


# ── 헬퍼 ──────────────────────────────────────────────────────────────
def _mover(name="테스트종목", code="000001", move_type="surge",
           change_pct=15.0, close=5000, amount=2e9, market="KOSPI",
           industry="반도체"):
    return Mover(code, name, market, move_type, close, change_pct,
                 100000, int(amount), int(1e11), industry=industry)


def _article(name="테스트종목", title="테스트 급등 분석", move_type="surge",
             industry="반도체"):
    m = _mover(name=name, move_type=move_type, industry=industry)
    return Article(title, "## 본문\n\n**강조** 텍스트\n\n| A | B |\n|---|---|\n| 1 | 2 |",
                   m, [], "gemini-2.0-flash", [])


def _config(tmp_path, token="test-token", site_id="12345", auto_publish=True):
    from src.config import Config
    return Config(
        llm_provider="gemini",
        anthropic_api_key="",
        gemini_api_key="test",
        naver_client_id="",
        naver_client_secret="",
        dart_api_key="",
        output_dir=tmp_path,
        timezone="Asia/Seoul",
        mover_threshold_pct=5.0,
        volume_surge_multiplier=3.0,
        top_n_movers=5,
        wp_access_token=token,
        wp_site_id=site_id,
        wp_auto_publish=auto_publish,
    )


# ── _md_to_html ──────────────────────────────────────────────────────
class TestMdToHtml:
    def test_header_conversion(self):
        html = _md_to_html("## 제목")
        # h2 태그에 font-size 인라인 스타일이 주입됨
        assert "<h2 style=" in html
        assert "font-size:16px" in html
        assert "제목</h2>" in html

    def test_bold_conversion(self):
        html = _md_to_html("**강조**")
        assert "<strong>" in html

    def test_table_conversion(self):
        md = "| A | B |\n|---|---|\n| 1 | 2 |"
        html = _md_to_html(md)
        assert "<table>" in html
        assert "<td>" in html

    def test_fenced_code(self):
        md = "```python\nprint('hello')\n```"
        html = _md_to_html(md)
        assert "<code" in html


# ── _build_tags ──────────────────────────────────────────────────────
class TestBuildTags:
    def test_surge_tags(self):
        art = _article(name="삼성전자", move_type="surge", industry="반도체")
        tags = _build_tags(art)
        assert "삼성전자" in tags
        assert "급등" in tags
        assert "주식" in tags
        assert "증시분석" in tags
        assert "KOSPI" in tags
        assert "반도체" in tags
        assert "급락" not in tags

    def test_plunge_tags(self):
        art = _article(move_type="plunge")
        tags = _build_tags(art)
        assert "급락" in tags
        assert "급등" not in tags

    def test_no_industry(self):
        art = _article(industry=None)
        tags = _build_tags(art)
        assert None not in tags


# ── _build_categories ────────────────────────────────────────────────
class TestBuildCategories:
    def test_surge_categories(self):
        art = _article(move_type="surge")
        cats = _build_categories(art)
        assert "종목분석" in cats
        assert "급등분석" in cats
        assert "KOSPI" in cats

    def test_plunge_categories(self):
        art = _article(move_type="plunge")
        cats = _build_categories(art)
        assert "급락분석" in cats
        assert "급등분석" not in cats


# ── publish_to_wordpress ─────────────────────────────────────────────
class TestPublishToWordpress:
    def test_no_credentials_returns_none(self, tmp_path):
        """크리덴셜 없으면 None 반환 (graceful skip)."""
        cfg = _config(tmp_path, token="", site_id="")
        art = _article()
        result = publish_to_wordpress(art, "2026-04-15", cfg)
        assert result is None

    def test_no_token_returns_none(self, tmp_path):
        cfg = _config(tmp_path, token="", site_id="12345")
        result = publish_to_wordpress(_article(), "2026-04-15", cfg)
        assert result is None

    def test_no_site_id_returns_none(self, tmp_path):
        cfg = _config(tmp_path, token="test-token", site_id="")
        result = publish_to_wordpress(_article(), "2026-04-15", cfg)
        assert result is None

    @patch("src.wordpress_publisher.requests.post")
    def test_successful_publish(self, mock_post, tmp_path):
        """성공 시 PublishResult 반환 + wp_posted.json 기록."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "ID": 42,
            "URL": "https://example.wordpress.com/?p=42",
        }
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        cfg = _config(tmp_path)
        art = _article()
        result = publish_to_wordpress(art, "2026-04-15", cfg)

        assert result is not None
        assert result.post_id == 42
        assert result.status == "draft"
        assert "example.wordpress.com" in result.url

        # wp_posted.json에 기록되었는지 확인
        posted_path = tmp_path / "wp_posted.json"
        assert posted_path.exists()
        posted = json.loads(posted_path.read_text(encoding="utf-8"))
        filename = art.filename("2026-04-15")
        assert filename in posted
        assert posted[filename]["post_id"] == 42

        # API 호출 검증
        call_args = mock_post.call_args
        assert "/posts/new" in call_args[0][0]
        payload = call_args[1]["json"]
        assert payload["status"] == "draft"
        assert payload["title"] == art.title

    @patch("src.wordpress_publisher.requests.post")
    def test_duplicate_skip(self, mock_post, tmp_path):
        """이미 발행된 글은 스킵한다."""
        art = _article()
        filename = art.filename("2026-04-15")

        # 이미 발행 기록이 있는 상태
        posted = {filename: {"post_id": 99, "url": "https://example.com/?p=99"}}
        (tmp_path / "wp_posted.json").write_text(
            json.dumps(posted, ensure_ascii=False), encoding="utf-8"
        )

        cfg = _config(tmp_path)
        result = publish_to_wordpress(art, "2026-04-15", cfg)

        assert result is None
        mock_post.assert_not_called()

    @patch("src.wordpress_publisher.requests.post")
    def test_http_error_returns_none(self, mock_post, tmp_path):
        """HTTP 에러 시 None 반환 (파이프라인 중단 안 함)."""
        import requests as req
        mock_post.side_effect = req.RequestException("403 Forbidden")

        cfg = _config(tmp_path)
        result = publish_to_wordpress(_article(), "2026-04-15", cfg)

        assert result is None

    @patch("src.wordpress_publisher.requests.post")
    def test_html_content_sent(self, mock_post, tmp_path):
        """본문이 HTML로 변환되어 전송되는지 확인."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"ID": 1, "URL": "https://x.com/?p=1"}
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        cfg = _config(tmp_path)
        publish_to_wordpress(_article(), "2026-04-15", cfg)

        payload = mock_post.call_args[1]["json"]
        assert "<h2 style=" in payload["content"]
        assert "<strong>" in payload["content"]

    @patch("src.wordpress_publisher.requests.post")
    def test_tags_and_categories_sent(self, mock_post, tmp_path):
        """태그와 카테고리가 전송되는지 확인."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"ID": 1, "URL": "https://x.com/?p=1"}
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        cfg = _config(tmp_path)
        publish_to_wordpress(_article(), "2026-04-15", cfg)

        payload = mock_post.call_args[1]["json"]
        assert "주식" in payload["tags"]
        assert "급등" in payload["tags"]
        assert "종목분석" in payload["categories"]


# ── publish_articles ─────────────────────────────────────────────────
class TestPublishArticles:
    @patch("src.wordpress_publisher.requests.post")
    def test_publishes_multiple(self, mock_post, tmp_path):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"ID": 1, "URL": "https://x.com/?p=1"}
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        cfg = _config(tmp_path)
        articles = [
            _article(name="종목A", title="A 분석"),
            _article(name="종목B", title="B 분석"),
        ]
        results = publish_articles(articles, "2026-04-15", cfg)
        assert len(results) == 2

    def test_no_credentials_empty_list(self, tmp_path):
        cfg = _config(tmp_path, token="", site_id="")
        results = publish_articles([_article()], "2026-04-15", cfg)
        assert results == []

    @patch("src.wordpress_publisher.requests.post")
    def test_partial_failure(self, mock_post, tmp_path):
        """일부 실패해도 나머지는 발행된다."""
        import requests as req

        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise req.RequestException("timeout")
            resp = MagicMock()
            resp.json.return_value = {"ID": 2, "URL": "https://x.com/?p=2"}
            resp.raise_for_status = MagicMock()
            return resp

        mock_post.side_effect = side_effect

        cfg = _config(tmp_path)
        articles = [
            _article(name="실패종목", title="실패"),
            _article(name="성공종목", title="성공"),
        ]
        results = publish_articles(articles, "2026-04-15", cfg)
        assert len(results) == 1
        assert results[0].title == "성공"


# ── publish_content_post ──────────────────────────────────────────────

def _content_post(title="시황 제목", content_type="daily_market"):
    return ContentPost(
        title=title,
        body="## 시황 본문\n\n내용입니다",
        content_type=content_type,
        model="gemini-2.0-flash",
        tags=["데일리시황", "KOSPI"],
        categories=["데일리시황", "시장분석"],
    )


class TestPublishContentPost:
    def test_no_credentials_returns_none(self, tmp_path):
        cfg = _config(tmp_path, token="", site_id="")
        result = publish_content_post(_content_post(), "2026-04-15", cfg)
        assert result is None

    @patch("src.wordpress_publisher.requests.post")
    def test_successful_publish(self, mock_post, tmp_path):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"ID": 100, "URL": "https://x.com/?p=100"}
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        cfg = _config(tmp_path)
        result = publish_content_post(_content_post(), "2026-04-15", cfg)

        assert result is not None
        assert result.post_id == 100
        assert result.status == "draft"

        payload = mock_post.call_args[1]["json"]
        assert "데일리시황" in payload["tags"]
        assert "시장분석" in payload["categories"]

    @patch("src.wordpress_publisher.requests.post")
    def test_duplicate_skip(self, mock_post, tmp_path):
        post = _content_post()
        filename = post.filename("2026-04-15")
        posted = {filename: {"post_id": 99, "url": "https://x.com/?p=99"}}
        (tmp_path / "wp_posted.json").write_text(
            json.dumps(posted, ensure_ascii=False), encoding="utf-8"
        )

        cfg = _config(tmp_path)
        result = publish_content_post(post, "2026-04-15", cfg)
        assert result is None
        mock_post.assert_not_called()

    @patch("src.wordpress_publisher.requests.post")
    def test_http_error_returns_none(self, mock_post, tmp_path):
        """HTTP 에러 시 None 반환 (파이프라인 중단 안 함)."""
        import requests as req
        mock_post.side_effect = req.RequestException("500 Internal Server Error")

        cfg = _config(tmp_path)
        result = publish_content_post(_content_post(), "2026-04-15", cfg)
        assert result is None


class TestPublishContentPosts:
    @patch("src.wordpress_publisher.requests.post")
    def test_publishes_multiple(self, mock_post, tmp_path):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"ID": 1, "URL": "https://x.com/?p=1"}
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        cfg = _config(tmp_path)
        posts = [
            _content_post(title="시황", content_type="daily_market"),
            _content_post(title="섹터", content_type="sector_report"),
        ]
        results = publish_content_posts(posts, "2026-04-15", cfg)
        assert len(results) == 2
