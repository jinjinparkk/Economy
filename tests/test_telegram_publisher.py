"""telegram_publisher 모듈 단위 테스트."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from src.analyzer import Article
from src.content_post import ContentPost
from src.detect_movers import Mover
from src.telegram_publisher import (
    _extract_summary,
    _format_message,
    _escape_html,
    send_to_telegram,
    publish_article_to_telegram,
    publish_articles_to_telegram,
    publish_content_post_to_telegram,
    publish_content_posts_to_telegram,
)
from src.wordpress_publisher import PublishResult


# ── 헬퍼 ──────────────────────────────────────────────────────────────

def _mover(name="테스트종목", code="000001", move_type="surge",
           change_pct=15.0, close=5000, amount=2e9, market="KOSPI",
           industry="반도체"):
    return Mover(code, name, market, move_type, close, change_pct,
                 100000, int(amount), int(1e11), industry=industry)


def _article(name="테스트종목", title="테스트 급등 분석", move_type="surge",
             industry="반도체"):
    m = _mover(name=name, move_type=move_type, industry=industry)
    return Article(title, "## 30초 요약\n\n- 포인트1\n- 포인트2\n- 포인트3\n\n## 본문\n\n내용입니다",
                   m, [], "gemini-2.0-flash", [])


def _content_post(title="시황 제목", content_type="daily_market"):
    return ContentPost(
        title=title,
        body="## 30초 요약\n\n- 시황 포인트1\n- 시황 포인트2\n\n## 상세분석\n\n본문입니다",
        content_type=content_type,
        model="gemini-2.0-flash",
        tags=["데일리시황", "KOSPI"],
        categories=["데일리시황", "시장분석"],
    )


def _config(tmp_path, token="test-token:123", channel_id="@test_channel",
            auto_post=True):
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
        wp_access_token="",
        wp_site_id="",
        wp_auto_publish=False,
        telegram_bot_token=token,
        telegram_channel_id=channel_id,
        telegram_auto_post=auto_post,
        naver_access_token="",
        naver_auto_publish=False,
    )


# ── _extract_summary ─────────────────────────────────────────────────

class TestExtractSummary:
    def test_extracts_30sec_summary(self):
        body = "## 30초 요약\n\n- 포인트1\n- 포인트2\n- 포인트3\n\n## 본문"
        summary = _extract_summary(body)
        assert "포인트1" in summary
        assert "포인트2" in summary
        assert "포인트3" in summary

    def test_stops_at_next_heading(self):
        body = "## 30초 요약\n\n- 핵심1\n- 핵심2\n\n## 상세분석\n\n본문"
        summary = _extract_summary(body)
        assert "핵심1" in summary
        assert "상세분석" not in summary

    def test_fallback_to_bullets(self):
        body = "# 제목\n\n- 불릿1\n- 불릿2\n\n내용"
        summary = _extract_summary(body)
        assert "불릿1" in summary

    def test_fallback_to_first_lines(self):
        body = "# 제목\n\n첫번째 문장입니다.\n두번째 문장입니다.\n세번째 문장입니다."
        summary = _extract_summary(body)
        assert "첫번째" in summary

    def test_empty_body(self):
        summary = _extract_summary("")
        assert summary == ""

    def test_max_5_lines(self):
        lines = "\n".join([f"- 포인트{i}" for i in range(10)])
        body = f"## 30초 요약\n\n{lines}\n\n## 본문"
        summary = _extract_summary(body)
        assert summary.count("포인트") == 5


# ── _escape_html ─────────────────────────────────────────────────────

class TestEscapeHtml:
    def test_escapes_ampersand(self):
        assert _escape_html("A & B") == "A &amp; B"

    def test_escapes_angle_brackets(self):
        assert _escape_html("<b>") == "&lt;b&gt;"

    def test_no_change_normal_text(self):
        assert _escape_html("안녕하세요") == "안녕하세요"


# ── _format_message ──────────────────────────────────────────────────

class TestFormatMessage:
    def test_includes_title(self):
        msg = _format_message("삼성전자 급등 분석", "## 30초 요약\n\n- 포인트1")
        assert "<b>삼성전자 급등 분석</b>" in msg

    def test_includes_summary(self):
        msg = _format_message("제목", "## 30초 요약\n\n- 핵심 포인트\n\n## 본문")
        assert "핵심 포인트" in msg

    def test_includes_wp_url(self):
        msg = _format_message("제목", "본문", wp_url="https://example.com/?p=42")
        assert '<a href="https://example.com/?p=42">전문 보기</a>' in msg

    def test_no_wp_url(self):
        msg = _format_message("제목", "본문")
        assert "전문 보기" not in msg

    def test_includes_disclaimer(self):
        msg = _format_message("제목", "본문")
        assert "투자 판단은 본인 책임" in msg

    def test_html_parse_mode_safe(self):
        msg = _format_message("A <B> & C", "## 30초 요약\n\n- 포인트")
        assert "&lt;B&gt;" in msg
        assert "&amp;" in msg

    def test_max_length_enforced(self):
        long_body = "## 30초 요약\n\n" + ("- " + "X" * 200 + "\n") * 30 + "\n## 본문"
        msg = _format_message("제목", long_body)
        assert len(msg) <= 4096

    def test_bullet_replacement(self):
        msg = _format_message("제목", "## 30초 요약\n\n- 항목1\n- 항목2")
        assert "▸" in msg


# ── send_to_telegram ─────────────────────────────────────────────────

class TestSendToTelegram:
    def test_no_credentials_returns_none(self, tmp_path):
        cfg = _config(tmp_path, token="", channel_id="")
        result = send_to_telegram("제목", "본문", cfg)
        assert result is None

    def test_no_token_returns_none(self, tmp_path):
        cfg = _config(tmp_path, token="", channel_id="@ch")
        result = send_to_telegram("제목", "본문", cfg)
        assert result is None

    def test_no_channel_returns_none(self, tmp_path):
        cfg = _config(tmp_path, token="tok", channel_id="")
        result = send_to_telegram("제목", "본문", cfg)
        assert result is None

    @patch("src.telegram_publisher.requests.post")
    def test_successful_send(self, mock_post, tmp_path):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "ok": True,
            "result": {"message_id": 42},
        }
        mock_post.return_value = mock_resp

        cfg = _config(tmp_path)
        result = send_to_telegram("테스트 제목", "## 30초 요약\n\n- 포인트1", cfg)

        assert result is not None
        assert result.post_id == 42
        assert result.status == "sent"
        assert result.title == "테스트 제목"

        # API 호출 검증
        call_args = mock_post.call_args
        payload = call_args[1]["json"]
        assert payload["parse_mode"] == "HTML"
        assert payload["chat_id"] == "@test_channel"

    @patch("src.telegram_publisher.requests.post")
    def test_http_error_returns_none(self, mock_post, tmp_path):
        mock_resp = MagicMock()
        mock_resp.status_code = 400
        mock_resp.text = "Bad Request"
        mock_post.return_value = mock_resp

        cfg = _config(tmp_path)
        result = send_to_telegram("제목", "본문", cfg)
        assert result is None

    @patch("src.telegram_publisher.requests.post")
    def test_api_error_not_ok(self, mock_post, tmp_path):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "ok": False,
            "description": "Forbidden: bot is not a member",
        }
        mock_post.return_value = mock_resp

        cfg = _config(tmp_path)
        result = send_to_telegram("제목", "본문", cfg)
        assert result is None

    @patch("src.telegram_publisher.requests.post")
    def test_request_exception_returns_none(self, mock_post, tmp_path):
        import requests as req
        mock_post.side_effect = req.RequestException("Connection refused")

        cfg = _config(tmp_path)
        result = send_to_telegram("제목", "본문", cfg)
        assert result is None

    @patch("src.telegram_publisher.time.sleep")
    @patch("src.telegram_publisher.requests.post")
    def test_429_retry_then_success(self, mock_post, mock_sleep, tmp_path):
        """429 rate limit 시 retry_after 대기 후 재시도."""
        resp_429 = MagicMock()
        resp_429.status_code = 429
        resp_429.json.return_value = {"parameters": {"retry_after": 3}}

        resp_ok = MagicMock()
        resp_ok.status_code = 200
        resp_ok.json.return_value = {
            "ok": True,
            "result": {"message_id": 99},
        }

        mock_post.side_effect = [resp_429, resp_ok]

        cfg = _config(tmp_path)
        result = send_to_telegram("제목", "본문", cfg)

        assert result is not None
        assert result.post_id == 99
        mock_sleep.assert_called_once_with(3)

    @patch("src.telegram_publisher.time.sleep")
    @patch("src.telegram_publisher.requests.post")
    def test_429_exhausted_returns_none(self, mock_post, mock_sleep, tmp_path):
        """429가 3회 연속이면 None 반환."""
        resp_429 = MagicMock()
        resp_429.status_code = 429
        resp_429.json.return_value = {"parameters": {"retry_after": 1}}
        mock_post.return_value = resp_429

        cfg = _config(tmp_path)
        result = send_to_telegram("제목", "본문", cfg)

        assert result is None
        assert mock_sleep.call_count == 2  # 첫 2번만 재시도 대기


# ── publish_article_to_telegram ──────────────────────────────────────

class TestPublishArticleToTelegram:
    def test_no_credentials_returns_none(self, tmp_path):
        cfg = _config(tmp_path, token="", channel_id="")
        result = publish_article_to_telegram(_article(), "2026-04-15", cfg)
        assert result is None

    @patch("src.telegram_publisher.requests.post")
    def test_successful_publish(self, mock_post, tmp_path):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "ok": True,
            "result": {"message_id": 42},
        }
        mock_post.return_value = mock_resp

        cfg = _config(tmp_path)
        art = _article()
        result = publish_article_to_telegram(art, "2026-04-15", cfg)

        assert result is not None
        assert result.post_id == 42
        assert result.status == "sent"

        # telegram_posted.json에 기록되었는지 확인
        posted_path = tmp_path / "telegram_posted.json"
        assert posted_path.exists()
        posted = json.loads(posted_path.read_text(encoding="utf-8"))
        filename = art.filename("2026-04-15")
        assert filename in posted
        assert posted[filename]["msg_id"] == 42

    @patch("src.telegram_publisher.requests.post")
    def test_duplicate_skip(self, mock_post, tmp_path):
        """이미 발송된 글은 스킵한다."""
        art = _article()
        filename = art.filename("2026-04-15")
        posted = {filename: {"msg_id": 99, "title": art.title}}
        (tmp_path / "telegram_posted.json").write_text(
            json.dumps(posted, ensure_ascii=False), encoding="utf-8"
        )

        cfg = _config(tmp_path)
        result = publish_article_to_telegram(art, "2026-04-15", cfg)

        assert result is None
        mock_post.assert_not_called()

    @patch("src.telegram_publisher.requests.post")
    def test_includes_wp_url_if_exists(self, mock_post, tmp_path):
        """wp_posted.json에 URL이 있으면 전문 보기 링크를 포함한다."""
        art = _article()
        filename = art.filename("2026-04-15")

        # WordPress 발행 기록 설정
        wp_posted = {filename: {"post_id": 10, "url": "https://example.com/?p=10"}}
        (tmp_path / "wp_posted.json").write_text(
            json.dumps(wp_posted, ensure_ascii=False), encoding="utf-8"
        )

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "ok": True,
            "result": {"message_id": 50},
        }
        mock_post.return_value = mock_resp

        cfg = _config(tmp_path)
        result = publish_article_to_telegram(art, "2026-04-15", cfg)

        assert result is not None
        # 전송된 메시지에 WP URL이 포함되어야 함
        payload = mock_post.call_args[1]["json"]
        assert "example.com" in payload["text"]
        assert "전문 보기" in payload["text"]


# ── publish_content_post_to_telegram ─────────────────────────────────

class TestPublishContentPostToTelegram:
    def test_no_credentials_returns_none(self, tmp_path):
        cfg = _config(tmp_path, token="", channel_id="")
        result = publish_content_post_to_telegram(_content_post(), "2026-04-15", cfg)
        assert result is None

    @patch("src.telegram_publisher.requests.post")
    def test_successful_publish(self, mock_post, tmp_path):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "ok": True,
            "result": {"message_id": 100},
        }
        mock_post.return_value = mock_resp

        cfg = _config(tmp_path)
        result = publish_content_post_to_telegram(_content_post(), "2026-04-15", cfg)

        assert result is not None
        assert result.post_id == 100
        assert result.status == "sent"

    @patch("src.telegram_publisher.requests.post")
    def test_duplicate_skip(self, mock_post, tmp_path):
        post = _content_post()
        filename = post.filename("2026-04-15")
        posted = {filename: {"msg_id": 99, "title": post.title}}
        (tmp_path / "telegram_posted.json").write_text(
            json.dumps(posted, ensure_ascii=False), encoding="utf-8"
        )

        cfg = _config(tmp_path)
        result = publish_content_post_to_telegram(post, "2026-04-15", cfg)
        assert result is None
        mock_post.assert_not_called()


# ── publish_articles_to_telegram (배치) ──────────────────────────────

class TestPublishArticlesToTelegram:
    @patch("src.telegram_publisher.requests.post")
    def test_publishes_multiple(self, mock_post, tmp_path):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "ok": True,
            "result": {"message_id": 1},
        }
        mock_post.return_value = mock_resp

        cfg = _config(tmp_path)
        articles = [
            _article(name="종목A", title="A 분석"),
            _article(name="종목B", title="B 분석"),
        ]
        results = publish_articles_to_telegram(articles, "2026-04-15", cfg)
        assert len(results) == 2

    def test_no_credentials_empty_list(self, tmp_path):
        cfg = _config(tmp_path, token="", channel_id="")
        results = publish_articles_to_telegram([_article()], "2026-04-15", cfg)
        assert results == []

    @patch("src.telegram_publisher.requests.post")
    def test_partial_failure(self, mock_post, tmp_path):
        """일부 실패해도 나머지는 발송된다."""
        import requests as req

        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise req.RequestException("timeout")
            resp = MagicMock()
            resp.status_code = 200
            resp.json.return_value = {
                "ok": True,
                "result": {"message_id": 2},
            }
            return resp

        mock_post.side_effect = side_effect

        cfg = _config(tmp_path)
        articles = [
            _article(name="실패종목", title="실패"),
            _article(name="성공종목", title="성공"),
        ]
        results = publish_articles_to_telegram(articles, "2026-04-15", cfg)
        assert len(results) == 1
        assert results[0].title == "성공"


# ── publish_content_posts_to_telegram (배치) ─────────────────────────

class TestPublishContentPostsToTelegram:
    @patch("src.telegram_publisher.requests.post")
    def test_publishes_multiple(self, mock_post, tmp_path):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "ok": True,
            "result": {"message_id": 1},
        }
        mock_post.return_value = mock_resp

        cfg = _config(tmp_path)
        posts = [
            _content_post(title="시황", content_type="daily_market"),
            _content_post(title="섹터", content_type="sector_report"),
        ]
        results = publish_content_posts_to_telegram(posts, "2026-04-15", cfg)
        assert len(results) == 2
