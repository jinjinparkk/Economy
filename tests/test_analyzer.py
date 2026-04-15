"""analyzer 모듈 단위 테스트."""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import patch, MagicMock

import pytest

from src.analyzer import (
    Article,
    FORBIDDEN_WORDS,
    _build_user_prompt,
    _check_forbidden,
    _parse_response,
    generate_article,
)
from src.detect_movers import Mover
from src.fetch_news import NewsItem


# ── 헬퍼 ──────────────────────────────────────────────────────────────
def _mover(move_type="surge", name="삼성전자", code="005930",
           change_pct=15.0, close=70000, volume=1000000,
           amount=5e10, marcap=4e14):
    return Mover(code, name, "KOSPI", move_type, close, change_pct,
                 volume, int(amount), int(marcap), reason_hints=["거래량급증"])


def _news(title="테스트 뉴스", press="한경"):
    return NewsItem(title, "설명", "http://x", press,
                    datetime(2026, 4, 9, 10, 0, tzinfo=timezone.utc))


# ── Article ──────────────────────────────────────────────────────────
class TestArticle:
    def test_to_markdown(self):
        art = Article("제목입니다", "본문입니다", _mover(), [], "gemini-2.0-flash", [])
        md = art.to_markdown()
        assert md.startswith("# 제목입니다")
        assert "본문입니다" in md

    def test_filename_surge(self):
        art = Article("T", "B", _mover(move_type="surge", name="삼성전자"),
                       [], "m", [])
        fn = art.filename("2026-04-09")
        assert fn == "2026-04-09_삼성전자_급등.md"

    def test_filename_plunge(self):
        art = Article("T", "B", _mover(move_type="plunge", name="LG전자"),
                       [], "m", [])
        fn = art.filename("2026-04-09")
        assert fn == "2026-04-09_LG전자_급락.md"

    def test_filename_special_chars(self):
        art = Article("T", "B", _mover(name="A/B&C(D)"), [], "m", [])
        fn = art.filename("2026-04-09")
        # 특수문자가 _ 로 치환
        assert "/" not in fn
        assert "&" not in fn


# ── _build_user_prompt ───────────────────────────────────────────────
class TestBuildUserPrompt:
    def test_includes_stock_info(self):
        m = _mover(name="코위버", code="056360", change_pct=30.0)
        prompt = _build_user_prompt(m, [])
        assert "코위버" in prompt
        assert "056360" in prompt
        assert "30.00%" in prompt or "+30.00%" in prompt

    def test_includes_news(self):
        m = _mover()
        news = [_news("KT와 양자보안 협약")]
        prompt = _build_user_prompt(m, news)
        assert "KT와 양자보안 협약" in prompt

    def test_no_news(self):
        prompt = _build_user_prompt(_mover(), [])
        assert "관련 뉴스 없음" in prompt

    def test_includes_index_summary(self):
        idx = {"KOSPI": {"Close": 2700.0, "ChangePct": -1.5}}
        prompt = _build_user_prompt(_mover(), [], index_summary=idx)
        assert "KOSPI" in prompt
        assert "2,700.00" in prompt

    def test_includes_hints(self):
        m = _mover()
        m.reason_hints = ["상한가", "거래량급증"]
        prompt = _build_user_prompt(m, [])
        assert "상한가" in prompt
        assert "거래량급증" in prompt

    def test_includes_macro_summary(self):
        macro = {
            "S&P 500": {"Close": 5400.0, "ChangePct": 1.89, "Change": 100.0},
            "원/달러": {"Close": 1380.0, "ChangePct": -0.5, "Change": -7.0},
        }
        prompt = _build_user_prompt(_mover(), [], macro_summary=macro)
        assert "S&P 500" in prompt
        assert "5,400.00" in prompt
        assert "원/달러" in prompt
        assert "글로벌 거시경제 지표" in prompt

    def test_includes_industry(self):
        m = _mover()
        m.industry = "반도체 제조업"
        prompt = _build_user_prompt(m, [])
        assert "반도체 제조업" in prompt

    def test_no_macro(self):
        prompt = _build_user_prompt(_mover(), [])
        assert "글로벌 거시경제 지표" not in prompt


# ── _check_forbidden ────────────────────────────────────────────────
class TestCheckForbidden:
    def test_clean_text(self):
        assert _check_forbidden("정상적인 분석 기사") == []

    def test_detects_forbidden(self):
        for word in FORBIDDEN_WORDS:
            warnings = _check_forbidden(f"이것은 {word} 포함 텍스트")
            assert len(warnings) >= 1, f"'{word}' not detected"

    def test_multiple_forbidden(self):
        text = "매수 추천 드리며 목표가는..."
        warnings = _check_forbidden(text)
        assert len(warnings) >= 2


# ── _parse_response ──────────────────────────────────────────────────
class TestParseResponse:
    def test_title_colon_format(self):
        text = "제목: [코위버] 30% 급등 분석\n\n## 본문 시작"
        title, body = _parse_response(text)
        assert title == "[코위버] 30% 급등 분석"
        assert "## 본문 시작" in body

    def test_title_h1_fallback(self):
        text = "# 급등 분석 기사\n\n본문 내용"
        title, body = _parse_response(text)
        assert title == "급등 분석 기사"

    def test_no_title(self):
        text = "그냥 본문만 있는 경우"
        title, body = _parse_response(text)
        assert title == "제목 추출 실패"

    def test_title_with_markdown(self):
        text = "제목: **[삼성전자] 분석**\n\n본문"
        title, body = _parse_response(text)
        assert "**" not in title  # 마크다운 기호 제거
        assert "삼성전자" in title

    def test_full_colon_format(self):
        text = "제목： 한글 콜론도 인식\n\n본문"
        title, body = _parse_response(text)
        assert title == "한글 콜론도 인식"


# ── generate_article (LLM 모킹) ────────────────────────────────────
class TestGenerateArticle:
    @patch("src.analyzer._generate_with_gemini")
    def test_gemini_success(self, mock_gemini):
        mock_gemini.return_value = (
            "제목: [코위버] +30% 급등 이유\n\n## 주가 현황\n종가 9,100원.\n\n"
            "---\n> 면책: 본 글은 투자 권유가 아닙니다.",
            "gemini-2.0-flash",
        )
        from src.config import Config
        import pathlib
        cfg = Config(
            llm_provider="gemini", anthropic_api_key="",
            gemini_api_key="fake-key",
            naver_client_id="", naver_client_secret="", dart_api_key="",
            output_dir=pathlib.Path("."), timezone="Asia/Seoul",
            mover_threshold_pct=5.0, volume_surge_multiplier=3.0, top_n_movers=5,
            wp_access_token="", wp_site_id="", wp_auto_publish=False,
        )
        m = _mover(name="코위버", code="056360", change_pct=30.0)
        art = generate_article(m, [_news()], cfg, provider="gemini")
        assert "코위버" in art.title
        assert art.model == "gemini-2.0-flash"
        assert art.warnings == []

    @patch("src.analyzer._generate_with_claude")
    def test_claude_success(self, mock_claude):
        mock_claude.return_value = (
            "제목: [삼성전자] 분석\n\n본문",
            "claude-sonnet-4-6",
        )
        from src.config import Config
        import pathlib
        cfg = Config(
            llm_provider="claude", anthropic_api_key="fake-key",
            gemini_api_key="",
            naver_client_id="", naver_client_secret="", dart_api_key="",
            output_dir=pathlib.Path("."), timezone="Asia/Seoul",
            mover_threshold_pct=5.0, volume_surge_multiplier=3.0, top_n_movers=5,
            wp_access_token="", wp_site_id="", wp_auto_publish=False,
        )
        art = generate_article(_mover(), [_news()], cfg, provider="claude")
        assert art.model == "claude-sonnet-4-6"

    def test_missing_gemini_key_raises(self):
        from src.config import Config
        import pathlib
        cfg = Config(
            llm_provider="gemini", anthropic_api_key="",
            gemini_api_key="",
            naver_client_id="", naver_client_secret="", dart_api_key="",
            output_dir=pathlib.Path("."), timezone="Asia/Seoul",
            mover_threshold_pct=5.0, volume_surge_multiplier=3.0, top_n_movers=5,
            wp_access_token="", wp_site_id="", wp_auto_publish=False,
        )
        with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
            generate_article(_mover(), [], cfg, provider="gemini")

    def test_unknown_provider_raises(self):
        from src.config import Config
        import pathlib
        cfg = Config(
            llm_provider="unknown", anthropic_api_key="",
            gemini_api_key="",
            naver_client_id="", naver_client_secret="", dart_api_key="",
            output_dir=pathlib.Path("."), timezone="Asia/Seoul",
            mover_threshold_pct=5.0, volume_surge_multiplier=3.0, top_n_movers=5,
            wp_access_token="", wp_site_id="", wp_auto_publish=False,
        )
        with pytest.raises(ValueError, match="unknown"):
            generate_article(_mover(), [], cfg, provider="unknown")

    @patch("src.analyzer._generate_with_gemini")
    def test_forbidden_word_warning(self, mock_gemini):
        mock_gemini.return_value = (
            "제목: 분석\n\n이 종목을 매수 추천합니다. 목표가 10만원.",
            "gemini-2.0-flash",
        )
        from src.config import Config
        import pathlib
        cfg = Config(
            llm_provider="gemini", anthropic_api_key="",
            gemini_api_key="fake-key",
            naver_client_id="", naver_client_secret="", dart_api_key="",
            output_dir=pathlib.Path("."), timezone="Asia/Seoul",
            mover_threshold_pct=5.0, volume_surge_multiplier=3.0, top_n_movers=5,
            wp_access_token="", wp_site_id="", wp_auto_publish=False,
        )
        art = generate_article(_mover(), [], cfg, provider="gemini")
        assert len(art.warnings) >= 2  # 매수 추천, 목표가
