"""content_post + content_generator 모듈 단위 테스트."""
from __future__ import annotations

import pathlib
from unittest.mock import patch, MagicMock

import pytest

from src.config import Config
from src.content_post import ContentPost
from src.content_generator import (
    build_daily_market_prompt,
    build_sector_report_prompt,
    build_quant_insight_prompt,
    build_pre_market_prompt,
    generate_daily_market,
    generate_sector_report,
    generate_quant_insight,
    generate_pre_market,
    _dispatch_llm,
    _parse_content_response,
)
from src.predictor import OutlookData, PatternStats, DirectionPrediction
from src.technical import TechnicalIndicators


# ── 헬퍼 ──────────────────────────────────────────────────────────────

def _config(provider="gemini"):
    return Config(
        llm_provider=provider,
        anthropic_api_key="fake-claude-key" if provider == "claude" else "",
        gemini_api_key="fake-gemini-key" if provider == "gemini" else "",
        naver_client_id="",
        naver_client_secret="",
        dart_api_key="",
        output_dir=pathlib.Path("."),
        timezone="Asia/Seoul",
        mover_threshold_pct=5.0,
        volume_surge_multiplier=3.0,
        top_n_movers=5,
        wp_access_token="",
        wp_site_id="",
        wp_auto_publish=False,
    )


def _macro_summary():
    return {
        "S&P 500": {"Close": 5400.0, "ChangePct": 1.89, "Change": 100.0},
        "원/달러": {"Close": 1380.0, "ChangePct": 0.73, "Change": 10.0},
        "_yield_spread": 0.45,
        "_market_regime": "보통",
    }


def _breadth():
    return {
        "total_up": 600,
        "total_down": 400,
        "total_unchanged": 50,
        "total": 1050,
        "up_ratio": 57.1,
    }


def _sector_breadth():
    return {
        "반도체": {"up_count": 10, "down_count": 5, "total": 15, "avg_change_pct": 2.5},
        "자동차": {"up_count": 3, "down_count": 7, "total": 10, "avg_change_pct": -1.2},
        "바이오": {"up_count": 8, "down_count": 2, "total": 10, "avg_change_pct": 3.1},
        "건설": {"up_count": 2, "down_count": 8, "total": 10, "avg_change_pct": -2.0},
        "철강": {"up_count": 4, "down_count": 6, "total": 10, "avg_change_pct": -0.5},
        "은행": {"up_count": 6, "down_count": 4, "total": 10, "avg_change_pct": 1.0},
    }


def _outlook_map():
    tech = TechnicalIndicators(
        rsi_14=72.5,
        macd=0.005,
        macd_signal=0.003,
        macd_histogram=0.002,
        bb_position="상단근접",
        ma_trend="정배열",
        ma_5=5000.0,
        ma_20=4800.0,
        ma_60=4500.0,
        volume_ratio=2.5,
        signal_summary="단기 과열, 매수 시그널 유지",
    )
    pattern = PatternStats(
        event_type="급등(5%+)",
        sample_count=50,
        avg_return_1d=1.5,
        avg_return_5d=2.0,
        positive_rate_1d=60.0,
        positive_rate_5d=55.0,
    )
    pred = DirectionPrediction(
        direction="상승",
        confidence=0.72,
        features_used=["RSI", "MACD_hist"],
        confidence_grade="보통",
        cv_accuracy=0.65,
    )
    tech2 = TechnicalIndicators(
        rsi_14=28.0,
        macd=-0.01,
        macd_signal=-0.005,
        bb_position="하단근접",
        ma_trend="역배열",
        ma_5=3000.0,
        ma_20=3200.0,
        ma_60=3500.0,
    )
    pred2 = DirectionPrediction(
        direction="하락",
        confidence=0.55,
        features_used=["RSI"],
        confidence_grade="낮음",
    )
    return {
        "005930": OutlookData(technical=tech, pattern=pattern, prediction=pred),
        "000660": OutlookData(technical=tech2, prediction=pred2),
    }


# ═══════════════════════════════════════════════════════════════════════
# ContentPost 테스트
# ═══════════════════════════════════════════════════════════════════════

class TestContentPost:
    def test_to_markdown(self):
        post = ContentPost("시황 제목", "본문입니다", "daily_market", "gemini-2.0-flash")
        md = post.to_markdown()
        assert md == "# 시황 제목\n\n본문입니다\n"

    def test_filename_daily_market(self):
        post = ContentPost("제목", "본문", "daily_market", "m")
        assert post.filename("2026-04-15") == "2026-04-15_데일리시황.md"

    def test_filename_sector_report(self):
        post = ContentPost("제목", "본문", "sector_report", "m")
        assert post.filename("2026-04-15") == "2026-04-15_섹터리포트.md"

    def test_filename_quant_insight(self):
        post = ContentPost("제목", "본문", "quant_insight", "m")
        assert post.filename("2026-04-15") == "2026-04-15_퀀트인사이트.md"

    def test_filename_unknown_type(self):
        post = ContentPost("제목", "본문", "unknown_type", "m")
        assert post.filename("2026-04-15") == "2026-04-15_unknown_type.md"

    def test_default_lists(self):
        post = ContentPost("제목", "본문", "daily_market", "m")
        assert post.tags == []
        assert post.categories == []
        assert post.warnings == []


# ═══════════════════════════════════════════════════════════════════════
# 프롬프트 빌더 테스트
# ═══════════════════════════════════════════════════════════════════════

class TestBuildDailyMarketPrompt:
    def test_contains_macro(self):
        prompt = build_daily_market_prompt(
            macro_summary=_macro_summary(),
            trade_date="2026-04-15",
        )
        assert "S&P 500" in prompt
        assert "원/달러" in prompt
        assert "장단기 금리차" in prompt
        assert "시장 체제" in prompt

    def test_contains_breadth(self):
        prompt = build_daily_market_prompt(
            breadth=_breadth(),
            trade_date="2026-04-15",
        )
        assert "상승 종목: 600" in prompt
        assert "하락 종목: 400" in prompt
        assert "57.1%" in prompt

    def test_contains_index_summary(self):
        prompt = build_daily_market_prompt(
            index_summary={"KOSPI": {"Close": 2700.0, "ChangePct": -1.0}},
            trade_date="2026-04-15",
        )
        assert "KOSPI" in prompt
        assert "2,700.00" in prompt

    def test_contains_movers(self):
        prompt = build_daily_market_prompt(
            surges_summary=["삼성전자 +15%", "SK하이닉스 +10%"],
            plunges_summary=["카카오 -8%"],
            trade_date="2026-04-15",
        )
        assert "삼성전자 +15%" in prompt
        assert "카카오 -8%" in prompt

    def test_contains_sector_top_bottom(self):
        prompt = build_daily_market_prompt(
            sector_breadth=_sector_breadth(),
            trade_date="2026-04-15",
        )
        assert "상위 5개 업종" in prompt
        assert "하위 5개 업종" in prompt

    def test_minimal_prompt(self):
        prompt = build_daily_market_prompt(trade_date="2026-04-15")
        assert "2026-04-15" in prompt
        assert "작성 지침" in prompt


class TestBuildSectorReportPrompt:
    def test_contains_all_sectors(self):
        prompt = build_sector_report_prompt(
            sector_breadth=_sector_breadth(),
            trade_date="2026-04-15",
        )
        assert "반도체" in prompt
        assert "자동차" in prompt
        assert "강세 업종 TOP 5" in prompt
        assert "약세 업종 TOP 5" in prompt

    def test_contains_macro_narrative(self):
        prompt = build_sector_report_prompt(
            macro_narrative="위험자산 선호 흐름",
            trade_date="2026-04-15",
        )
        assert "위험자산 선호 흐름" in prompt

    def test_contains_movers_by_sector(self):
        prompt = build_sector_report_prompt(
            surges_by_sector={"반도체": ["삼성전자 +15%"]},
            plunges_by_sector={"자동차": ["현대차 -5%"]},
            trade_date="2026-04-15",
        )
        assert "삼성전자 +15%" in prompt
        assert "현대차 -5%" in prompt

    def test_minimal_prompt(self):
        prompt = build_sector_report_prompt(trade_date="2026-04-15")
        assert "2026-04-15" in prompt
        assert "작성 지침" in prompt


class TestBuildQuantInsightPrompt:
    def test_contains_rsi_distribution(self):
        prompt = build_quant_insight_prompt(
            outlook_map=_outlook_map(),
            trade_date="2026-04-15",
        )
        assert "RSI(14) 분포" in prompt
        assert "과매수" in prompt
        assert "과매도" in prompt

    def test_contains_macd_signals(self):
        prompt = build_quant_insight_prompt(
            outlook_map=_outlook_map(),
            trade_date="2026-04-15",
        )
        assert "MACD 시그널 분포" in prompt
        assert "매수 시그널" in prompt

    def test_contains_ml_predictions(self):
        prompt = build_quant_insight_prompt(
            outlook_map=_outlook_map(),
            trade_date="2026-04-15",
        )
        assert "ML 방향 예측" in prompt
        assert "상승 예측" in prompt
        assert "하락 예측" in prompt

    def test_contains_pattern_stats(self):
        prompt = build_quant_insight_prompt(
            outlook_map=_outlook_map(),
            trade_date="2026-04-15",
        )
        assert "과거 유사 패턴" in prompt
        assert "익일 평균 수익률" in prompt

    def test_contains_bb_distribution(self):
        prompt = build_quant_insight_prompt(
            outlook_map=_outlook_map(),
            trade_date="2026-04-15",
        )
        assert "볼린저밴드 위치 분포" in prompt

    def test_contains_ma_trend(self):
        prompt = build_quant_insight_prompt(
            outlook_map=_outlook_map(),
            trade_date="2026-04-15",
        )
        assert "이동평균 트렌드 분포" in prompt
        assert "정배열" in prompt
        assert "역배열" in prompt

    def test_empty_map(self):
        prompt = build_quant_insight_prompt(
            outlook_map={},
            trade_date="2026-04-15",
        )
        assert "데이터 없음" in prompt or "데이터가 부족" in prompt
        assert "면책 문구" in prompt

    def test_none_map(self):
        prompt = build_quant_insight_prompt(
            outlook_map=None,
            trade_date="2026-04-15",
        )
        assert "데이터 없음" in prompt or "데이터가 부족" in prompt
        assert "면책 문구" in prompt


# ═══════════════════════════════════════════════════════════════════════
# _dispatch_llm 테스트
# ═══════════════════════════════════════════════════════════════════════

class TestDispatchLlm:
    @patch("src.content_generator._generate_with_gemini")
    def test_gemini_dispatch(self, mock_gemini):
        mock_gemini.return_value = ("텍스트", "gemini-2.0-flash")
        raw, model = _dispatch_llm("sys", "user", _config("gemini"))
        assert model == "gemini-2.0-flash"
        mock_gemini.assert_called_once()

    @patch("src.content_generator._generate_with_claude")
    def test_claude_dispatch(self, mock_claude):
        mock_claude.return_value = ("텍스트", "claude-sonnet-4-6")
        raw, model = _dispatch_llm("sys", "user", _config("claude"))
        assert model == "claude-sonnet-4-6"
        mock_claude.assert_called_once()

    def test_unknown_provider_raises(self):
        cfg = Config(
            llm_provider="unknown",
            anthropic_api_key="", gemini_api_key="",
            naver_client_id="", naver_client_secret="", dart_api_key="",
            output_dir=pathlib.Path("."), timezone="Asia/Seoul",
            mover_threshold_pct=5.0, volume_surge_multiplier=3.0, top_n_movers=5,
            wp_access_token="", wp_site_id="", wp_auto_publish=False,
        )
        with pytest.raises(ValueError, match="unknown"):
            _dispatch_llm("sys", "user", cfg)

    def test_no_gemini_key_raises(self):
        cfg = Config(
            llm_provider="gemini", anthropic_api_key="", gemini_api_key="",
            naver_client_id="", naver_client_secret="", dart_api_key="",
            output_dir=pathlib.Path("."), timezone="Asia/Seoul",
            mover_threshold_pct=5.0, volume_surge_multiplier=3.0, top_n_movers=5,
            wp_access_token="", wp_site_id="", wp_auto_publish=False,
        )
        with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
            _dispatch_llm("sys", "user", cfg)

    def test_no_claude_key_raises(self):
        cfg = Config(
            llm_provider="claude", anthropic_api_key="", gemini_api_key="",
            naver_client_id="", naver_client_secret="", dart_api_key="",
            output_dir=pathlib.Path("."), timezone="Asia/Seoul",
            mover_threshold_pct=5.0, volume_surge_multiplier=3.0, top_n_movers=5,
            wp_access_token="", wp_site_id="", wp_auto_publish=False,
        )
        with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
            _dispatch_llm("sys", "user", cfg)


# ═══════════════════════════════════════════════════════════════════════
# _parse_content_response 테스트
# ═══════════════════════════════════════════════════════════════════════

class TestParseContentResponse:
    def test_normal_response(self):
        raw = "제목: 시황 분석 제목\n\n## 본문\n내용입니다"
        title, body = _parse_content_response(raw, "2026-04-15", "데일리시황")
        assert title == "시황 분석 제목"
        assert "본문" in body

    def test_fallback_title(self):
        raw = "본문만 있는 응답"
        title, body = _parse_content_response(raw, "2026-04-15", "데일리시황")
        assert title == "2026-04-15 데일리시황"


# ═══════════════════════════════════════════════════════════════════════
# 생성 함수 테스트 (LLM mock)
# ═══════════════════════════════════════════════════════════════════════

class TestGenerateDailyMarket:
    @patch("src.content_generator._generate_with_gemini")
    def test_returns_content_post(self, mock_gemini):
        mock_gemini.return_value = (
            "제목: 2026년 04월 15일 시황 — KOSPI 상승\n\n## 매크로\n본문",
            "gemini-2.0-flash",
        )
        post = generate_daily_market(
            macro_summary=_macro_summary(),
            breadth=_breadth(),
            trade_date="2026-04-15",
            config=_config(),
        )
        assert isinstance(post, ContentPost)
        assert post.content_type == "daily_market"
        assert post.model == "gemini-2.0-flash"
        assert "데일리시황" in post.categories
        assert "시장분석" in post.categories
        assert "KOSPI" in post.tags

    @patch("src.content_generator._generate_with_gemini")
    def test_forbidden_word_detected(self, mock_gemini):
        mock_gemini.return_value = (
            "제목: 시황\n\n지금 사야 합니다",
            "gemini-2.0-flash",
        )
        post = generate_daily_market(trade_date="2026-04-15", config=_config())
        assert len(post.warnings) > 0
        assert any("지금 사야" in w for w in post.warnings)


class TestGenerateSectorReport:
    @patch("src.content_generator._generate_with_gemini")
    def test_returns_content_post(self, mock_gemini):
        mock_gemini.return_value = (
            "제목: 섹터 리포트\n\n## 섹터 지형도\n본문",
            "gemini-2.0-flash",
        )
        post = generate_sector_report(
            sector_breadth=_sector_breadth(),
            trade_date="2026-04-15",
            config=_config(),
        )
        assert isinstance(post, ContentPost)
        assert post.content_type == "sector_report"
        assert "섹터리포트" in post.categories
        assert "섹터분석" in post.tags

    @patch("src.content_generator._generate_with_claude")
    def test_claude_provider(self, mock_claude):
        mock_claude.return_value = (
            "제목: 섹터 리포트\n\n본문",
            "claude-sonnet-4-6",
        )
        post = generate_sector_report(
            trade_date="2026-04-15",
            config=_config("claude"),
        )
        assert post.model == "claude-sonnet-4-6"
        mock_claude.assert_called_once()


class TestGenerateQuantInsight:
    @patch("src.content_generator._generate_with_gemini")
    def test_returns_content_post(self, mock_gemini):
        mock_gemini.return_value = (
            "제목: 퀀트 인사이트\n\n## RSI 분포\n본문",
            "gemini-2.0-flash",
        )
        post = generate_quant_insight(
            outlook_map=_outlook_map(),
            trade_date="2026-04-15",
            config=_config(),
        )
        assert isinstance(post, ContentPost)
        assert post.content_type == "quant_insight"
        assert "퀀트인사이트" in post.categories
        assert "기술적분석" in post.categories
        assert "퀀트" in post.tags

    @patch("src.content_generator._generate_with_gemini")
    def test_empty_outlook_map(self, mock_gemini):
        mock_gemini.return_value = (
            "제목: 퀀트 인사이트\n\n데이터 부족",
            "gemini-2.0-flash",
        )
        post = generate_quant_insight(
            outlook_map={},
            trade_date="2026-04-15",
            config=_config(),
        )
        assert isinstance(post, ContentPost)

    @patch("src.content_generator._generate_with_gemini")
    def test_none_outlook_map(self, mock_gemini):
        mock_gemini.return_value = (
            "제목: 퀀트 인사이트\n\n데이터 없음",
            "gemini-2.0-flash",
        )
        post = generate_quant_insight(
            outlook_map=None,
            trade_date="2026-04-15",
            config=_config(),
        )
        assert isinstance(post, ContentPost)


# ═══════════════════════════════════════════════════════════════════════
# 프리마켓 브리핑 프롬프트 빌더 테스트
# ═══════════════════════════════════════════════════════════════════════

class TestBuildPreMarketPrompt:
    def test_contains_macro_summary(self):
        prompt = build_pre_market_prompt(
            macro_summary=_macro_summary(),
            trade_date="2026-04-16",
        )
        assert "S&P 500" in prompt
        assert "원/달러" in prompt
        assert "프리마켓 브리핑" in prompt

    def test_contains_yield_spread(self):
        prompt = build_pre_market_prompt(
            yield_spread=0.45,
            trade_date="2026-04-16",
        )
        assert "장단기 금리차" in prompt
        assert "0.45" in prompt

    def test_contains_market_regime(self):
        prompt = build_pre_market_prompt(
            market_regime="보통",
            trade_date="2026-04-16",
        )
        assert "시장 체제" in prompt
        assert "보통" in prompt

    def test_contains_macro_narrative(self):
        prompt = build_pre_market_prompt(
            macro_narrative="위험자산 선호 흐름이 지속",
            trade_date="2026-04-16",
        )
        assert "위험자산 선호 흐름" in prompt
        assert "거시경제 내러티브" in prompt

    def test_contains_writing_instructions(self):
        prompt = build_pre_market_prompt(trade_date="2026-04-16")
        assert "2026-04-16" in prompt
        assert "한국 시장" in prompt
        assert "면책 문구" in prompt

    def test_minimal_prompt(self):
        prompt = build_pre_market_prompt(trade_date="2026-04-16")
        assert "프리마켓 브리핑" in prompt
        assert "작성 지침" in prompt


# ═══════════════════════════════════════════════════════════════════════
# 프리마켓 브리핑 생성 함수 테스트
# ═══════════════════════════════════════════════════════════════════════

class TestGeneratePreMarket:
    @patch("src.content_generator._generate_with_gemini")
    def test_returns_content_post(self, mock_gemini):
        mock_gemini.return_value = (
            "제목: 2026년 04월 16일 프리마켓 브리핑 — 나스닥 강세\n\n## 미국 증시 마감\n본문",
            "gemini-2.0-flash",
        )
        post = generate_pre_market(
            macro_summary=_macro_summary(),
            macro_narrative="위험자산 선호",
            market_regime="보통",
            yield_spread=0.45,
            trade_date="2026-04-16",
            config=_config(),
        )
        assert isinstance(post, ContentPost)
        assert post.content_type == "pre_market"
        assert post.model == "gemini-2.0-flash"
        assert "프리마켓브리핑" in post.categories
        assert "시장분석" in post.categories
        assert "프리마켓" in post.tags
        assert "미국증시" in post.tags
        assert "KOSPI전망" in post.tags

    @patch("src.content_generator._generate_with_gemini")
    def test_forbidden_word_detected(self, mock_gemini):
        mock_gemini.return_value = (
            "제목: 프리마켓\n\n지금 사야 합니다",
            "gemini-2.0-flash",
        )
        post = generate_pre_market(
            trade_date="2026-04-16",
            config=_config(),
        )
        assert len(post.warnings) > 0

    @patch("src.content_generator._generate_with_gemini")
    def test_filename_pre_market(self, mock_gemini):
        mock_gemini.return_value = (
            "제목: 프리마켓 브리핑\n\n본문",
            "gemini-2.0-flash",
        )
        post = generate_pre_market(
            trade_date="2026-04-16",
            config=_config(),
        )
        assert post.filename("2026-04-16") == "2026-04-16_프리마켓브리핑.md"
