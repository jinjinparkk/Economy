"""main 파이프라인 오케스트레이터 단위 테스트."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

from src.main import run_pipeline, run_pre_market_pipeline, _save_article, _save_content_post, _save_dry_run_summary
from src.analyzer import Article
from src.content_post import ContentPost
from src.detect_movers import Mover, MoverReport
from src.fetch_macro import MacroSnapshot, MacroIndicator


# ── 헬퍼 ──────────────────────────────────────────────────────────────
def _mover(name="테스트종목", code="000001", move_type="surge",
           change_pct=15.0, close=5000, amount=2e9):
    return Mover(code, name, "KOSPI", move_type, close, change_pct,
                 100000, int(amount), int(1e11), industry="테스트업종")


def _article(name="테스트종목", title="테스트 제목"):
    m = _mover(name=name)
    return Article(title, "본문 내용", m, [], "gemini-2.0-flash", [])


def _macro():
    snap = MacroSnapshot()
    snap.us_indices["SP500"] = MacroIndicator("S&P 500", "US500", 5400, 5300, 100, 1.89, "2026-04-08")
    snap.fx["USDKRW"] = MacroIndicator("원/달러", "USD/KRW", 1380, 1370, 10, 0.73, "2026-04-08")
    return snap


# ── _save_article ────────────────────────────────────────────────────
class TestSaveArticle:
    def test_saves_markdown(self, tmp_path):
        art = _article(name="삼성전자", title="급등 분석")
        path = _save_article(art, tmp_path, "2026-04-09")
        assert path.exists()
        assert path.name == "2026-04-09_삼성전자_급등.md"
        content = path.read_text(encoding="utf-8")
        assert "# 급등 분석" in content

    def test_creates_subdirectory(self, tmp_path):
        sub = tmp_path / "deep" / "nested"
        art = _article()
        path = _save_article(art, sub, "2026-04-09")
        assert path.exists()
        assert sub.exists()


# ── _save_dry_run_summary ────────────────────────────────────────────
class TestSaveDryRunSummary:
    def test_saves_json(self, tmp_path):
        rpt = MoverReport("2026-04-09", [_mover()], [_mover(move_type="plunge")])
        news_map = {
            "000001": [{"title": "뉴스", "description": "설명", "link": "http://x",
                        "press": "한경", "pub_date": None}],
        }
        macro = _macro()
        path = _save_dry_run_summary(rpt, news_map, macro, tmp_path)
        assert path.exists()
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["trade_date"] == "2026-04-09"
        assert len(data["surges"]) == 1
        assert len(data["plunges"]) == 1
        # 거시경제 데이터 포함
        assert "macro" in data
        assert "S&P 500" in data["macro"]

    def test_includes_industry(self, tmp_path):
        rpt = MoverReport("2026-04-09", [_mover()], [])
        path = _save_dry_run_summary(rpt, {}, MacroSnapshot(), tmp_path)
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["surges"][0]["industry"] == "테스트업종"


# ── run_pipeline dry-run (전체 모킹) ────────────────────────────────
class TestRunPipeline:
    @patch("src.main.fetch_macro_snapshot")
    @patch("src.main.fetch_market_snapshot")
    @patch("src.main.detect_movers")
    @patch("src.main.fetch_news_for_stock")
    def test_dry_run_no_llm_call(self, mock_news, mock_detect, mock_fetch, mock_macro, tmp_path):
        from src.fetch_market import MarketSnapshot

        mock_macro.return_value = _macro()
        snap = MarketSnapshot(
            trade_date=date(2026, 4, 9),
            kospi=pd.DataFrame(),
            kosdaq=pd.DataFrame(),
            indices={"KOSPI": pd.Series({"Close": 2700, "ChangePct": -1.0})},
        )
        mock_fetch.return_value = snap
        mock_detect.return_value = MoverReport(
            "2026-04-09", [_mover()], [_mover(move_type="plunge")]
        )
        from src.fetch_news import NewsItem
        mock_news.return_value = [NewsItem("뉴스", "설명", "http://x", "한경")]

        with patch("src.main.Config") as MockConfig:
            cfg = MagicMock()
            cfg.output_dir = tmp_path
            cfg.llm_provider = "gemini"
            cfg.gemini_api_key = ""
            MockConfig.load.return_value = cfg

            run_pipeline(top_n=1, dry_run=True)

        # dry-run JSON이 생성돼야 함
        files = list(tmp_path.glob("*_dry_run.json"))
        assert len(files) == 1
        data = json.loads(files[0].read_text(encoding="utf-8"))
        assert "macro" in data

    @patch("src.main.fetch_macro_snapshot")
    @patch("src.main.fetch_market_snapshot")
    @patch("src.main.detect_movers")
    def test_no_movers_exits_early(self, mock_detect, mock_fetch, mock_macro, tmp_path):
        from src.fetch_market import MarketSnapshot

        mock_macro.return_value = MacroSnapshot()
        snap = MarketSnapshot(
            trade_date=date(2026, 4, 9),
            kospi=pd.DataFrame(),
            kosdaq=pd.DataFrame(),
            indices={},
        )
        mock_fetch.return_value = snap
        mock_detect.return_value = MoverReport("2026-04-09", [], [])

        with patch("src.main.Config") as MockConfig:
            cfg = MagicMock()
            cfg.output_dir = tmp_path
            MockConfig.load.return_value = cfg

            # 에러 없이 조기 종료
            run_pipeline(top_n=1, dry_run=False)

        assert list(tmp_path.glob("*.md")) == []


# ── _save_content_post ──────────────────────────────────────────────
class TestSaveContentPost:
    def test_saves_markdown(self, tmp_path):
        post = ContentPost("시황 제목", "본문", "daily_market", "gemini-2.0-flash")
        path = _save_content_post(post, tmp_path, "2026-04-15")
        assert path.exists()
        assert path.name == "2026-04-15_데일리시황.md"
        content = path.read_text(encoding="utf-8")
        assert "# 시황 제목" in content

    def test_creates_subdirectory(self, tmp_path):
        sub = tmp_path / "deep" / "nested"
        post = ContentPost("섹터", "본문", "sector_report", "m")
        path = _save_content_post(post, sub, "2026-04-15")
        assert path.exists()
        assert sub.exists()


# ── run_pipeline with content posts ────────────────────────────────
class TestRunPipelineWithContentPosts:
    @patch("src.main.generate_quant_insight")
    @patch("src.main.generate_sector_report")
    @patch("src.main.generate_daily_market")
    @patch("src.main.generate_article")
    @patch("src.main.compute_outlook")
    @patch("src.main.fetch_news_for_stock")
    @patch("src.main.detect_movers")
    @patch("src.main.fetch_market_snapshot")
    @patch("src.main.fetch_macro_snapshot")
    def test_content_posts_saved(
        self, mock_macro, mock_fetch, mock_detect, mock_news,
        mock_outlook, mock_article, mock_daily, mock_sector, mock_quant,
        tmp_path,
    ):
        from src.fetch_market import MarketSnapshot
        from src.fetch_news import NewsItem
        from src.predictor import OutlookData

        mock_macro.return_value = _macro()
        snap = MarketSnapshot(
            trade_date=date(2026, 4, 15),
            kospi=pd.DataFrame(),
            kosdaq=pd.DataFrame(),
            indices={"KOSPI": pd.Series({"Close": 2700, "ChangePct": -1.0})},
        )
        mock_fetch.return_value = snap
        mock_detect.return_value = MoverReport(
            "2026-04-15", [_mover()], [],
        )
        mock_news.return_value = [NewsItem("뉴스", "설명", "http://x", "한경")]
        mock_outlook.return_value = OutlookData()
        mock_article.return_value = _article()

        mock_daily.return_value = ContentPost(
            "시황 제목", "시황 본문", "daily_market", "gemini-2.0-flash",
            tags=["데일리시황"], categories=["데일리시황"],
        )
        mock_sector.return_value = ContentPost(
            "섹터 제목", "섹터 본문", "sector_report", "gemini-2.0-flash",
            tags=["섹터분석"], categories=["섹터리포트"],
        )
        mock_quant.return_value = ContentPost(
            "퀀트 제목", "퀀트 본문", "quant_insight", "gemini-2.0-flash",
            tags=["퀀트"], categories=["퀀트인사이트"],
        )

        with patch("src.main.Config") as MockConfig:
            cfg = MagicMock()
            cfg.output_dir = tmp_path
            cfg.llm_provider = "gemini"
            cfg.gemini_api_key = "fake"
            cfg.wp_auto_publish = False
            cfg.wp_access_token = ""
            cfg.wp_site_id = ""
            MockConfig.load.return_value = cfg

            run_pipeline(top_n=1, dry_run=False)

        # 종목분석 1개 + 콘텐츠 포스트 3개 = 4개 md 파일
        md_files = sorted(p.name for p in tmp_path.glob("*.md"))
        assert len(md_files) == 4
        assert any("데일리시황" in f for f in md_files)
        assert any("섹터리포트" in f for f in md_files)
        assert any("퀀트인사이트" in f for f in md_files)

    @patch("src.main.generate_quant_insight")
    @patch("src.main.generate_sector_report")
    @patch("src.main.generate_daily_market")
    @patch("src.main.generate_article")
    @patch("src.main.compute_outlook")
    @patch("src.main.fetch_news_for_stock")
    @patch("src.main.detect_movers")
    @patch("src.main.fetch_market_snapshot")
    @patch("src.main.fetch_macro_snapshot")
    def test_content_generators_called(
        self, mock_macro, mock_fetch, mock_detect, mock_news,
        mock_outlook, mock_article, mock_daily, mock_sector, mock_quant,
        tmp_path,
    ):
        from src.fetch_market import MarketSnapshot
        from src.fetch_news import NewsItem
        from src.predictor import OutlookData

        mock_macro.return_value = _macro()
        snap = MarketSnapshot(
            trade_date=date(2026, 4, 15),
            kospi=pd.DataFrame(),
            kosdaq=pd.DataFrame(),
            indices={"KOSPI": pd.Series({"Close": 2700, "ChangePct": -1.0})},
        )
        mock_fetch.return_value = snap
        mock_detect.return_value = MoverReport("2026-04-15", [_mover()], [])
        mock_news.return_value = [NewsItem("뉴스", "설명", "http://x", "한경")]
        mock_outlook.return_value = OutlookData()
        mock_article.return_value = _article()

        mock_daily.return_value = ContentPost(
            "시황", "본문", "daily_market", "m",
        )
        mock_sector.return_value = ContentPost(
            "섹터", "본문", "sector_report", "m",
        )
        mock_quant.return_value = ContentPost(
            "퀀트", "본문", "quant_insight", "m",
        )

        with patch("src.main.Config") as MockConfig:
            cfg = MagicMock()
            cfg.output_dir = tmp_path
            cfg.llm_provider = "gemini"
            cfg.gemini_api_key = "fake"
            cfg.wp_auto_publish = False
            cfg.wp_access_token = ""
            cfg.wp_site_id = ""
            MockConfig.load.return_value = cfg

            run_pipeline(top_n=1, dry_run=False)

        # 모든 콘텐츠 생성 함수가 호출됐는지 확인
        mock_daily.assert_called_once()
        mock_sector.assert_called_once()
        mock_quant.assert_called_once()

    @patch("src.main.fetch_macro_snapshot")
    @patch("src.main.fetch_market_snapshot")
    @patch("src.main.detect_movers")
    @patch("src.main.fetch_news_for_stock")
    def test_dry_run_skips_content_generation(
        self, mock_news, mock_detect, mock_fetch, mock_macro, tmp_path
    ):
        """dry-run에서는 콘텐츠 생성이 스킵되어야 한다."""
        from src.fetch_market import MarketSnapshot

        mock_macro.return_value = _macro()
        snap = MarketSnapshot(
            trade_date=date(2026, 4, 15),
            kospi=pd.DataFrame(),
            kosdaq=pd.DataFrame(),
            indices={"KOSPI": pd.Series({"Close": 2700, "ChangePct": -1.0})},
        )
        mock_fetch.return_value = snap
        mock_detect.return_value = MoverReport("2026-04-15", [_mover()], [])
        mock_news.return_value = []

        with patch("src.main.Config") as MockConfig:
            cfg = MagicMock()
            cfg.output_dir = tmp_path
            cfg.llm_provider = "gemini"
            cfg.gemini_api_key = ""
            MockConfig.load.return_value = cfg

            run_pipeline(top_n=1, dry_run=True)

        # dry-run에서는 콘텐츠 md 파일 없어야 함
        md_files = list(tmp_path.glob("*시황*")) + list(tmp_path.glob("*섹터*")) + list(tmp_path.glob("*퀀트*"))
        assert len(md_files) == 0


# ── run_pre_market_pipeline ────────────────────────────────────────
class TestRunPreMarketPipeline:
    @patch("src.main.generate_pre_market")
    @patch("src.main.fetch_macro_snapshot")
    def test_pre_market_saves_briefing(self, mock_macro, mock_gen, tmp_path):
        mock_macro.return_value = _macro()
        mock_gen.return_value = ContentPost(
            "프리마켓 브리핑 제목", "본문 내용", "pre_market", "gemini-2.0-flash",
            tags=["프리마켓"], categories=["프리마켓브리핑"],
        )

        with patch("src.main.Config") as MockConfig:
            cfg = MagicMock()
            cfg.output_dir = tmp_path
            cfg.llm_provider = "gemini"
            cfg.gemini_api_key = "fake"
            cfg.wp_auto_publish = False
            cfg.wp_access_token = ""
            cfg.wp_site_id = ""
            MockConfig.load.return_value = cfg

            run_pre_market_pipeline()

        # 프리마켓 md 파일이 생성돼야 함
        md_files = list(tmp_path.glob("*프리마켓브리핑*"))
        assert len(md_files) == 1
        content = md_files[0].read_text(encoding="utf-8")
        assert "프리마켓 브리핑 제목" in content

    @patch("src.main.generate_pre_market")
    @patch("src.main.fetch_macro_snapshot")
    def test_pre_market_calls_generate(self, mock_macro, mock_gen, tmp_path):
        mock_macro.return_value = _macro()
        mock_gen.return_value = ContentPost(
            "제목", "본문", "pre_market", "m",
        )

        with patch("src.main.Config") as MockConfig:
            cfg = MagicMock()
            cfg.output_dir = tmp_path
            cfg.llm_provider = "gemini"
            cfg.gemini_api_key = "fake"
            cfg.wp_auto_publish = False
            cfg.wp_access_token = ""
            cfg.wp_site_id = ""
            MockConfig.load.return_value = cfg

            run_pre_market_pipeline()

        mock_macro.assert_called_once()
        mock_gen.assert_called_once()
        # generate_pre_market에 macro 데이터가 전달됐는지 확인
        call_kwargs = mock_gen.call_args.kwargs
        assert call_kwargs["macro_summary"] is not None
        assert call_kwargs["macro_narrative"] is not None

    @patch("src.main.fetch_macro_snapshot")
    def test_pre_market_no_api_key_exits(self, mock_macro, tmp_path):
        """API 키 없으면 LLM 호출 없이 종료."""
        mock_macro.return_value = _macro()

        with patch("src.main.Config") as MockConfig:
            cfg = MagicMock()
            cfg.output_dir = tmp_path
            cfg.llm_provider = "gemini"
            cfg.gemini_api_key = ""
            cfg.anthropic_api_key = ""
            cfg.wp_auto_publish = False
            MockConfig.load.return_value = cfg

            run_pre_market_pipeline()

        # 파일이 생성되지 않아야 함
        md_files = list(tmp_path.glob("*.md"))
        assert len(md_files) == 0
