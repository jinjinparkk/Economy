"""fetch_earnings 모듈 단위 테스트."""
from __future__ import annotations

import pathlib
from unittest.mock import patch, MagicMock

import pytest

from src.config import Config
from src.fetch_earnings import (
    EarningsEvent,
    EarningsSnapshot,
    _determine_report_type,
    _determine_surprise,
    _reprt_code_from_type,
    _TRACKED_CORPS,
    fetch_earnings_snapshot,
)


def _config(dart_key="fake-dart-key"):
    return Config(
        llm_provider="gemini",
        anthropic_api_key="",
        gemini_api_key="fake-gemini-key",
        naver_client_id="",
        naver_client_secret="",
        dart_api_key=dart_key,
        output_dir=pathlib.Path("."),
        timezone="Asia/Seoul",
        mover_threshold_pct=5.0,
        volume_surge_multiplier=3.0,
        top_n_movers=5,
        wp_access_token="",
        wp_site_id="",
        wp_auto_publish=False,
        telegram_bot_token="",
        telegram_channel_id="",
        telegram_auto_post=False,
        naver_access_token="",
        naver_auto_publish=False,
    )


# ── EarningsEvent ────────────────────────────────────────────────────

class TestEarningsEvent:
    def test_creation(self):
        event = EarningsEvent(
            corp_code="00126380",
            corp_name="삼성전자",
            report_date="2026-04-20",
            report_type="1Q",
            revenue=77_000_000_000_000,
            operating_profit=6_500_000_000_000,
            net_income=5_000_000_000_000,
            revenue_yoy=3.5,
            op_yoy=15.2,
            d_day=0,
            surprise="부합",
        )
        assert event.corp_name == "삼성전자"
        assert event.report_type == "1Q"
        assert event.surprise == "부합"

    def test_defaults(self):
        event = EarningsEvent("00126380", "삼성전자", "2026-04-20", "1Q")
        assert event.revenue is None
        assert event.d_day == 0
        assert event.surprise is None


# ── EarningsSnapshot ─────────────────────────────────────────────────

class TestEarningsSnapshot:
    def test_to_dict(self):
        snap = EarningsSnapshot(
            upcoming=[
                EarningsEvent("00126380", "삼성전자", "2026-04-25", "1Q", d_day=5),
            ],
            recent=[
                EarningsEvent("00164779", "SK하이닉스", "2026-04-18", "1Q",
                              revenue=15_000_000_000_000, operating_profit=5_000_000_000_000,
                              op_yoy=30.0, d_day=-2, surprise="어닝서프라이즈"),
            ],
        )
        d = snap.to_dict()
        assert len(d["upcoming"]) == 1
        assert d["upcoming"][0]["corp_name"] == "삼성전자"
        assert d["upcoming"][0]["d_day"] == 5
        assert len(d["recent"]) == 1
        assert d["recent"][0]["surprise"] == "어닝서프라이즈"

    def test_to_narrative_with_data(self):
        snap = EarningsSnapshot(
            upcoming=[
                EarningsEvent("00126380", "삼성전자", "2026-04-25", "1Q", d_day=5),
            ],
            recent=[
                EarningsEvent("00164779", "SK하이닉스", "2026-04-18", "1Q",
                              op_yoy=30.0, d_day=-2, surprise="어닝서프라이즈"),
            ],
        )
        narrative = snap.to_narrative()
        assert "SK하이닉스" in narrative
        assert "삼성전자" in narrative
        assert "어닝서프라이즈" in narrative

    def test_to_narrative_empty(self):
        snap = EarningsSnapshot()
        narrative = snap.to_narrative()
        assert "없음" in narrative

    def test_to_narrative_upcoming_only(self):
        snap = EarningsSnapshot(
            upcoming=[
                EarningsEvent("00126380", "삼성전자", "2026-04-25", "1Q", d_day=5),
            ],
        )
        narrative = snap.to_narrative()
        assert "삼성전자" in narrative
        assert "D-5" in narrative


# ── 헬퍼 함수 ─────────────────────────────────────────────────────────

class TestDetermineReportType:
    def test_q1(self):
        assert _determine_report_type("1분기보고서") == "1Q"

    def test_q2_halfyear(self):
        assert _determine_report_type("반기보고서") == "2Q"

    def test_q3(self):
        assert _determine_report_type("3분기보고서") == "3Q"

    def test_q4_annual(self):
        assert _determine_report_type("사업보고서") == "4Q"

    def test_unknown(self):
        assert _determine_report_type("기타보고서") == "4Q"


class TestDetermineSurprise:
    def test_surprise(self):
        assert _determine_surprise(25.0) == "어닝서프라이즈"

    def test_shock(self):
        assert _determine_surprise(-30.0) == "어닝쇼크"

    def test_inline(self):
        assert _determine_surprise(5.0) == "부합"

    def test_none(self):
        assert _determine_surprise(None) is None

    def test_boundary_positive(self):
        assert _determine_surprise(20.0) == "어닝서프라이즈"

    def test_boundary_negative(self):
        assert _determine_surprise(-20.0) == "어닝쇼크"


class TestReprtCode:
    def test_q1(self):
        assert _reprt_code_from_type("1Q") == "11013"

    def test_q2(self):
        assert _reprt_code_from_type("2Q") == "11012"

    def test_q3(self):
        assert _reprt_code_from_type("3Q") == "11014"

    def test_q4(self):
        assert _reprt_code_from_type("4Q") == "11011"


class TestTrackedCorps:
    def test_has_10_corps(self):
        assert len(_TRACKED_CORPS) == 10

    def test_samsung_included(self):
        assert "삼성전자" in _TRACKED_CORPS


# ── fetch_earnings_snapshot (mocked) ─────────────────────────────────

class TestFetchEarningsSnapshot:
    def test_no_api_key_returns_none(self):
        cfg = _config(dart_key="")
        result = fetch_earnings_snapshot(cfg, "2026-04-20")
        assert result is None

    @patch("src.fetch_earnings._fetch_financials")
    @patch("src.fetch_earnings._fetch_disclosure_list")
    def test_basic_snapshot(self, mock_list, mock_fin):
        """기본 공시 목록 → 스냅샷 생성."""
        # 하나의 종목에 대해 최근 발표 공시 1건 반환
        def side_effect(api_key, corp_code, bgn, end):
            if corp_code == "00126380":  # 삼성전자
                return [{
                    "rcept_dt": "20260418",
                    "report_nm": "1분기보고서",
                }]
            return []
        mock_list.side_effect = side_effect
        mock_fin.return_value = {
            "revenue": 77_000_000_000_000,
            "operating_profit": 6_500_000_000_000,
        }

        cfg = _config()
        result = fetch_earnings_snapshot(cfg, "2026-04-20")
        assert result is not None
        assert isinstance(result, EarningsSnapshot)
        # 삼성전자는 2026-04-18 발표 → d_day = -2 → recent에 추가
        assert len(result.recent) >= 1
        samsung = [e for e in result.recent if e.corp_name == "삼성전자"]
        assert len(samsung) == 1
        assert samsung[0].report_type == "1Q"
        assert samsung[0].revenue == 77_000_000_000_000

    @patch("src.fetch_earnings._fetch_disclosure_list")
    def test_upcoming_event(self, mock_list):
        """향후 예정 공시."""
        def side_effect(api_key, corp_code, bgn, end):
            if corp_code == "00126380":
                return [{
                    "rcept_dt": "20260425",
                    "report_nm": "1분기보고서",
                }]
            return []
        mock_list.side_effect = side_effect

        cfg = _config()
        result = fetch_earnings_snapshot(cfg, "2026-04-20")
        assert result is not None
        assert len(result.upcoming) >= 1
        samsung = [e for e in result.upcoming if e.corp_name == "삼성전자"]
        assert len(samsung) == 1
        assert samsung[0].d_day == 5

    @patch("src.fetch_earnings._fetch_disclosure_list")
    def test_exception_graceful(self, mock_list):
        mock_list.side_effect = RuntimeError("network error")
        cfg = _config()
        result = fetch_earnings_snapshot(cfg, "2026-04-20")
        assert result is None

    @patch("src.fetch_earnings._fetch_disclosure_list")
    def test_empty_disclosures(self, mock_list):
        mock_list.return_value = []
        cfg = _config()
        result = fetch_earnings_snapshot(cfg, "2026-04-20")
        assert result is not None
        assert len(result.recent) == 0
        assert len(result.upcoming) == 0
