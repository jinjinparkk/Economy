"""sentiment 모듈 단위 테스트."""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.sentiment import (
    SentimentIndex,
    _vix_score,
    _breadth_score,
    _yield_spread_score,
    _credit_score,
    _momentum_score,
    _rsi_distribution_score,
    _score_to_label,
    _WEIGHTS,
    compute_sentiment,
    _load_prev_score,
    _save_history,
)
from src.fetch_macro import MacroSnapshot, MacroIndicator


# ── 헬퍼 ──────────────────────────────────────────────────────────────

def _macro(vix=18.0, yield_spread_val=0.5, hyg_pct=0.1):
    """테스트용 MacroSnapshot 생성."""
    snap = MacroSnapshot()
    snap.volatility["VIX"] = MacroIndicator("VIX", "VIX", vix, vix + 1, -1, -5.0, "2026-04-20")
    snap.bonds["US10Y"] = MacroIndicator("미국채10Y", "US10YT", 4.0 + yield_spread_val, 4.0, yield_spread_val, 0, "2026-04-20")
    snap.bonds["US2Y"] = MacroIndicator("미국채2Y", "US2YT", 4.0, 4.0, 0, 0, "2026-04-20")
    snap.credit["HYG"] = MacroIndicator("하이일드채권", "HYG", 80.0, 79.9, 0.1, hyg_pct, "2026-04-20")
    return snap


def _market_snapshot(up_ratio=55.0, change_pct=-0.5):
    """테스트용 MarketSnapshot MagicMock 생성."""
    snap = MagicMock()
    snap.market_breadth.return_value = {
        "total_up": 550, "total_down": 450, "total_unchanged": 50,
        "total": 1050, "up_ratio": up_ratio,
    }
    snap.indices = {
        "KOSPI": pd.Series({"Close": 2700.0, "ChangePct": change_pct}),
    }
    df = pd.DataFrame({
        "ChangeRatio": [3.0, -2.0, 6.0, -6.0, 1.0, -1.0, 0.5, -0.3, 7.0, -7.0],
    })
    snap.all_stocks.return_value = df
    return snap


# ── 컴포넌트 경계값 테스트 ────────────────────────────────────────────

class TestVixScore:
    def test_low_vix(self):
        assert _vix_score(10.0) == 100.0

    def test_high_vix(self):
        assert _vix_score(35.0) == 0.0

    def test_boundary_12(self):
        assert _vix_score(12.0) == 100.0

    def test_boundary_30(self):
        assert _vix_score(30.0) == 0.0

    def test_midpoint(self):
        score = _vix_score(21.0)
        assert 40 < score < 60  # 약 50

    def test_linear_interpolation(self):
        # 21 → (30-21)/(30-12)*100 = 9/18*100 = 50
        assert _vix_score(21.0) == 50.0


class TestBreadthScore:
    def test_low_ratio(self):
        assert _breadth_score(20.0) == 0.0

    def test_high_ratio(self):
        assert _breadth_score(80.0) == 100.0

    def test_boundary_30(self):
        assert _breadth_score(30.0) == 0.0

    def test_boundary_70(self):
        assert _breadth_score(70.0) == 100.0

    def test_midpoint(self):
        assert _breadth_score(50.0) == 50.0


class TestYieldSpreadScore:
    def test_inverted(self):
        assert _yield_spread_score(-1.0) == 0.0

    def test_normal(self):
        assert _yield_spread_score(2.0) == 100.0

    def test_boundary_low(self):
        assert _yield_spread_score(-0.5) == 0.0

    def test_boundary_high(self):
        assert _yield_spread_score(1.5) == 100.0

    def test_midpoint(self):
        assert _yield_spread_score(0.5) == 50.0


class TestCreditScore:
    def test_crash(self):
        assert _credit_score(-3.0) == 0.0

    def test_rally(self):
        assert _credit_score(2.0) == 100.0

    def test_boundary_low(self):
        assert _credit_score(-2.0) == 0.0

    def test_boundary_high(self):
        assert _credit_score(1.0) == 100.0

    def test_neutral(self):
        # -0.5 → (-0.5+2)/3*100 = 1.5/3*100 = 50
        assert _credit_score(-0.5) == 50.0


class TestMomentumScore:
    def test_below_ma(self):
        assert _momentum_score(95.0, 100.0) == 0.0  # -5% → boundary → 0.0

    def test_above_ma(self):
        assert _momentum_score(105.0, 100.0) == 100.0

    def test_far_below(self):
        assert _momentum_score(90.0, 100.0) == 0.0

    def test_at_ma(self):
        assert _momentum_score(100.0, 100.0) == 50.0

    def test_zero_ma(self):
        assert _momentum_score(100.0, 0.0) == 50.0


class TestRsiDistributionScore:
    def test_all_oversold(self):
        assert _rsi_distribution_score(100.0, 0.0) == 0.0

    def test_all_overbought(self):
        assert _rsi_distribution_score(0.0, 100.0) == 100.0

    def test_balanced(self):
        assert _rsi_distribution_score(10.0, 10.0) == 50.0

    def test_mostly_oversold(self):
        score = _rsi_distribution_score(30.0, 5.0)
        assert score < 50.0


# ── 라벨 매핑 ─────────────────────────────────────────────────────────

class TestScoreToLabel:
    def test_extreme_fear(self):
        assert _score_to_label(10) == "극단적 공포"
        assert _score_to_label(20) == "극단적 공포"

    def test_fear(self):
        assert _score_to_label(30) == "공포"
        assert _score_to_label(40) == "공포"

    def test_neutral(self):
        assert _score_to_label(50) == "중립"
        assert _score_to_label(60) == "중립"

    def test_greed(self):
        assert _score_to_label(70) == "탐욕"
        assert _score_to_label(80) == "탐욕"

    def test_extreme_greed(self):
        assert _score_to_label(90) == "극단적 탐욕"
        assert _score_to_label(100) == "극단적 탐욕"


# ── 가중합 정확성 ─────────────────────────────────────────────────────

class TestWeightSum:
    def test_weights_sum_to_one(self):
        assert abs(sum(_WEIGHTS.values()) - 1.0) < 1e-9

    def test_all_keys_present(self):
        expected_keys = {"VIX", "시장_폭", "금리_스프레드", "신용_스프레드", "모멘텀", "RSI_분포"}
        assert set(_WEIGHTS.keys()) == expected_keys


# ── compute_sentiment 통합 ───────────────────────────────────────────

class TestComputeSentiment:
    def test_returns_sentiment_index(self):
        macro = _macro()
        snapshot = _market_snapshot()
        result = compute_sentiment(macro, snapshot)
        assert isinstance(result, SentimentIndex)
        assert 0 <= result.score <= 100
        assert result.label in {"극단적 공포", "공포", "중립", "탐욕", "극단적 탐욕"}

    def test_components_all_present(self):
        macro = _macro()
        snapshot = _market_snapshot()
        result = compute_sentiment(macro, snapshot)
        for key in _WEIGHTS:
            assert key in result.components
            assert 0 <= result.components[key] <= 100

    def test_high_fear_scenario(self):
        """VIX 높고, 하락장, 금리역전 → 낮은 점수."""
        macro = _macro(vix=35.0, yield_spread_val=-1.0, hyg_pct=-3.0)
        snapshot = _market_snapshot(up_ratio=20.0, change_pct=-3.0)
        result = compute_sentiment(macro, snapshot)
        assert result.score < 30

    def test_high_greed_scenario(self):
        """VIX 낮고, 상승장, 정상 금리 → 높은 점수."""
        macro = _macro(vix=10.0, yield_spread_val=1.5, hyg_pct=1.5)
        snapshot = _market_snapshot(up_ratio=75.0, change_pct=3.0)
        result = compute_sentiment(macro, snapshot)
        assert result.score > 70

    def test_to_dict(self):
        macro = _macro()
        snapshot = _market_snapshot()
        result = compute_sentiment(macro, snapshot)
        d = result.to_dict()
        assert "score" in d
        assert "label" in d
        assert "components" in d

    def test_missing_vix_uses_default(self):
        macro = MacroSnapshot()
        snapshot = _market_snapshot()
        result = compute_sentiment(macro, snapshot)
        assert result.components["VIX"] == 50.0

    def test_missing_hyg_uses_default(self):
        macro = _macro()
        macro.credit = {}
        snapshot = _market_snapshot()
        result = compute_sentiment(macro, snapshot)
        assert result.components["신용_스프레드"] == 50.0

    def test_missing_indices_uses_default(self):
        macro = _macro()
        snapshot = _market_snapshot()
        snapshot.indices = {}
        result = compute_sentiment(macro, snapshot)
        assert result.components["모멘텀"] == 50.0


# ── 히스토리 저장/로드 ────────────────────────────────────────────────

class TestHistory:
    def test_save_and_load(self, tmp_path):
        _save_history(tmp_path, 55.0, "중립")
        prev = _load_prev_score(tmp_path)
        assert prev == 55.0

    def test_load_empty_dir(self, tmp_path):
        prev = _load_prev_score(tmp_path)
        assert prev is None

    def test_history_accumulates(self, tmp_path):
        _save_history(tmp_path, 40.0, "공포")
        _save_history(tmp_path, 60.0, "중립")
        prev = _load_prev_score(tmp_path)
        assert prev == 60.0

        data = json.loads((tmp_path / "sentiment_history.json").read_text())
        assert len(data) == 2

    def test_history_max_90(self, tmp_path):
        for i in range(100):
            _save_history(tmp_path, float(i), "중립")
        data = json.loads((tmp_path / "sentiment_history.json").read_text())
        assert len(data) == 90

    def test_prev_score_in_compute(self, tmp_path):
        macro = _macro()
        snapshot = _market_snapshot()
        # 첫 번째 — prev 없음
        r1 = compute_sentiment(macro, snapshot, output_dir=tmp_path)
        assert r1.prev_score is None
        assert r1.change is None
        # 두 번째 — prev 있음
        r2 = compute_sentiment(macro, snapshot, output_dir=tmp_path)
        assert r2.prev_score == r1.score
        assert r2.change is not None

    def test_corrupted_history_graceful(self, tmp_path):
        path = tmp_path / "sentiment_history.json"
        path.write_text("not json", encoding="utf-8")
        prev = _load_prev_score(tmp_path)
        assert prev is None
