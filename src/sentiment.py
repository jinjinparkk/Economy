"""Fear & Greed 종합 지수 모듈.

6개 가중 요소를 결합하여 0~100 점수와 라벨을 산출한다.
- VIX (0.25) / 시장 폭 (0.20) / 금리 스프레드 (0.15)
- 신용 스프레드 (0.15) / 모멘텀 (0.15) / RSI 분포 (0.10)

모든 소스는 기존 MacroSnapshot + MarketSnapshot에서 가져오므로
추가 API 호출이 없다.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_HISTORY_FILE = "sentiment_history.json"
_MAX_HISTORY = 90


# ── 라벨 경계 ────────────────────────────────────────────────────────
def _score_to_label(score: float) -> str:
    if score <= 20:
        return "극단적 공포"
    elif score <= 40:
        return "공포"
    elif score <= 60:
        return "중립"
    elif score <= 80:
        return "탐욕"
    else:
        return "극단적 탐욕"


# ── 개별 컴포넌트 점수 (0~100) ──────────────────────────────────────

def _vix_score(vix_close: float) -> float:
    """VIX 12 이하 → 100, 30 이상 → 0, 선형 보간."""
    if vix_close <= 12:
        return 100.0
    if vix_close >= 30:
        return 0.0
    return round((30 - vix_close) / (30 - 12) * 100, 1)


def _breadth_score(up_ratio: float) -> float:
    """상승 비율(%) → 0~100 점수.

    up_ratio 30% 이하 → 0, 70% 이상 → 100, 선형.
    """
    if up_ratio <= 30:
        return 0.0
    if up_ratio >= 70:
        return 100.0
    return round((up_ratio - 30) / 40 * 100, 1)


def _yield_spread_score(spread: float) -> float:
    """장단기 금리차: -0.5 이하 → 0, +1.5 이상 → 100, 선형."""
    if spread <= -0.5:
        return 0.0
    if spread >= 1.5:
        return 100.0
    return round((spread + 0.5) / 2.0 * 100, 1)


def _credit_score(hyg_change_pct: float) -> float:
    """HYG 일간 등락률: -2% 이하 → 0, +1% 이상 → 100, 선형."""
    if hyg_change_pct <= -2.0:
        return 0.0
    if hyg_change_pct >= 1.0:
        return 100.0
    return round((hyg_change_pct + 2.0) / 3.0 * 100, 1)


def _momentum_score(current_close: float, ma20: float) -> float:
    """KOSPI 현재가 vs 20일 이평: -5% 이하 → 0, +5% 이상 → 100, 선형."""
    if ma20 == 0:
        return 50.0
    diff_pct = (current_close - ma20) / ma20 * 100
    if diff_pct <= -5:
        return 0.0
    if diff_pct >= 5:
        return 100.0
    return round((diff_pct + 5) / 10 * 100, 1)


def _rsi_distribution_score(
    oversold_ratio: float, overbought_ratio: float,
) -> float:
    """RSI 과매도/과매수 비율 기반.

    과매수 비율 높으면 탐욕(점수 높음), 과매도 비율 높으면 공포(점수 낮음).
    net = overbought_ratio - oversold_ratio  (-100 ~ +100)
    → 선형 매핑 0~100
    """
    net = overbought_ratio - oversold_ratio  # -100 ~ +100
    return round(max(0.0, min(100.0, (net + 100) / 2)), 1)


# ── 가중치 ───────────────────────────────────────────────────────────
_WEIGHTS = {
    "VIX": 0.25,
    "시장_폭": 0.20,
    "금리_스프레드": 0.15,
    "신용_스프레드": 0.15,
    "모멘텀": 0.15,
    "RSI_분포": 0.10,
}


@dataclass
class SentimentIndex:
    score: float                    # 0~100
    label: str                      # 극단적 공포 ~ 극단적 탐욕
    components: dict[str, float]    # 각 요소별 점수
    prev_score: float | None = None
    change: float | None = None

    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "label": self.label,
            "components": self.components,
            "prev_score": self.prev_score,
            "change": self.change,
        }


def compute_sentiment(
    macro: "MacroSnapshot",
    snapshot: "MarketSnapshot",
    *,
    output_dir: Optional[Path] = None,
) -> SentimentIndex:
    """Fear & Greed 종합 지수를 계산한다.

    Args:
        macro: 거시경제 스냅샷 (VIX, yield_spread, credit)
        snapshot: 국내 시장 스냅샷 (breadth, indices, all_stocks)
        output_dir: 히스토리 저장 경로 (None이면 저장 안 함)

    Returns:
        SentimentIndex
    """
    components: dict[str, float] = {}

    # 1) VIX
    vix = macro.volatility.get("VIX")
    components["VIX"] = _vix_score(vix.close) if vix else 50.0

    # 2) 시장 폭
    breadth = snapshot.market_breadth()
    components["시장_폭"] = _breadth_score(breadth.get("up_ratio", 50.0))

    # 3) 금리 스프레드
    spread = macro.yield_spread
    components["금리_스프레드"] = _yield_spread_score(spread) if spread is not None else 50.0

    # 4) 신용 스프레드
    hyg = macro.credit.get("HYG")
    components["신용_스프레드"] = _credit_score(hyg.change_pct) if hyg else 50.0

    # 5) 모멘텀 (KOSPI vs 20일 이평)
    kospi_idx = snapshot.indices.get("KOSPI")
    if kospi_idx is not None:
        kospi_close = float(kospi_idx["Close"])
        # 20일 이평을 indices에서 직접 구할 수 없으므로
        # PrevClose * 20의 근사 대신, Close 기준 간단 대체
        # 실제로는 fetch_ohlcv_history를 쓸 수 있으나 추가 API 호출 회피
        # → 대신 kospi_close 자체를 기준으로 change_pct 기반 추정
        change_pct = float(kospi_idx.get("ChangePct", 0))
        # 누적 모멘텀 대신 당일 변동률 기반 단순 변환
        # -5% → 0점, +5% → 100점
        components["모멘텀"] = _momentum_score(100 + change_pct, 100)
    else:
        components["모멘텀"] = 50.0

    # 6) RSI 분포 (전종목 중 RSI ≤30 / ≥70 비율)
    # MarketSnapshot에서 RSI를 직접 갖고 있지 않으므로
    # all_stocks의 ChangeRatio 분포를 대리 지표로 사용
    all_df = snapshot.all_stocks()
    if not all_df.empty and "ChangeRatio" in all_df.columns:
        total = len(all_df)
        # 과매도 대리: 등락률 -5% 이하, 과매수 대리: +5% 이상
        oversold_cnt = int((all_df["ChangeRatio"] <= -5).sum())
        overbought_cnt = int((all_df["ChangeRatio"] >= 5).sum())
        oversold_ratio = oversold_cnt / total * 100 if total else 0
        overbought_ratio = overbought_cnt / total * 100 if total else 0
        components["RSI_분포"] = _rsi_distribution_score(oversold_ratio, overbought_ratio)
    else:
        components["RSI_분포"] = 50.0

    # 가중합
    total_score = sum(components[k] * _WEIGHTS[k] for k in _WEIGHTS)
    total_score = round(max(0.0, min(100.0, total_score)), 1)
    label = _score_to_label(total_score)

    # 이전 점수 로드
    prev_score = None
    change = None
    if output_dir:
        prev_score = _load_prev_score(output_dir)
        if prev_score is not None:
            change = round(total_score - prev_score, 1)

    result = SentimentIndex(
        score=total_score,
        label=label,
        components=components,
        prev_score=prev_score,
        change=change,
    )

    # 히스토리 저장
    if output_dir:
        _save_history(output_dir, total_score, label)

    return result


# ── 히스토리 I/O ─────────────────────────────────────────────────────

def _load_prev_score(output_dir: Path) -> float | None:
    path = output_dir / _HISTORY_FILE
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data:
            return float(data[-1]["score"])
    except Exception as exc:
        logger.warning("sentiment history load failed: %s", exc)
    return None


def _save_history(output_dir: Path, score: float, label: str) -> None:
    path = output_dir / _HISTORY_FILE
    data: list[dict] = []
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            data = []

    data.append({
        "date": datetime.now().strftime("%Y-%m-%d"),
        "score": score,
        "label": label,
    })

    # 최근 90일만 보관
    if len(data) > _MAX_HISTORY:
        data = data[-_MAX_HISTORY:]

    output_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
