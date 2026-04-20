"""거시경제·해외 증시 데이터 수집 모듈.

전일 미국 3대 지수, 환율, 유가, VIX, 미국채 10년물, 반도체 지수를 가져온다.
한국 장 시작 전(08:00 KST) 시점에 전일 미국 장 마감 데이터가 반영된다.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import pandas as pd

# FinanceDataReader / yfinance 독립 import — FDR 설치 환경에서도 yfinance fallback 가능
try:
    import FinanceDataReader as fdr
    _USE_FDR = True
except ImportError:
    fdr = None  # type: ignore[assignment]
    _USE_FDR = False

try:
    import yfinance as yf
except ImportError:
    yf = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


@dataclass
class MacroIndicator:
    """개별 거시경제 지표."""

    name: str           # 표시 이름 (예: "S&P 500")
    code: str           # FDR 코드
    close: float
    prev_close: float
    change: float
    change_pct: float
    date: str           # YYYY-MM-DD

    def __str__(self) -> str:
        return f"{self.name}: {self.close:,.2f} ({self.change_pct:+.2f}%)"


@dataclass
class MacroSnapshot:
    """거시경제 전체 스냅샷."""

    us_indices: dict[str, MacroIndicator] = field(default_factory=dict)
    fx: dict[str, MacroIndicator] = field(default_factory=dict)
    commodities: dict[str, MacroIndicator] = field(default_factory=dict)
    volatility: dict[str, MacroIndicator] = field(default_factory=dict)
    bonds: dict[str, MacroIndicator] = field(default_factory=dict)
    # ── 신규 카테고리 ──
    sectors: dict[str, MacroIndicator] = field(default_factory=dict)
    mega_caps: dict[str, MacroIndicator] = field(default_factory=dict)
    style: dict[str, MacroIndicator] = field(default_factory=dict)
    asia: dict[str, MacroIndicator] = field(default_factory=dict)
    europe: dict[str, MacroIndicator] = field(default_factory=dict)
    credit: dict[str, MacroIndicator] = field(default_factory=dict)

    @property
    def all_indicators(self) -> dict[str, MacroIndicator]:
        return {**self.us_indices, **self.fx, **self.commodities,
                **self.volatility, **self.bonds,
                **self.sectors, **self.mega_caps, **self.style,
                **self.asia, **self.europe, **self.credit}

    @property
    def yield_spread(self) -> float | None:
        """10Y - 2Y 수익률 스프레드 (양수=정상, 음수=역전)."""
        us10y = self.bonds.get("US10Y")
        us2y = self.bonds.get("US2Y")
        if us10y and us2y:
            return round(us10y.close - us2y.close, 3)
        return None

    # VIX 기반 시장 체제 경계값
    _VIX_STABLE = 15
    _VIX_NORMAL = 20
    _VIX_FEAR = 30

    @property
    def market_regime(self) -> str:
        """VIX 기반 시장 체제 판단."""
        vix = self.volatility.get("VIX")
        if vix is None:
            return "판단불가"
        v = vix.close
        if v < self._VIX_STABLE:
            return "안정"
        elif v < self._VIX_NORMAL:
            return "보통"
        elif v < self._VIX_FEAR:
            return "불안"
        else:
            return "공포"

    @property
    def sector_top_bottom(self) -> tuple[str, str] | None:
        """섹터 ETF 중 최고/최저 등락률 섹터 반환."""
        if not self.sectors:
            return None
        best = max(self.sectors.values(), key=lambda x: x.change_pct)
        worst = min(self.sectors.values(), key=lambda x: x.change_pct)
        return (f"{best.name}({best.change_pct:+.2f}%)",
                f"{worst.name}({worst.change_pct:+.2f}%)")

    @property
    def growth_value_ratio(self) -> float | None:
        """QQQ/VTV 등락률 차이. 양수=성장 우위, 음수=가치 우위."""
        qqq = self.style.get("QQQ")
        vtv = self.style.get("VTV")
        if qqq and vtv:
            return round(qqq.change_pct - vtv.change_pct, 2)
        return None

    @property
    def credit_stress(self) -> str | None:
        """HYG 등락률 기반 신용 스트레스 판단."""
        hyg = self.credit.get("HYG")
        if hyg is None:
            return None
        if hyg.change_pct < -1.0:
            return "경고 (하이일드 급락)"
        elif hyg.change_pct < -0.3:
            return "주의"
        else:
            return "안정"

    @property
    def mag7_avg(self) -> float | None:
        """Magnificent 7 평균 등락률."""
        if not self.mega_caps:
            return None
        return round(sum(m.change_pct for m in self.mega_caps.values())
                     / len(self.mega_caps), 2)

    def to_narrative(self) -> str:
        """LLM 컨텍스트용 한국어 거시경제 '돈의 흐름' 내러티브."""
        parts: list[str] = []

        sp = self.us_indices.get("SP500")
        nq = self.us_indices.get("NASDAQ")
        if sp and nq:
            direction = "상승" if sp.change_pct > 0 else "하락"
            parts.append(f"전일 미국 증시는 S&P500 {sp.change_pct:+.2f}%, "
                         f"나스닥 {nq.change_pct:+.2f}%로 {direction} 마감.")

        soxx = self.us_indices.get("SOXX")
        if soxx:
            parts.append(f"필라델피아 반도체지수 {soxx.change_pct:+.2f}%.")

        regime = self.market_regime
        vix = self.volatility.get("VIX")
        if vix:
            parts.append(f"VIX {vix.close:.1f} ({regime} 구간).")

        spread = self.yield_spread
        if spread is not None:
            if spread < 0:
                parts.append(f"장단기 금리차 {spread:.3f}%p (역전 — 경기침체 경고 신호).")
            elif spread < 0.5:
                parts.append(f"장단기 금리차 {spread:.3f}%p (축소 — 긴축 우려 잔존).")
            else:
                parts.append(f"장단기 금리차 {spread:.3f}%p (정상 범위).")

        fx = self.fx.get("USDKRW")
        if fx and not math.isnan(fx.change_pct):
            direction = "약세" if fx.change_pct > 0 else "강세"
            parts.append(f"원/달러 {fx.close:,.2f}원 ({fx.change_pct:+.2f}%, 원화 {direction}).")

        gold = self.commodities.get("GOLD")
        if gold:
            parts.append(f"금 {gold.close:,.2f}달러 ({gold.change_pct:+.2f}%).")

        dxy = self.fx.get("DXY")
        if dxy and not (dxy.change_pct != dxy.change_pct):  # NaN check
            parts.append(f"달러인덱스 {dxy.close:.2f} ({dxy.change_pct:+.2f}%).")

        # 돈의 흐름 해석
        if vix and gold:
            if vix.change_pct > 5 and gold.change_pct > 1:
                parts.append("VIX 급등 + 금 강세 → 안전자산 선호(Risk-Off) 흐름.")
            elif vix.change_pct < -5 and gold.change_pct < -1:
                parts.append("VIX 급락 + 금 약세 → 위험자산 선호(Risk-On) 흐름.")

        # 섹터 로테이션
        top_bottom = self.sector_top_bottom
        if top_bottom:
            parts.append(f"섹터: {top_bottom[0]} 주도, {top_bottom[1]} 약세.")

        # 성장 vs 가치
        gv = self.growth_value_ratio
        if gv is not None:
            label = "성장주 우위" if gv > 0 else "가치주 우위"
            parts.append(f"QQQ-VTV 스프레드 {gv:+.2f}%p → {label}.")

        # 신용 리스크
        cs = self.credit_stress
        if cs:
            hyg = self.credit.get("HYG")
            hyg_pct = f"{hyg.change_pct:+.2f}%" if hyg else ""
            parts.append(f"HYG {hyg_pct} → 신용 스트레스 {cs}.")

        # Mag7 평균
        m7 = self.mag7_avg
        if m7 is not None:
            parts.append(f"Mag7 평균 {m7:+.2f}% → 대형 기술주 {'강세' if m7 > 0 else '약세'}.")

        # 아시아
        if self.asia:
            asia_parts = [f"{ind.name} {ind.change_pct:+.2f}%"
                          for ind in self.asia.values()]
            parts.append(f"아시아: {', '.join(asia_parts)}.")

        # 유럽
        if self.europe:
            eu_parts = [f"{ind.name} {ind.change_pct:+.2f}%"
                        for ind in self.europe.values()]
            parts.append(f"유럽: {', '.join(eu_parts)}.")

        return " ".join(parts)

    def to_summary_dict(self) -> dict[str, dict]:
        """LLM 프롬프트용 요약 딕셔너리."""
        result = {}
        for key, ind in self.all_indicators.items():
            result[ind.name] = {
                "Close": ind.close,
                "ChangePct": ind.change_pct,
                "Change": ind.change,
            }
        if self.yield_spread is not None:
            result["_yield_spread"] = self.yield_spread
        result["_market_regime"] = self.market_regime
        result["_sector_top_bottom"] = self.sector_top_bottom
        result["_growth_value_ratio"] = self.growth_value_ratio
        result["_credit_stress"] = self.credit_stress
        result["_mag7_avg"] = self.mag7_avg
        return result

    def is_empty(self) -> bool:
        return not any([self.us_indices, self.fx, self.commodities,
                        self.volatility, self.bonds,
                        self.sectors, self.mega_caps, self.style,
                        self.asia, self.europe, self.credit])


# ── 수집 대상 정의 ──────────────────────────────────────────────────
_US_INDICES = {
    "SP500": ("S&P 500", "US500"),
    "NASDAQ": ("NASDAQ", "IXIC"),
    "DOW": ("다우존스", "DJI"),
}

_FX = {
    "USDKRW": ("원/달러", "USD/KRW"),
}

_COMMODITIES = {
    "WTI": ("WTI유", "CL=F"),
    "GOLD": ("금", "GC=F"),
}

_VOLATILITY = {
    "VIX": ("VIX", "VIX"),
}

_BONDS = {
    "US10Y": ("미국채10Y", "US10YT"),
}

# US2YT는 FDR/Yahoo에서 404 발생 가능 — 실패 시 yield_spread=None으로 graceful 처리
_BONDS_OPTIONAL = {
    "US2Y": ("미국채2Y", "US2YT"),
}

_SEMI = {
    "SOXX": ("필라델피아반도체", "SOXX"),
}

# DX-Y.NYB는 NaN 반환 가능 — 실패 시 내러티브에서 제외
_FX_OPTIONAL = {
    "DXY": ("달러인덱스", "DX-Y.NYB"),
}

# US 섹터 ETF (11개)
_SECTORS = {
    "XLK": ("기술", "XLK"),
    "XLF": ("금융", "XLF"),
    "XLE": ("에너지", "XLE"),
    "XLV": ("헬스케어", "XLV"),
    "XLI": ("산업재", "XLI"),
    "XLC": ("커뮤니케이션", "XLC"),
    "XLY": ("경기소비재", "XLY"),
    "XLP": ("필수소비재", "XLP"),
    "XLRE": ("부동산", "XLRE"),
    "XLU": ("유틸리티", "XLU"),
    "XLB": ("소재", "XLB"),
}

# Magnificent 7 + 한국 관련주
_MEGA_CAPS = {
    "NVDA": ("엔비디아", "NVDA"),
    "AAPL": ("애플", "AAPL"),
    "MSFT": ("마이크로소프트", "MSFT"),
    "AMZN": ("아마존", "AMZN"),
    "GOOG": ("알파벳", "GOOG"),
    "META": ("메타", "META"),
    "TSLA": ("테슬라", "TSLA"),
}

# 소형주 / 성장 vs 가치
_STYLE = {
    "IWM": ("러셀2000", "IWM"),
    "QQQ": ("나스닥100ETF", "QQQ"),
    "VTV": ("가치ETF", "VTV"),
}

# 아시아 지수 (전일 종가)
_ASIA = {
    "NIKKEI": ("닛케이225", "^N225"),
    "HSI": ("항셍", "^HSI"),
    "SSEC": ("상해종합", "000001.SS"),
}

# 유럽 지수 (전일 종가)
_EUROPE = {
    "DAX": ("독일DAX", "^GDAXI"),
    "STOXX50": ("유로스톡스50", "^STOXX50E"),
}

# 추가 원자재
_COMMODITIES_EXT = {
    "COPPER": ("구리", "HG=F"),
    "NATGAS": ("천연가스", "NG=F"),
}

# 신용/리스크 ETF
_CREDIT = {
    "HYG": ("하이일드채권", "HYG"),
    "TLT": ("미장기채20Y+", "TLT"),
}

# FDR 코드 → yfinance 티커 매핑 (FDR 미설치 시 fallback용)
_FDR_TO_YF: dict[str, str] = {
    "US500": "^GSPC",
    "IXIC": "^IXIC",
    "DJI": "^DJI",
    "SOXX": "SOXX",
    "USD/KRW": "USDKRW=X",
    "DX-Y.NYB": "DX-Y.NYB",
    "CL=F": "CL=F",
    "GC=F": "GC=F",
    "VIX": "^VIX",
    "US10YT": "^TNX",
    "US2YT": "2YY=F",
    # 아시아·유럽 지수 (yfinance 전용)
    "^N225": "^N225",
    "^HSI": "^HSI",
    "000001.SS": "000001.SS",
    "^GDAXI": "^GDAXI",
    "^STOXX50E": "^STOXX50E",
    # 추가 원자재·신용
    "HG=F": "HG=F",
    "NG=F": "NG=F",
    "HYG": "HYG",
    "TLT": "TLT",
}


def _df_to_indicator(
    key: str, display_name: str, code: str, df: pd.DataFrame,
) -> MacroIndicator | None:
    """DataFrame → MacroIndicator 변환 (FDR/yfinance 공통)."""
    if df.empty or len(df) < 2:
        logger.warning("macro %s (%s): insufficient data", key, code)
        return None

    latest = df.iloc[-1]
    prev = df.iloc[-2]
    close = float(latest["Close"])
    prev_close = float(prev["Close"])

    # NaN 체크 (DXY 등 일부 지표에서 발생)
    if math.isnan(close) or math.isnan(prev_close):
        logger.warning("macro %s (%s): NaN value detected", key, code)
        return None

    close = round(close, 2)
    prev_close = round(prev_close, 2)
    change = round(close - prev_close, 2)
    change_pct = round((change / prev_close * 100), 2) if prev_close else 0.0

    return MacroIndicator(
        name=display_name,
        code=code,
        close=close,
        prev_close=prev_close,
        change=change,
        change_pct=change_pct,
        date=str(df.index[-1].date()),
    )


def _fetch_via_yfinance(
    key: str, display_name: str, fdr_code: str, start, end,
) -> MacroIndicator | None:
    """yfinance를 사용하여 지표를 가져온다."""
    if yf is None:
        return None
    yf_code = _FDR_TO_YF.get(fdr_code, fdr_code)
    try:
        df = yf.download(yf_code, start=str(start), end=str(end),
                         progress=False, auto_adjust=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return _df_to_indicator(key, display_name, fdr_code, df)
    except Exception as exc:
        logger.warning("macro %s (%s → yf:%s) failed: %s", key, fdr_code, yf_code, exc)
        return None


def _fetch_indicator(
    key: str, display_name: str, fdr_code: str, days: int = 15
) -> MacroIndicator | None:
    """지표를 가져와 MacroIndicator로 변환. FDR 우선, 실패/NaN 시 yfinance fallback."""
    end = datetime.now().date()
    start = end - timedelta(days=days)

    # 1) FDR 시도
    if _USE_FDR:
        try:
            df = fdr.DataReader(fdr_code, start, end)
            result = _df_to_indicator(key, display_name, fdr_code, df)
            if result is not None:
                return result
            # FDR이 NaN/빈 데이터 → yfinance fallback
            logger.warning("macro %s (%s) FDR returned empty/NaN, trying yfinance...", key, fdr_code)
        except Exception as exc:
            logger.warning("macro %s (%s) FDR failed: %s, trying yfinance...", key, fdr_code, exc)

    # 2) yfinance fallback
    result = _fetch_via_yfinance(key, display_name, fdr_code, start, end)
    if result is not None:
        return result

    # 양쪽 모두 실패
    if yf is None and not _USE_FDR:
        logger.error("macro %s: neither FinanceDataReader nor yfinance available", key)
    else:
        logger.warning("macro %s (%s): both FDR and yfinance failed", key, fdr_code)
    return None


def _fetch_group(
    source: dict[str, tuple[str, str]],
    target: dict[str, MacroIndicator],
    days: int = 15,
) -> None:
    """지표 그룹을 일괄 수집하여 target dict에 저장한다."""
    for key, (name, code) in source.items():
        ind = _fetch_indicator(key, name, code, days=days)
        if ind:
            target[key] = ind
            logger.info("  %s", ind)


def fetch_macro_snapshot() -> MacroSnapshot:
    """거시경제 전체 스냅샷을 수집한다."""
    snap = MacroSnapshot()

    _fetch_group(_US_INDICES, snap.us_indices)
    _fetch_group(_SEMI, snap.us_indices)
    _fetch_group(_FX, snap.fx)
    _fetch_group(_FX_OPTIONAL, snap.fx)
    _fetch_group(_COMMODITIES, snap.commodities)
    _fetch_group(_COMMODITIES_EXT, snap.commodities)
    _fetch_group(_VOLATILITY, snap.volatility)
    _fetch_group(_BONDS, snap.bonds)
    _fetch_group(_BONDS_OPTIONAL, snap.bonds)
    _fetch_group(_SECTORS, snap.sectors)
    _fetch_group(_MEGA_CAPS, snap.mega_caps)
    _fetch_group(_STYLE, snap.style)
    _fetch_group(_ASIA, snap.asia, days=30)  # 중국 데이터 누락 빈도 높아 확장
    _fetch_group(_EUROPE, snap.europe)
    _fetch_group(_CREDIT, snap.credit)

    total = len(snap.all_indicators)
    logger.info("macro snapshot: %d indicators fetched", total)
    return snap


if __name__ == "__main__":
    import sys

    for stream in (sys.stdout, sys.stderr):
        if stream.encoding and stream.encoding.lower() != "utf-8":
            try:
                stream.reconfigure(encoding="utf-8")
            except Exception:
                pass
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    snap = fetch_macro_snapshot()

    print(f"\n{'='*60}")
    print("거시경제 스냅샷")
    print(f"{'='*60}")

    sections = [
        ("미국 증시", snap.us_indices),
        ("환율", snap.fx),
        ("원자재", snap.commodities),
        ("변동성", snap.volatility),
        ("채권", snap.bonds),
        ("섹터 ETF", snap.sectors),
        ("Mag7", snap.mega_caps),
        ("스타일", snap.style),
        ("아시아", snap.asia),
        ("유럽", snap.europe),
        ("신용", snap.credit),
    ]
    for title, indicators in sections:
        if indicators:
            print(f"\n[{title}]")
            for ind in indicators.values():
                print(f"  {ind}")
