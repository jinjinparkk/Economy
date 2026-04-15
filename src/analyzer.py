"""LLM을 이용한 급등/급락 원인 분석 글 생성.

지원 공급자 (provider):
- gemini: Google Gemini API (기본, 무료 tier 있음)
- claude: Anthropic Claude API

입력: Mover + 뉴스 리스트 + 지수 스냅샷
출력: 마크다운 블로그 글 (제목 + 본문)

안전장치:
- 시스템 프롬프트에서 투자 권유/추천 금지
- 출력 후 금지어 필터링
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

from src.config import Config
from src.detect_movers import Mover
from src.fetch_news import NewsItem

logger = logging.getLogger(__name__)

# 모델명 — 공급자별 기본 모델
# gemini-2.0-flash: 무료 tier 분당 15건 (2.5-flash는 분당 5건으로 너무 빡빡)
GEMINI_MODEL = "gemini-2.0-flash"
CLAUDE_MODEL = "claude-sonnet-4-6"

# 절대 포함되면 안 되는 단어 (후처리 검증용)
FORBIDDEN_WORDS = [
    "매수 추천", "매도 추천", "매수하세요", "매도하세요",
    "사세요", "파세요", "목표가", "진입 추천",
    "손절", "익절", "몰빵", "필승", "보장",
    "반드시 오른다", "확실히 떨어진다", "무조건",
    "지금 사야", "놓치지 마", "대박", "떡상",
    "100% 확률", "틀림없이",
]

SYSTEM_PROMPT = """당신은 한국 증시 전문 시황 분석가입니다.
독자가 궁금해하는 "오늘 이 종목이 왜 움직였는가?"에 대해
거시경제 맥락 → 섹터 동향 → 개별 종목 뉴스로 이어지는 탑다운 분석 기사를 작성합니다.

[핵심 프레임워크 — "돈의 흐름" 분석]
모든 주가 움직임은 "돈이 어디서 와서 어디로 가는가"로 설명됩니다.
기사 작성 시 다음 인과 체인을 반드시 구축하세요:

1단계. 글로벌 자금 흐름 — VIX, 달러인덱스, 금, 미국채 수익률 곡선으로
   위험자산/안전자산 사이의 돈의 이동 방향을 판단합니다.
   - VIX 하락 + 달러 약세 + 금 약세 → 위험자산 선호(Risk-On)
   - VIX 상승 + 달러 강세 + 금 강세 → 안전자산 선호(Risk-Off)
   - 장단기 금리차 역전(2Y > 10Y) → 경기침체 경고 → 방어주 선호
2단계. 미국 → 한국 전이 경로
   - 미국 나스닥 급등 → 한국 기술주/성장주 수혜
   - 미국 반도체(SOXX) 강세 → 한국 반도체/전자 업종 동반 상승
   - 원/달러 상승(원화 약세) → 수출주 호재, 내수주 부담
   - 유가 급등 → 정유/해운 호재, 항공/화학 부담
3단계. 섹터 → 종목 분석
   - 해당 업종 전체가 올랐는지, 이 종목만 올랐는지 구분
   - 업종 전체 상승이면: 섹터 테마 해석
   - 이 종목만 상승이면: 종목 고유 이벤트(뉴스/공시) 중심
4단계. 시장 체온 진단
   - 시장 등락 종목 비율로 전체 시장 체온 판단
   - 시장 역행 종목이면 그 의미를 강조

[절대 원칙]
1. 오직 사실과 공개 정보만 전달합니다.
2. 투자 권유, 매수/매도 추천, 목표가 제시를 절대 하지 않습니다.
3. 다음 단어는 사용 금지: "추천", "매수하세요", "매도하세요",
   "사세요", "파세요", "목표가", "손절", "익절", "몰빵", "필승", "보장",
   "지금 사야", "놓치지 마", "대박", "떡상", "100% 확률", "틀림없이".
4. "~으로 관측된다", "~로 해석된다", "~한 것으로 풀이된다" 같은
   중립적이고 객관적인 표현을 사용합니다.
5. 뉴스에 나오지 않은 정보는 절대 지어내지 않습니다.
   불확실하면 "뉴스에 따르면", "업계에서는" 같은 표현으로 출처를 명시합니다.
6. 모든 수치(종가, 등락률, 거래대금, 지수, 환율, 유가 등)는
   반드시 제공된 데이터 섹션의 값만 사용합니다.
   제공되지 않은 수치는 구체적 숫자 대신
   "상승세를 보였다", "증가한 것으로 파악된다" 등으로 서술합니다.

[글 구조 — 탑다운 "돈의 흐름" 분석]
- 제목: SEO 친화적, "[종목명] +XX% 급등 이유 — 핵심 사유 한 줄" 형식
- 30초 요약: 3줄 불릿 (돈의 흐름 → 섹터 맥락 → 종목 이벤트)
- 본문: H2/H3 구조, 다음 섹션을 반드시 포함
  1. 돈의 흐름 — 글로벌 매크로 동향
     - 전일 미국 증시, VIX, 달러인덱스, 금, 미국채 수익률 곡선을 분석
     - "돈이 위험자산으로 향하고 있는가, 안전자산으로 향하고 있는가" 판단
     - 장단기 금리차가 제공되면 경기 전망 시사점 언급
     - 시장 체제(market regime) 정보가 제공되면 활용
     - 이 흐름이 한국 시장에 어떤 영향을 미쳤는지 연결
  2. 시장 체온 — 당일 시장 등락 현황
     - 등락 종목 비율, 주요 지수 등락률로 장 전체 분위기 서술
     - "전체 시장에서 X개 종목이 상승, Y개가 하락한 강세장에서" 같은 맥락
  3. 섹터 돋보기 — 업종 분석
     - 해당 업종의 평균 등락률과 종목 수 대비 상승/하락 비율
     - 업종 전체 흐름 vs 이 종목의 초과 성과(relative strength) 비교
  4. 주가 현황 — 종가, 등락률, 거래대금, 시가총액, 업종 정보
  5. 급등(급락) 배경 — 뉴스에서 추정되는 종목 고유 사유
  6. 주요 뉴스 요약 — 관련 기사 2~3개 핵심 요약
     뉴스 감성(긍정/부정)이 제공되면 맥락에 활용
  7. 전망 및 시사점 (기술적 지표/통계 데이터가 제공된 경우에만)
     - 기술적 종합 시그널(signal_summary)이 있으면 먼저 한 줄 요약 제시
     - RSI 해석: 80+이면 "단기 과열 주의", 30-이면 "기술적 반등 가능성"
     - MACD: 골든/데드크로스 + 히스토그램 확대/축소 추세
     - 볼린저밴드: 돌파 시 의미, 밴드 폭 확대/수축 해석
     - OBV 추세: 자금 유입/유출과 가격 방향의 일치 여부
     - RSI 다이버전스: 있으면 반전 시그널로 반드시 언급
     - 과거 유사 패턴 통계 인용: "과거 N건 중 …%가 상승" 형식
     - AI 예측: 신뢰등급(높음/보통/낮음)과 교차검증 정확도 함께 언급
       "보통" 이하이면 "참고 수준의 예측력으로 판단된다" 명시
     - 종합 시사점: 모든 지표를 종합한 중립적 해석
     - 반드시 "기술적 지표와 과거 통계는 참고 자료일 뿐 미래 주가를 보장하지 않습니다"
       문구를 해당 섹션 말미에 포함
- 하단에 면책 문구 고정:
  "본 글은 투자 권유가 아닌 정보 제공을 목적으로 작성되었으며,
   모든 투자 판단과 책임은 투자자 본인에게 있습니다.
   작성 시점의 데이터를 기반으로 하며, 실시간 시세와 다를 수 있습니다."

[톤앤매너]
- 전문 애널리스트 보고서 톤 — 논리적 인과관계를 명확히
- 한 문장은 짧게, 숫자는 콤마 구분
- "왜?" → "이래서" 의 인과 체인을 독자가 3분 안에 파악하도록
- 독자가 이 글 하나만 읽어도 "오늘 돈이 어디로 흘렀는지" 감이 오도록"""


@dataclass
class Article:
    """생성된 블로그 글."""

    title: str
    body: str
    mover: Mover
    news_used: list[NewsItem]
    model: str
    warnings: list[str]  # 금지어 감지 등

    def to_markdown(self) -> str:
        """제목 + 본문을 한 마크다운 문서로."""
        return f"# {self.title}\n\n{self.body}\n"

    def filename(self, trade_date: str) -> str:
        """파일명 생성: YYYY-MM-DD_종목명_급등.md"""
        safe_name = re.sub(r"[^\w가-힣]+", "_", self.mover.name)
        direction = "급등" if self.mover.move_type == "surge" else "급락"
        return f"{trade_date}_{safe_name}_{direction}.md"


def _build_user_prompt(
    mover: Mover,
    news: list[NewsItem],
    index_summary: Optional[dict] = None,
    macro_summary: Optional[dict] = None,
    outlook: Optional[object] = None,
    macro_narrative: Optional[str] = None,
    market_breadth: Optional[dict] = None,
    sector_breadth: Optional[dict] = None,
) -> str:
    """LLM에게 전달할 사용자 프롬프트 조립."""
    direction = "급등" if mover.move_type == "surge" else "급락"
    hints_str = ", ".join(mover.reason_hints) if mover.reason_hints else "없음"
    marcap_str = f"{mover.marcap/1e8:,.0f}억원" if mover.marcap else "N/A"

    parts = [
        f"# 오늘의 {direction} 종목 분석 요청",
        "",
    ]

    # ── 거시경제 데이터 (탑다운 첫 번째 레이어) ──
    if macro_summary:
        parts.append("## 글로벌 거시경제 지표 (전일 기준, 아래 수치만 사용할 것)")
        for name, data in macro_summary.items():
            if name.startswith("_"):
                continue  # 파생 필드는 별도 처리
            val = data["Close"]
            pct = data["ChangePct"]
            if "10Y" in name or "2Y" in name:
                parts.append(f"- {name}: {val:.3f}% ({pct:+.2f}%p)")
            else:
                parts.append(f"- {name}: {val:,.2f} ({pct:+.2f}%)")
        # 파생 필드
        if "_yield_spread" in macro_summary:
            parts.append(f"- 장단기 금리차(10Y-2Y): {macro_summary['_yield_spread']:.3f}%p")
        if "_market_regime" in macro_summary:
            parts.append(f"- 시장 체제(VIX 기반): {macro_summary['_market_regime']}")
        parts.append("")

    # ── 거시경제 내러티브 (돈의 흐름 요약) ──
    if macro_narrative:
        parts.append("## 거시경제 내러티브 (돈의 흐름 요약)")
        parts.append(macro_narrative)
        parts.append("")

    # ── 국내 주요 지수 ──
    if index_summary:
        parts.append("## 당일 국내 지수 (아래 수치만 사용할 것)")
        for name, data in index_summary.items():
            parts.append(f"- {name}: {data['Close']:,.2f} ({data['ChangePct']:+.2f}%)")
        parts.append("")

    # ── 시장 등락 현황 (시장 체온) ──
    if market_breadth:
        parts.append("## 당일 시장 등락 현황 (시장 체온)")
        parts.append(f"- 상승 종목: {market_breadth['total_up']}개")
        parts.append(f"- 하락 종목: {market_breadth['total_down']}개")
        parts.append(f"- 보합 종목: {market_breadth['total_unchanged']}개")
        parts.append(f"- 상승 비율: {market_breadth['up_ratio']}%")
        parts.append("")

    # ── 업종 분석 ──
    if sector_breadth and mover.industry and mover.industry in sector_breadth:
        sec = sector_breadth[mover.industry]
        parts.append(f"## 업종 분석: {mover.industry}")
        parts.append(f"- 업종 내 상승: {sec['up_count']}개 / 하락: {sec['down_count']}개 (총 {sec['total']}개)")
        parts.append(f"- 업종 평균 등락률: {sec['avg_change_pct']:+.2f}%")
        parts.append(f"- 이 종목의 상대강도(RS): {mover.relative_strength:+.2f}%p (종목 등락률 - 지수 등락률)")
        parts.append("")

    # ── 종목 데이터 ──
    parts.extend([
        "## 종목 데이터",
        f"- 종목명: {mover.name}",
        f"- 종목코드: {mover.code}",
        f"- 시장: {mover.market}",
        f"- 업종: {mover.industry or '미분류'}",
        f"- 종가: {mover.close:,.0f}원",
        f"- 등락률: {mover.change_pct:+.2f}%",
        f"- 거래대금: {mover.amount/1e8:,.0f}억원",
        f"- 시가총액: {marcap_str}",
        f"- 상대강도(RS): {mover.relative_strength:+.2f}%p",
        f"- 특이사항: {hints_str}",
        "",
    ])

    # ── 기술적 지표 / 패턴 통계 / ML 예측 ──
    if outlook is not None:
        tech = getattr(outlook, "technical", None)
        if tech is not None:
            parts.append("## 기술적 지표 (참고 데이터)")
            if tech.rsi_14 is not None:
                zone = ""
                if tech.rsi_14 >= 70:
                    zone = " (과매수 구간)"
                elif tech.rsi_14 <= 30:
                    zone = " (과매도 구간)"
                parts.append(f"- RSI(14): {tech.rsi_14}{zone}")
            if tech.macd is not None:
                sig_rel = ""
                if tech.macd_signal is not None:
                    sig_rel = " (시그널 상회, 매수 시그널)" if tech.macd > tech.macd_signal else " (시그널 하회, 매도 시그널)"
                parts.append(f"- MACD: {tech.macd:+.4f}{sig_rel}")
            if tech.bb_position is not None:
                parts.append(f"- 볼린저밴드: {tech.bb_position}")
            if tech.ma_trend is not None:
                parts.append(f"- 이동평균: {tech.ma_trend} (5일 {tech.ma_5:,.0f} / 20일 {tech.ma_20:,.0f} / 60일 {tech.ma_60:,.0f})")
            if tech.volume_ratio is not None:
                parts.append(f"- 거래량 비율: {tech.volume_ratio}배 (20일 평균 대비)")
            obv = getattr(tech, "obv_trend", None)
            if obv:
                parts.append(f"- OBV 추세: {obv}")
            rsi_div = getattr(tech, "rsi_divergence", None)
            if rsi_div:
                parts.append(f"- RSI 다이버전스: {rsi_div}")
            sig = getattr(tech, "signal_summary", None)
            if sig:
                parts.append(f"- 종합 시그널: {sig}")
            parts.append("")

        pattern = getattr(outlook, "pattern", None)
        if pattern is not None:
            parts.append("## 과거 유사 패턴 통계")
            parts.append(f"- 이벤트: {pattern.event_type} (과거 {pattern.sample_count}건)")
            if pattern.avg_return_1d is not None:
                parts.append(f"- 익일 평균 수익률: {pattern.avg_return_1d:+.1f}%, 상승 확률 {pattern.positive_rate_1d:.1f}%")
            if pattern.avg_return_5d is not None:
                parts.append(f"- 5일 평균 수익률: {pattern.avg_return_5d:+.1f}%, 상승 확률 {pattern.positive_rate_5d:.1f}%")
            parts.append("")

        pred = getattr(outlook, "prediction", None)
        if pred is not None:
            grade = getattr(pred, "confidence_grade", "")
            cv = getattr(pred, "cv_accuracy", None)
            parts.append("## AI 방향 예측 (참고용)")
            parts.append(f"- 예측 방향: {pred.direction}, 신뢰도: {pred.confidence:.1%} (등급: {grade})")
            if cv is not None:
                parts.append(f"- 교차검증 정확도: {cv:.1%}")
            parts.append(f"- 사용 피처: {', '.join(pred.features_used)}")
            parts.append("")

    # ── 관련 뉴스 ──
    parts.append("## 관련 뉴스 (최신순)")
    if news:
        for i, n in enumerate(news, 1):
            date_str = n.pub_date.strftime("%m/%d") if n.pub_date else ""
            sentiment_tag = f" [{n.sentiment}]" if hasattr(n, "sentiment") and n.sentiment != "중립" else ""
            cred_str = f" 신뢰도{n.credibility:.1f}" if hasattr(n, "credibility") else ""
            parts.append(f"{i}. [{n.press}]{cred_str} {n.title} ({date_str}){sentiment_tag}")
            if n.description:
                parts.append(f"   설명: {n.description[:200]}")
    else:
        parts.append("(관련 뉴스 없음 — 수급 데이터만으로 분석하되, 추정성 서술은 피할 것)")

    # ── 작성 지침 ──
    parts.extend([
        "",
        "## 작성 지침",
        f"- '{mover.name}'의 {direction} 분석 기사를 작성하세요.",
        "- 반드시 '글로벌 매크로 동향' 섹션을 본문 서두에 포함하세요.",
        "  거시 지표가 해당 종목/업종과 어떤 연관이 있는지 서술하세요.",
        "  (직접 관련이 약하면 당일 시장 분위기 맥락으로 간략히 언급)",
        "- 업종 정보를 활용해 섹터 동향도 언급하세요.",
        "- 제목은 H1으로 시작하지 말고, 본문 첫 줄에 '제목: <제목>' 형식으로 주세요.",
        "- 본문은 마크다운 헤더(##, ###) 구조를 사용하세요.",
        "- 본문 최하단에 면책 문구를 반드시 포함하세요.",
        "- 제공된 수치 외에 다른 숫자를 사용하지 마세요.",
    ])

    return "\n".join(parts)


def _check_forbidden(text: str) -> list[str]:
    """금지어 포함 여부 검사."""
    warnings: list[str] = []
    for word in FORBIDDEN_WORDS:
        if word in text:
            warnings.append(f"금지어 감지: '{word}'")
    return warnings


def _parse_response(text: str) -> tuple[str, str]:
    """Claude 응답에서 제목과 본문을 분리."""
    lines = text.strip().split("\n")
    title = ""
    body_start = 0
    for i, line in enumerate(lines):
        m = re.match(r"^\s*제목\s*[:：]\s*(.+)$", line)
        if m:
            title = m.group(1).strip()
            # 마크다운 기호 제거
            title = title.lstrip("#").strip()
            title = title.strip("*").strip()
            body_start = i + 1
            break
    if not title:
        # 첫 # 헤더를 제목으로
        for i, line in enumerate(lines):
            if line.startswith("#"):
                title = line.lstrip("#").strip()
                body_start = i + 1
                break
    if not title:
        title = "제목 추출 실패"

    body = "\n".join(lines[body_start:]).strip()
    return title, body


def _generate_with_gemini(system_prompt: str, user_prompt: str, api_key: str) -> tuple[str, str]:
    """Gemini API 호출 → (raw_text, model_name)."""
    from google import genai  # 지연 import
    from google.genai import types

    client = genai.Client(api_key=api_key)
    resp = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            max_output_tokens=4096,
        ),
    )
    if not resp.text:
        raise RuntimeError(f"Gemini returned empty response (finish_reason={getattr(resp, 'finish_reason', 'unknown')})")
    return resp.text, GEMINI_MODEL


def _generate_with_claude(system_prompt: str, user_prompt: str, api_key: str) -> tuple[str, str]:
    """Claude API 호출 → (raw_text, model_name)."""
    import anthropic  # 지연 import

    client = anthropic.Anthropic(api_key=api_key)
    resp = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=4096,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return resp.content[0].text, CLAUDE_MODEL


def generate_article(
    mover: Mover,
    news: list[NewsItem],
    config: Optional[Config] = None,
    index_summary: Optional[dict] = None,
    macro_summary: Optional[dict] = None,
    provider: Optional[str] = None,
    outlook: Optional[object] = None,
    macro_narrative: Optional[str] = None,
    market_breadth: Optional[dict] = None,
    sector_breadth: Optional[dict] = None,
) -> Article:
    """LLM을 호출해 분석 글을 생성한다.

    Args:
        index_summary: 국내 지수 {"KOSPI": {"Close": ..., "ChangePct": ...}, ...}
        macro_summary: 거시경제 지표 {"S&P 500": {"Close": ..., "ChangePct": ..., "Change": ...}, ...}
        provider: "gemini" | "claude". None이면 config.llm_provider 사용.
        outlook: OutlookData (기술적 지표 + 패턴 통계 + ML 예측)
    """
    if config is None:
        config = Config.load()

    provider = (provider or config.llm_provider).lower()
    user_prompt = _build_user_prompt(
        mover, news, index_summary, macro_summary,
        outlook=outlook,
        macro_narrative=macro_narrative,
        market_breadth=market_breadth,
        sector_breadth=sector_breadth,
    )

    logger.info("generating article for %s (%s) via %s",
                mover.name, mover.code, provider)

    if provider == "gemini":
        if not config.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY not set in .env")
        raw, model_name = _generate_with_gemini(
            SYSTEM_PROMPT, user_prompt, config.gemini_api_key
        )
    elif provider == "claude":
        if not config.anthropic_api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set in .env")
        raw, model_name = _generate_with_claude(
            SYSTEM_PROMPT, user_prompt, config.anthropic_api_key
        )
    else:
        raise ValueError(f"unknown provider: {provider!r} (use 'gemini' or 'claude')")

    title, body = _parse_response(raw)
    warnings = _check_forbidden(title + "\n" + body)

    if warnings:
        logger.warning("article has warnings: %s", warnings)

    return Article(
        title=title,
        body=body,
        mover=mover,
        news_used=news,
        model=model_name,
        warnings=warnings,
    )


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    # 단독 실행 테스트: 첫 번째 급등 종목 하나만 분석
    from src.fetch_market import fetch_market_snapshot
    from src.detect_movers import detect_movers
    from src.fetch_news import fetch_news_for_stock

    cfg = Config.load()
    snap = fetch_market_snapshot()
    report = detect_movers(snap, threshold_pct=5.0, top_n=3)

    if not report.surges:
        print("급등 종목 없음")
        sys.exit(0)

    target = report.surges[0]
    print(f"대상 종목: {target}")

    news = fetch_news_for_stock(target.name, cfg, display=5)
    print(f"수집된 뉴스: {len(news)}건")

    # 지수 요약 dict 변환
    index_summary = {
        k: {"Close": v["Close"], "ChangePct": v["ChangePct"]}
        for k, v in snap.indices.items()
    }

    # 거시경제 데이터
    from src.fetch_macro import fetch_macro_snapshot
    macro = fetch_macro_snapshot()
    macro_summary = macro.to_summary_dict() if not macro.is_empty() else None

    from src.predictor import compute_outlook
    outlook = compute_outlook(target)

    article = generate_article(target, news, cfg, index_summary, macro_summary=macro_summary, outlook=outlook)

    print(f"\n{'='*70}")
    print(f"제목: {article.title}")
    print(f"{'='*70}\n")
    print(article.body)
    if article.warnings:
        print(f"\n⚠️ 경고: {article.warnings}")
