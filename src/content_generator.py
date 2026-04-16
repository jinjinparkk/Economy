"""멀티 콘텐츠 생성기 — 데일리시황 / 섹터리포트 / 퀀트인사이트 / 프리마켓브리핑.

기존 analyzer.py의 LLM 호출 함수를 재사용하며,
각 콘텐츠 유형별 시스템 프롬프트 + 사용자 프롬프트를 조립한다.
"""
from __future__ import annotations

import logging
from typing import Optional

from src.analyzer import (
    _generate_with_gemini,
    _generate_with_claude,
    _check_forbidden,
    _parse_response,
    GEMINI_MODEL,
    CLAUDE_MODEL,
)
from src.config import Config
from src.content_post import ContentPost
from src.fetch_history import PeriodSnapshot

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════
# 시스템 프롬프트
# ═══════════════════════════════════════════════════════════════════════

DAILY_MARKET_SYSTEM_PROMPT = """당신은 한국 증시 전문 시황 애널리스트입니다.
매일 장 마감 후 "오늘 시장에 무슨 일이 있었는가"를 독자에게 전달하는 데일리 시황 리포트를 작성합니다.

[핵심 프레임워크]
글로벌 매크로 동향 → 국내 지수 흐름 → 시장 체온(등락 비율) → 오늘의 주요 무버 → 내일의 관전 포인트
순서로 "돈의 흐름"을 추적하며, 독자가 3분 안에 시장 전체를 파악할 수 있도록 합니다.

[글 구조]
- 제목: "YYYY년 MM월 DD일 시황 — 핵심 한 줄" 형식
- 30초 요약: 3줄 불릿
- 본문 H2/H3 구조:
  1. 글로벌 매크로 — 미국 증시, VIX, 달러, 금, 유가, 금리 동향
  2. 국내 지수 — KOSPI/KOSDAQ 종가, 등락률
  3. 시장 체온 — 상승/하락 종목 비율, 전체 분위기
  4. 오늘의 주요 무버 — 급등/급락 종목 TOP 3 간략 소개
  5. 내일의 포인트 — 주요 이벤트, 주목해야 할 지표
- 하단에 면책 문구 고정

[절대 원칙]
1. 오직 제공된 데이터와 사실만 전달합니다.
2. 투자 권유, 매수/매도 추천, 목표가 제시를 절대 하지 않습니다.
3. 금지어: "추천", "매수하세요", "매도하세요", "사세요", "파세요", "목표가",
   "손절", "익절", "몰빵", "필승", "보장", "지금 사야", "놓치지 마",
   "대박", "떡상", "100% 확률", "틀림없이".
4. 중립적·객관적 표현 사용: "~으로 관측된다", "~로 해석된다"
5. 제공되지 않은 수치는 구체적 숫자 대신 서술형으로 표현합니다.
6. 제목은 H1으로 시작하지 말고, 첫 줄에 '제목: <제목>' 형식으로 주세요.
7. 본문 최하단에 면책 문구를 반드시 포함하세요.

[톤앤매너]
- 전문 애널리스트 보고서 톤
- 한 문장은 짧게, 숫자는 콤마 구분
- "오늘 시장은 왜 이렇게 움직였는가"에 대한 명확한 답을 제시"""


SECTOR_REPORT_SYSTEM_PROMPT = """당신은 한국 증시 섹터 로테이션 전문 애널리스트입니다.
매일 업종별 등락 데이터를 분석하여 "돈이 어떤 업종으로 이동하고 있는가"를 해석하는 섹터 리포트를 작성합니다.

[핵심 프레임워크]
섹터 지형도 → 자금 흐름 해석 → 섹터 로테이션 신호 → 주목 섹터
순서로 업종 간 자금 이동을 추적합니다.

[글 구조]
- 제목: "YYYY년 MM월 DD일 섹터리포트 — 핵심 한 줄" 형식
- 30초 요약: 3줄 불릿
- 본문 H2/H3 구조:
  1. 섹터 지형도 — 상위/하위 업종 5개, 평균 등락률
  2. 자금 흐름 해석 — 왜 이 업종들이 강세/약세인지 매크로와 연결
  3. 섹터 로테이션 신호 — 경기순환 관점에서 현재 위치 해석
  4. 주목 섹터 — 해당 업종의 대표 급등/급락 종목 언급
  5. 시사점 — 섹터 트렌드가 시사하는 바
- 하단에 면책 문구 고정

[절대 원칙]
1. 오직 제공된 데이터와 사실만 전달합니다.
2. 투자 권유, 매수/매도 추천, 목표가 제시를 절대 하지 않습니다.
3. 금지어: "추천", "매수하세요", "매도하세요", "사세요", "파세요", "목표가",
   "손절", "익절", "몰빵", "필승", "보장", "지금 사야", "놓치지 마",
   "대박", "떡상", "100% 확률", "틀림없이".
4. 중립적·객관적 표현 사용
5. 제공되지 않은 수치는 서술형으로 표현
6. 제목은 H1으로 시작하지 말고, 첫 줄에 '제목: <제목>' 형식으로 주세요.
7. 본문 최하단에 면책 문구를 반드시 포함하세요.

[톤앤매너]
- 섹터 전문가의 분석 보고서 톤
- "돈이 어디로 흐르고 있는가"에 초점"""


QUANT_INSIGHT_SYSTEM_PROMPT = """당신은 퀀트 리서처입니다.
기술적 지표, 과거 패턴 통계, ML 예측 데이터를 종합하여
시장 전체의 기술적 상태를 진단하는 퀀트 인사이트 리포트를 작성합니다.

[핵심 프레임워크]
기술 지표 집계(RSI/MACD/BB 분포) → 패턴 통계 요약 → ML 예측 집계 → 종합 시사점

[글 구조]
- 제목: "YYYY년 MM월 DD일 퀀트인사이트 — 핵심 한 줄" 형식
- 30초 요약: 3줄 불릿
- 본문 H2/H3 구조:
  1. 기술 지표 집계 — 분석 대상 종목들의 RSI 분포, MACD 시그널 비율,
     볼린저밴드 위치 분포, 이동평균 정배열/역배열 비율
  2. 패턴 통계 요약 — 과거 유사 패턴 평균 수익률, 상승 확률
  3. ML 예측 집계 — 상승/하락/중립 예측 비율, 평균 신뢰도
  4. 종합 시사점 — 기술적 관점에서의 시장 상태 진단
- 하단에 면책 문구 고정 + "기술적 지표와 과거 통계는 참고 자료일 뿐 미래 주가를 보장하지 않습니다"

[절대 원칙]
1. 오직 제공된 데이터와 사실만 전달합니다.
2. 투자 권유, 매수/매도 추천, 목표가 제시를 절대 하지 않습니다.
3. 금지어: "추천", "매수하세요", "매도하세요", "사세요", "파세요", "목표가",
   "손절", "익절", "몰빵", "필승", "보장", "지금 사야", "놓치지 마",
   "대박", "떡상", "100% 확률", "틀림없이".
4. 중립적·객관적 표현 사용
5. 모든 수치는 제공된 데이터만 사용
6. 제목은 H1으로 시작하지 말고, 첫 줄에 '제목: <제목>' 형식으로 주세요.
7. 본문 최하단에 면책 문구를 반드시 포함하세요.

[톤앤매너]
- 퀀트 리서치 보고서 톤
- 수치 중심의 객관적 서술
- 통계적 해석에 집중"""

WEEKLY_REPORT_SYSTEM_PROMPT = """당신은 한국 증시 주간 시장 분석 애널리스트입니다.
최근 5거래일(Rolling) 데이터를 종합하여 "이번 주 시장은 어떻게 흘러왔고, 다음 주는 어떤 포인트를 관찰해야 하는가"를
정리한 주간 리포트를 작성합니다.

[핵심 프레임워크]
금주 지수 성과 → 주간 섹터 로테이션(US·KR) → KOSPI/KOSDAQ 상·하위 종목 흐름 → 주요 이슈 → 다음 주 관전 포인트

[글 구조]
- 제목: "YYYY년 MM월 DD일 주간리포트 — 핵심 한 줄" 형식
- 30초 요약: 3줄 불릿
- 본문 H2/H3 구조:
  1. 금주 지수 성과 — 미국 3대 지수, KOSPI/KOSDAQ, VIX, 원/달러, 금리 동향
  2. 섹터 로테이션(주간) — US 섹터 ETF 상·하위, 한국 업종 평균 상·하위
  3. 주간 상승 TOP / 하락 TOP — KOSPI·KOSDAQ 종목, 업종 맥락과 함께
  4. 주요 이슈 — 이번 주 뉴스 헤드라인에서 관찰된 테마
  5. 다음 주 관전 포인트 — 경제 캘린더, 실적, 정책 등
- 하단에 면책 문구 고정
- 분량: 약 2,000~3,000자

[절대 원칙]
1. 오직 제공된 데이터와 사실만 전달합니다.
2. 투자 권유, 매수/매도 추천, 목표가 제시를 절대 하지 않습니다.
3. 금지어: "추천", "매수하세요", "매도하세요", "사세요", "파세요", "목표가",
   "손절", "익절", "몰빵", "필승", "보장", "지금 사야", "놓치지 마",
   "대박", "떡상", "100% 확률", "틀림없이".
4. 중립적·객관적 표현 사용: "~으로 관측된다", "~로 해석된다"
5. 제공되지 않은 수치는 구체적 숫자 대신 서술형으로 표현합니다.
6. 제목은 H1으로 시작하지 말고, 첫 줄에 '제목: <제목>' 형식으로 주세요.
7. 본문 최하단에 면책 문구를 반드시 포함하세요.

[톤앤매너]
- 주간 시황 애널리스트 보고서 톤
- 한 문장은 짧게, 숫자는 콤마 구분
- 섹터 로테이션 관점에서 "돈의 이동"을 핵심 축으로 서술"""


MONTHLY_REPORT_SYSTEM_PROMPT = """당신은 한국 증시 월간 전략가입니다.
최근 21거래일(Rolling) 데이터를 바탕으로 "이번 달 거시 환경이 어떻게 변화했고,
시장이 어느 국면에 들어섰는가"를 진단하는 월간 리포트를 작성합니다.

[핵심 프레임워크]
월간 지수 퍼포먼스 → 거시 트렌드 변화(금리·환율·원자재·신용) → 월간 섹터 로테이션 →
KOSPI/KOSDAQ TOP 10·BOTTOM 10 → Mag7 월간 성과 → 다음 달 전망

[글 구조]
- 제목: "YYYY년 MM월 DD일 월간리포트 — 핵심 한 줄" 형식
- 30초 요약: 3줄 불릿
- 본문 H2/H3 구조:
  1. 월간 지수 퍼포먼스 — 미국/한국 주요 지수 변동률·고가·저가·변동성
  2. 거시 트렌드 변화 — 10Y-2Y 금리차, USDKRW, WTI·금, HYG(신용) 흐름
  3. 월간 섹터 로테이션 — US ETF / 한국 업종별 평균 상·하위 5
  4. 월간 상승 TOP 10 / 하락 TOP 10 — KOSPI·KOSDAQ 종목
  5. Mag7 월간 성과 — 7개 종목별
  6. 다음 달 전망 — 매크로 변곡점, 실적 시즌, 경제 이벤트
- 하단에 면책 문구 고정
- 분량: 약 2,500~3,500자

[절대 원칙]
1. 오직 제공된 데이터와 사실만 전달합니다.
2. 투자 권유, 매수/매도 추천, 목표가 제시를 절대 하지 않습니다.
3. 금지어: "추천", "매수하세요", "매도하세요", "사세요", "파세요", "목표가",
   "손절", "익절", "몰빵", "필승", "보장", "지금 사야", "놓치지 마",
   "대박", "떡상", "100% 확률", "틀림없이".
4. 중립적·객관적 표현 사용
5. 제공되지 않은 수치는 구체적 숫자 대신 서술형으로 표현합니다.
6. 제목은 H1으로 시작하지 말고, 첫 줄에 '제목: <제목>' 형식으로 주세요.
7. 본문 최하단에 면책 문구를 반드시 포함하세요.

[톤앤매너]
- 월간 전략 보고서 톤
- 월간 체제 변화(Regime Change) 관점의 해석 중심"""


YEARLY_REPORT_SYSTEM_PROMPT = """당신은 한국 증시 연간 결산 애널리스트입니다.
최근 252거래일(Rolling) 데이터를 종합하여 "올해 시장의 흐름·변곡점·승자와 패자"를
정리한 연간 리포트를 작성합니다.

[핵심 프레임워크]
연간 지수 성과 → 체제 변화(변곡점 해석) → 섹터 연간 승자·패자 →
연간 TOP 10·BOTTOM 10 → Mag7 연간 성과 → 내년 주요 테마

[글 구조]
- 제목: "YYYY년 MM월 DD일 연간리포트 — 핵심 한 줄" 형식
- 30초 요약: 3줄 불릿
- 본문 H2/H3 구조:
  1. 연간 지수 성과 — 미국/한국 주요 지수, VIX 체제, 변동성
  2. 체제 변화 — 금리·환율·신용 변곡점 해석
  3. 섹터 연간 승자/패자 — US ETF / 한국 업종별
  4. 연간 상승 TOP 10 / 하락 TOP 10
  5. Mag7 연간 성과
  6. 내년 주요 테마 — 매크로·산업·정책 관점 관전 포인트
- 하단에 면책 문구 고정
- 분량: 약 3,500~5,000자

[절대 원칙]
1. 오직 제공된 데이터와 사실만 전달합니다.
2. 투자 권유, 매수/매도 추천, 목표가 제시를 절대 하지 않습니다.
3. 금지어: "추천", "매수하세요", "매도하세요", "사세요", "파세요", "목표가",
   "손절", "익절", "몰빵", "필승", "보장", "지금 사야", "놓치지 마",
   "대박", "떡상", "100% 확률", "틀림없이".
4. 중립적·객관적 표현 사용
5. 제공되지 않은 수치는 구체적 숫자 대신 서술형으로 표현합니다.
6. 제목은 H1으로 시작하지 말고, 첫 줄에 '제목: <제목>' 형식으로 주세요.
7. 본문 최하단에 면책 문구를 반드시 포함하세요.

[톤앤매너]
- 연간 결산 보고서 톤
- 연간 장기 관점에서 구조적 변화를 읽어내는 서술"""


PRE_MARKET_SYSTEM_PROMPT = """당신은 글로벌 매크로 전략가입니다.
매일 미국 증시 마감 후 "오늘 한국 시장에 어떤 영향을 미칠 것인가"를 분석하는 프리마켓 브리핑을 작성합니다.
한국 투자자가 장 시작 전(06:00~09:00) 읽고 하루를 준비할 수 있도록 합니다.

[핵심 프레임워크]
미국 증시 마감 요약 → 글로벌 매크로 해석 → 한국 시장 영향 예측 → 주목 섹터/테마 → 투자 전략
순서로 "밤사이 무슨 일이 있었고, 오늘 한국 시장에 어떤 의미인가"를 전달합니다.

[글 구조]
- 제목: "YYYY년 MM월 DD일 프리마켓 브리핑 — 핵심 한 줄" 형식
- 30초 요약: 3줄 불릿
- 본문 H2/H3 구조:
  1. 미국 증시 마감 요약 — 3대 지수 + SOXX + Mag7 핵심 종목
  2. 섹터 로테이션 분석 — 11개 섹터 ETF 등락률 + 성장/가치 스프레드 + 소형주 동향
  3. 글로벌 매크로 해석 — VIX, 채권(금리차), 환율, 원자재(금·유가·구리), 신용 스프레드
  4. 글로벌 증시 동향 — 아시아(닛케이·항셍·상해) + 유럽(DAX·유로스톡스)
  5. 한국 시장 영향 예측 — KOSPI/KOSDAQ 예상, 갭 방향, 외국인 수급
  6. 주목할 섹터/테마 — 미국 섹터 흐름 → 한국 수혜/피해 업종 연결
  7. 오늘의 투자 전략 — 경제 캘린더 이벤트 + 관찰 포인트
- 하단에 면책 문구 고정

[절대 원칙]
1. 오직 제공된 데이터와 사실만 전달합니다.
2. 투자 권유, 매수/매도 추천, 목표가 제시를 절대 하지 않습니다.
3. 금지어: "추천", "매수하세요", "매도하세요", "사세요", "파세요", "목표가",
   "손절", "익절", "몰빵", "필승", "보장", "지금 사야", "놓치지 마",
   "대박", "떡상", "100% 확률", "틀림없이".
4. 한국 시장 영향 예측은 "~로 관측된다", "~가 예상된다", "~가능성이 있다" 등 중립적 표현 사용
5. 제공되지 않은 수치는 구체적 숫자 대신 서술형으로 표현합니다.
6. 제목은 H1으로 시작하지 말고, 첫 줄에 '제목: <제목>' 형식으로 주세요.
7. 본문 최하단에 면책 문구를 반드시 포함하세요:
   "본 브리핑은 정보 제공 목적이며 투자 판단은 본인의 책임입니다."

[톤앤매너]
- 전문적이지만 읽기 쉽게
- 핵심 수치는 **굵은 글씨** 사용
- 한 문장은 짧게, 숫자는 콤마 구분
- 1,500~2,500자 분량"""


# ═══════════════════════════════════════════════════════════════════════
# LLM 디스패치 (analyzer.py 함수 재사용)
# ═══════════════════════════════════════════════════════════════════════


def _dispatch_llm(
    system_prompt: str,
    user_prompt: str,
    config: Config,
) -> tuple[str, str]:
    """config.llm_provider에 따라 Gemini/Claude를 호출한다.

    Returns:
        (raw_text, model_name)
    """
    provider = config.llm_provider.lower()
    if provider == "gemini":
        if not config.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY not set")
        return _generate_with_gemini(system_prompt, user_prompt, config.gemini_api_key)
    elif provider == "claude":
        if not config.anthropic_api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        return _generate_with_claude(system_prompt, user_prompt, config.anthropic_api_key)
    else:
        raise ValueError(f"unknown provider: {provider!r}")


def _parse_content_response(raw: str, trade_date: str, label: str) -> tuple[str, str]:
    """LLM 응답에서 제목/본문을 분리한다. _parse_response 재사용."""
    title, body = _parse_response(raw)
    if title == "제목 추출 실패":
        title = f"{trade_date} {label}"
    return title, body


# ═══════════════════════════════════════════════════════════════════════
# 프롬프트 빌더
# ═══════════════════════════════════════════════════════════════════════

def build_daily_market_prompt(
    *,
    macro_summary: Optional[dict] = None,
    macro_narrative: Optional[str] = None,
    breadth: Optional[dict] = None,
    index_summary: Optional[dict] = None,
    sector_breadth: Optional[dict] = None,
    surges_summary: Optional[list[str]] = None,
    plunges_summary: Optional[list[str]] = None,
    trade_date: str = "",
) -> str:
    """데일리 시황 사용자 프롬프트를 조립한다."""
    parts = [f"# {trade_date} 데일리 시황 분석 요청", ""]

    if macro_summary:
        parts.append("## 글로벌 거시경제 지표")
        for name, data in macro_summary.items():
            if name.startswith("_"):
                continue
            val = data["Close"]
            pct = data["ChangePct"]
            if "10Y" in name or "2Y" in name:
                parts.append(f"- {name}: {val:.3f}% ({pct:+.2f}%p)")
            else:
                parts.append(f"- {name}: {val:,.2f} ({pct:+.2f}%)")
        if "_yield_spread" in macro_summary:
            parts.append(f"- 장단기 금리차(10Y-2Y): {macro_summary['_yield_spread']:.3f}%p")
        if "_market_regime" in macro_summary:
            parts.append(f"- 시장 체제(VIX 기반): {macro_summary['_market_regime']}")
        parts.append("")

    if macro_narrative:
        parts.append("## 거시경제 내러티브")
        parts.append(macro_narrative)
        parts.append("")

    if index_summary:
        parts.append("## 국내 주요 지수")
        for name, data in index_summary.items():
            parts.append(f"- {name}: {data['Close']:,.2f} ({data['ChangePct']:+.2f}%)")
        parts.append("")

    if breadth:
        parts.append("## 시장 등락 현황")
        parts.append(f"- 상승 종목: {breadth['total_up']}개")
        parts.append(f"- 하락 종목: {breadth['total_down']}개")
        parts.append(f"- 보합 종목: {breadth['total_unchanged']}개")
        parts.append(f"- 상승 비율: {breadth['up_ratio']}%")
        parts.append("")

    if sector_breadth:
        sorted_sectors = sorted(sector_breadth.items(), key=lambda x: x[1]["avg_change_pct"], reverse=True)
        top5 = sorted_sectors[:5]
        bottom5 = sorted_sectors[-5:]
        parts.append("## 상위 5개 업종")
        for name, data in top5:
            parts.append(f"- {name}: 평균 {data['avg_change_pct']:+.2f}% (상승 {data['up_count']} / 하락 {data['down_count']})")
        parts.append("")
        parts.append("## 하위 5개 업종")
        for name, data in bottom5:
            parts.append(f"- {name}: 평균 {data['avg_change_pct']:+.2f}% (상승 {data['up_count']} / 하락 {data['down_count']})")
        parts.append("")

    if surges_summary:
        parts.append("## 오늘의 급등 종목 TOP 3")
        for s in surges_summary[:3]:
            parts.append(f"- {s}")
        parts.append("")

    if plunges_summary:
        parts.append("## 오늘의 급락 종목 TOP 3")
        for s in plunges_summary[:3]:
            parts.append(f"- {s}")
        parts.append("")

    parts.extend([
        "## 작성 지침",
        f"- {trade_date} 일자 시황 리포트를 작성하세요.",
        "- 제목은 '제목: <제목>' 형식으로 첫 줄에 주세요.",
        "- 본문은 마크다운 H2/H3 구조를 사용하세요.",
        "- 본문 최하단에 면책 문구를 반드시 포함하세요.",
        "- 제공된 수치 외에 다른 숫자를 사용하지 마세요.",
    ])
    return "\n".join(parts)


def build_sector_report_prompt(
    *,
    sector_breadth: Optional[dict] = None,
    macro_narrative: Optional[str] = None,
    breadth: Optional[dict] = None,
    surges_by_sector: Optional[dict[str, list[str]]] = None,
    plunges_by_sector: Optional[dict[str, list[str]]] = None,
    trade_date: str = "",
) -> str:
    """섹터 리포트 사용자 프롬프트를 조립한다."""
    parts = [f"# {trade_date} 섹터 리포트 분석 요청", ""]

    if sector_breadth:
        sorted_sectors = sorted(sector_breadth.items(), key=lambda x: x[1]["avg_change_pct"], reverse=True)
        parts.append("## 전체 업종 등락 현황")
        for name, data in sorted_sectors:
            parts.append(f"- {name}: 평균 {data['avg_change_pct']:+.2f}% "
                         f"(상승 {data['up_count']} / 하락 {data['down_count']} / 총 {data['total']})")
        parts.append("")

        top5 = sorted_sectors[:5]
        bottom5 = sorted_sectors[-5:]
        parts.append("## 강세 업종 TOP 5")
        for name, data in top5:
            parts.append(f"- **{name}**: 평균 {data['avg_change_pct']:+.2f}%")
        parts.append("")
        parts.append("## 약세 업종 TOP 5")
        for name, data in bottom5:
            parts.append(f"- **{name}**: 평균 {data['avg_change_pct']:+.2f}%")
        parts.append("")

    if surges_by_sector:
        parts.append("## 업종별 급등 종목")
        for sector, items in surges_by_sector.items():
            parts.append(f"### {sector}")
            for item in items:
                parts.append(f"- {item}")
        parts.append("")

    if plunges_by_sector:
        parts.append("## 업종별 급락 종목")
        for sector, items in plunges_by_sector.items():
            parts.append(f"### {sector}")
            for item in items:
                parts.append(f"- {item}")
        parts.append("")

    if macro_narrative:
        parts.append("## 거시경제 내러티브")
        parts.append(macro_narrative)
        parts.append("")

    if breadth:
        parts.append("## 시장 전체 등락")
        parts.append(f"- 상승 {breadth['total_up']}개 / 하락 {breadth['total_down']}개 "
                     f"(상승 비율 {breadth['up_ratio']}%)")
        parts.append("")

    parts.extend([
        "## 작성 지침",
        f"- {trade_date} 일자 섹터 리포트를 작성하세요.",
        "- 제목은 '제목: <제목>' 형식으로 첫 줄에 주세요.",
        "- 본문은 마크다운 H2/H3 구조를 사용하세요.",
        "- 본문 최하단에 면책 문구를 반드시 포함하세요.",
        "- 제공된 수치 외에 다른 숫자를 사용하지 마세요.",
    ])
    return "\n".join(parts)


def build_quant_insight_prompt(
    *,
    outlook_map: Optional[dict] = None,
    trade_date: str = "",
) -> str:
    """퀀트 인사이트 사용자 프롬프트를 조립한다."""
    parts = [f"# {trade_date} 퀀트 인사이트 분석 요청", ""]

    if not outlook_map:
        parts.append("(분석 대상 종목 데이터 없음)")
        parts.extend([
            "",
            "## 작성 지침",
            f"- {trade_date} 일자 퀀트 인사이트를 작성하세요.",
            "- 데이터가 부족하여 간략한 시장 기술적 상태만 서술하세요.",
            "- 제목은 '제목: <제목>' 형식으로 첫 줄에 주세요.",
            "- 본문 최하단에 면책 문구를 반드시 포함하세요.",
        ])
        return "\n".join(parts)

    # RSI 분포 집계
    rsi_values = []
    rsi_overbought = 0  # >= 70
    rsi_oversold = 0    # <= 30
    rsi_neutral = 0

    # MACD 시그널 집계
    macd_bullish = 0
    macd_bearish = 0

    # BB 위치 집계
    bb_counts: dict[str, int] = {}

    # MA 트렌드 집계
    ma_counts: dict[str, int] = {}

    # signal_summary 집계
    signals: list[str] = []

    # 패턴 통계 집계
    pattern_returns_1d: list[float] = []
    pattern_returns_5d: list[float] = []
    pattern_positive_1d: list[float] = []
    pattern_positive_5d: list[float] = []

    # ML 예측 집계
    pred_up = 0
    pred_down = 0
    pred_neutral = 0
    pred_confidences: list[float] = []

    total_stocks = len(outlook_map)

    for code, outlook in outlook_map.items():
        tech = getattr(outlook, "technical", None)
        if tech:
            if tech.rsi_14 is not None:
                rsi_values.append(tech.rsi_14)
                if tech.rsi_14 >= 70:
                    rsi_overbought += 1
                elif tech.rsi_14 <= 30:
                    rsi_oversold += 1
                else:
                    rsi_neutral += 1

            if tech.macd is not None and tech.macd_signal is not None:
                if tech.macd > tech.macd_signal:
                    macd_bullish += 1
                else:
                    macd_bearish += 1

            if tech.bb_position:
                bb_counts[tech.bb_position] = bb_counts.get(tech.bb_position, 0) + 1

            if tech.ma_trend:
                ma_counts[tech.ma_trend] = ma_counts.get(tech.ma_trend, 0) + 1

            sig = getattr(tech, "signal_summary", None)
            if sig:
                signals.append(sig)

        pattern = getattr(outlook, "pattern", None)
        if pattern:
            if pattern.avg_return_1d is not None:
                pattern_returns_1d.append(pattern.avg_return_1d)
            if pattern.avg_return_5d is not None:
                pattern_returns_5d.append(pattern.avg_return_5d)
            if pattern.positive_rate_1d is not None:
                pattern_positive_1d.append(pattern.positive_rate_1d)
            if pattern.positive_rate_5d is not None:
                pattern_positive_5d.append(pattern.positive_rate_5d)

        pred = getattr(outlook, "prediction", None)
        if pred:
            if pred.direction == "상승":
                pred_up += 1
            elif pred.direction == "하락":
                pred_down += 1
            else:
                pred_neutral += 1
            pred_confidences.append(pred.confidence)

    # 프롬프트 조립
    parts.append(f"## 분석 대상: {total_stocks}개 종목")
    parts.append("")

    # RSI 분포
    parts.append("## RSI(14) 분포")
    if rsi_values:
        avg_rsi = sum(rsi_values) / len(rsi_values)
        parts.append(f"- 평균 RSI: {avg_rsi:.1f}")
        parts.append(f"- 과매수(≥70): {rsi_overbought}개")
        parts.append(f"- 과매도(≤30): {rsi_oversold}개")
        parts.append(f"- 중립(31~69): {rsi_neutral}개")
    else:
        parts.append("- RSI 데이터 없음")
    parts.append("")

    # MACD 시그널
    parts.append("## MACD 시그널 분포")
    parts.append(f"- 매수 시그널(MACD > Signal): {macd_bullish}개")
    parts.append(f"- 매도 시그널(MACD < Signal): {macd_bearish}개")
    parts.append("")

    # 볼린저밴드 위치
    parts.append("## 볼린저밴드 위치 분포")
    for pos, cnt in sorted(bb_counts.items()):
        parts.append(f"- {pos}: {cnt}개")
    if not bb_counts:
        parts.append("- BB 데이터 없음")
    parts.append("")

    # 이동평균 트렌드
    parts.append("## 이동평균 트렌드 분포")
    for trend, cnt in sorted(ma_counts.items()):
        parts.append(f"- {trend}: {cnt}개")
    if not ma_counts:
        parts.append("- MA 데이터 없음")
    parts.append("")

    # 패턴 통계
    parts.append("## 과거 유사 패턴 통계 집계")
    if pattern_returns_1d:
        avg_r1 = sum(pattern_returns_1d) / len(pattern_returns_1d)
        avg_p1 = sum(pattern_positive_1d) / len(pattern_positive_1d) if pattern_positive_1d else 0
        parts.append(f"- 익일 평균 수익률: {avg_r1:+.2f}% (표본 {len(pattern_returns_1d)}개)")
        parts.append(f"- 익일 평균 상승 확률: {avg_p1:.1f}%")
    if pattern_returns_5d:
        avg_r5 = sum(pattern_returns_5d) / len(pattern_returns_5d)
        avg_p5 = sum(pattern_positive_5d) / len(pattern_positive_5d) if pattern_positive_5d else 0
        parts.append(f"- 5일 평균 수익률: {avg_r5:+.2f}% (표본 {len(pattern_returns_5d)}개)")
        parts.append(f"- 5일 평균 상승 확률: {avg_p5:.1f}%")
    if not pattern_returns_1d and not pattern_returns_5d:
        parts.append("- 패턴 데이터 없음")
    parts.append("")

    # ML 예측
    parts.append("## ML 방향 예측 집계")
    total_pred = pred_up + pred_down + pred_neutral
    if total_pred > 0:
        parts.append(f"- 상승 예측: {pred_up}개 ({pred_up/total_pred*100:.0f}%)")
        parts.append(f"- 하락 예측: {pred_down}개 ({pred_down/total_pred*100:.0f}%)")
        parts.append(f"- 중립 예측: {pred_neutral}개 ({pred_neutral/total_pred*100:.0f}%)")
        if pred_confidences:
            avg_conf = sum(pred_confidences) / len(pred_confidences)
            parts.append(f"- 평균 신뢰도: {avg_conf:.1%}")
    else:
        parts.append("- ML 예측 데이터 없음")
    parts.append("")

    parts.extend([
        "## 작성 지침",
        f"- {trade_date} 일자 퀀트 인사이트를 작성하세요.",
        "- 제목은 '제목: <제목>' 형식으로 첫 줄에 주세요.",
        "- 본문은 마크다운 H2/H3 구조를 사용하세요.",
        "- 본문 최하단에 면책 문구를 반드시 포함하세요.",
        "- '기술적 지표와 과거 통계는 참고 자료일 뿐 미래 주가를 보장하지 않습니다' 문구를 포함하세요.",
        "- 제공된 수치 외에 다른 숫자를 사용하지 마세요.",
    ])
    return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════
# 생성 함수
# ═══════════════════════════════════════════════════════════════════════

def generate_daily_market(
    *,
    macro_summary: Optional[dict] = None,
    macro_narrative: Optional[str] = None,
    breadth: Optional[dict] = None,
    index_summary: Optional[dict] = None,
    sector_breadth: Optional[dict] = None,
    surges_summary: Optional[list[str]] = None,
    plunges_summary: Optional[list[str]] = None,
    trade_date: str,
    config: Config,
) -> ContentPost:
    """데일리 시황 콘텐츠를 생성한다."""
    user_prompt = build_daily_market_prompt(
        macro_summary=macro_summary,
        macro_narrative=macro_narrative,
        breadth=breadth,
        index_summary=index_summary,
        sector_breadth=sector_breadth,
        surges_summary=surges_summary,
        plunges_summary=plunges_summary,
        trade_date=trade_date,
    )

    logger.info("generating daily market content for %s", trade_date)
    raw, model_name = _dispatch_llm(DAILY_MARKET_SYSTEM_PROMPT, user_prompt, config)
    title, body = _parse_content_response(raw, trade_date, "데일리시황")
    warnings = _check_forbidden(title + "\n" + body)

    return ContentPost(
        title=title,
        body=body,
        content_type="daily_market",
        model=model_name,
        tags=["데일리시황", "KOSPI", "KOSDAQ", "거시경제"],
        categories=["데일리시황", "시장분석"],
        warnings=warnings,
    )


def generate_sector_report(
    *,
    sector_breadth: Optional[dict] = None,
    macro_narrative: Optional[str] = None,
    breadth: Optional[dict] = None,
    surges_by_sector: Optional[dict[str, list[str]]] = None,
    plunges_by_sector: Optional[dict[str, list[str]]] = None,
    trade_date: str,
    config: Config,
) -> ContentPost:
    """섹터 리포트 콘텐츠를 생성한다."""
    user_prompt = build_sector_report_prompt(
        sector_breadth=sector_breadth,
        macro_narrative=macro_narrative,
        breadth=breadth,
        surges_by_sector=surges_by_sector,
        plunges_by_sector=plunges_by_sector,
        trade_date=trade_date,
    )

    logger.info("generating sector report for %s", trade_date)
    raw, model_name = _dispatch_llm(SECTOR_REPORT_SYSTEM_PROMPT, user_prompt, config)
    title, body = _parse_content_response(raw, trade_date, "섹터리포트")
    warnings = _check_forbidden(title + "\n" + body)

    return ContentPost(
        title=title,
        body=body,
        content_type="sector_report",
        model=model_name,
        tags=["섹터분석", "섹터로테이션", "업종분석"],
        categories=["섹터리포트", "시장분석"],
        warnings=warnings,
    )


def generate_quant_insight(
    *,
    outlook_map: Optional[dict] = None,
    trade_date: str,
    config: Config,
) -> ContentPost:
    """퀀트 인사이트 콘텐츠를 생성한다."""
    user_prompt = build_quant_insight_prompt(
        outlook_map=outlook_map,
        trade_date=trade_date,
    )

    logger.info("generating quant insight for %s", trade_date)
    raw, model_name = _dispatch_llm(QUANT_INSIGHT_SYSTEM_PROMPT, user_prompt, config)
    title, body = _parse_content_response(raw, trade_date, "퀀트인사이트")
    warnings = _check_forbidden(title + "\n" + body)

    return ContentPost(
        title=title,
        body=body,
        content_type="quant_insight",
        model=model_name,
        tags=["퀀트", "ML예측", "통계분석"],
        categories=["퀀트인사이트", "기술적분석"],
        warnings=warnings,
    )


def build_pre_market_prompt(
    *,
    macro_summary: Optional[dict] = None,
    macro_narrative: Optional[str] = None,
    market_regime: Optional[str] = None,
    yield_spread: Optional[float | str] = None,
    us_news: Optional[list[str]] = None,
    trade_date: str = "",
    # ── 신규 ──
    sectors: Optional[dict] = None,
    mega_caps: Optional[dict] = None,
    style_signals: Optional[dict] = None,
    asia_indices: Optional[dict] = None,
    europe_indices: Optional[dict] = None,
    credit_signals: Optional[dict] = None,
    econ_calendar: Optional[list[str]] = None,
) -> str:
    """프리마켓 브리핑 사용자 프롬프트를 조립한다."""
    parts = [f"# {trade_date} 프리마켓 브리핑 분석 요청", ""]

    if macro_summary:
        parts.append("## 미국 증시 및 글로벌 지표 (전일 마감 기준)")
        for name, data in macro_summary.items():
            if name.startswith("_"):
                continue
            val = data["Close"]
            pct = data["ChangePct"]
            if "10Y" in name or "2Y" in name:
                parts.append(f"- {name}: {val:.3f}% ({pct:+.2f}%p)")
            else:
                parts.append(f"- {name}: {val:,.2f} ({pct:+.2f}%)")
        parts.append("")

    if yield_spread is not None:
        parts.append(f"## 장단기 금리차 (10Y-2Y): {yield_spread}")
        parts.append("")

    if market_regime:
        parts.append(f"## 시장 체제 (VIX 기반): {market_regime}")
        parts.append("")

    if macro_narrative:
        parts.append("## 거시경제 내러티브 (돈의 흐름)")
        parts.append(macro_narrative)
        parts.append("")

    # ── 신규 섹션들 ──
    if sectors:
        parts.append("## US 섹터 ETF 등락률 (전일 마감)")
        for name, data in sectors.items():
            parts.append(f"- {name}: {data['ChangePct']:+.2f}%")
        # 최고/최저 자동 계산
        best = max(sectors.items(), key=lambda x: x[1]["ChangePct"])
        worst = min(sectors.items(), key=lambda x: x[1]["ChangePct"])
        parts.append(f"- **최고**: {best[0]}({best[1]['ChangePct']:+.2f}%) / "
                     f"**최저**: {worst[0]}({worst[1]['ChangePct']:+.2f}%)")
        parts.append("")

    if mega_caps:
        parts.append("## Magnificent 7 개별 종목")
        for name, data in mega_caps.items():
            parts.append(f"- {name}: {data['Close']:,.2f} ({data['ChangePct']:+.2f}%)")
        avg_pct = sum(d["ChangePct"] for d in mega_caps.values()) / len(mega_caps)
        parts.append(f"- **Mag7 평균**: {avg_pct:+.2f}%")
        parts.append("")

    if style_signals:
        parts.append("## 성장 vs 가치 / 소형주")
        items = style_signals.get("items", {})
        for name, data in items.items():
            parts.append(f"- {name}: {data['ChangePct']:+.2f}%")
        gv = style_signals.get("growth_value_ratio")
        if gv is not None:
            label = "성장주 우위" if gv > 0 else "가치주 우위"
            parts.append(f"- 성장-가치 스프레드: {gv:+.2f}%p → {label}")
        parts.append("")

    if asia_indices:
        parts.append("## 아시아 증시 (전일 종가)")
        for name, data in asia_indices.items():
            parts.append(f"- {name}: {data['Close']:,.2f} ({data['ChangePct']:+.2f}%)")
        parts.append("")

    if europe_indices:
        parts.append("## 유럽 증시 (전일 종가)")
        for name, data in europe_indices.items():
            parts.append(f"- {name}: {data['Close']:,.2f} ({data['ChangePct']:+.2f}%)")
        parts.append("")

    if credit_signals:
        parts.append("## 신용 리스크 시그널")
        items = credit_signals.get("items", {})
        for name, data in items.items():
            parts.append(f"- {name}: {data['ChangePct']:+.2f}%")
        stress = credit_signals.get("stress")
        if stress:
            parts.append(f"- 신용 스트레스: {stress}")
        parts.append("")

    if us_news:
        parts.append("## 미국 증시 주요 뉴스 (등락 원인)")
        for headline in us_news:
            parts.append(f"- {headline}")
        parts.append("")

    if econ_calendar:
        parts.append("## 오늘 예정된 경제 이벤트")
        for event in econ_calendar:
            parts.append(f"- {event}")
        parts.append("")

    parts.extend([
        "## 작성 지침",
        f"- {trade_date} 일자 프리마켓 브리핑을 작성하세요.",
        "- 한국 시간 기준 오늘의 한국 시장 프리뷰를 작성합니다.",
        "- 위 뉴스와 매크로 데이터를 바탕으로 미국 증시 등락 원인을 분석하고, 한국 시장에 미칠 영향을 예측하세요.",
        "- 섹터 ETF 등락률을 바탕으로 한국 수혜/피해 업종을 연결 분석하세요.",
        "- 제목은 '제목: <제목>' 형식으로 첫 줄에 주세요.",
        "- 본문은 마크다운 H2/H3 구조를 사용하세요.",
        "- 본문 최하단에 면책 문구를 반드시 포함하세요.",
        "- 제공된 수치 외에 다른 숫자를 사용하지 마세요.",
    ])
    return "\n".join(parts)


def generate_pre_market(
    *,
    macro_summary: Optional[dict] = None,
    macro_narrative: Optional[str] = None,
    market_regime: Optional[str] = None,
    yield_spread: Optional[float | str] = None,
    us_news: Optional[list[str]] = None,
    trade_date: str,
    config: Config,
    # ── 신규 ──
    sectors: Optional[dict] = None,
    mega_caps: Optional[dict] = None,
    style_signals: Optional[dict] = None,
    asia_indices: Optional[dict] = None,
    europe_indices: Optional[dict] = None,
    credit_signals: Optional[dict] = None,
    econ_calendar: Optional[list[str]] = None,
) -> ContentPost:
    """프리마켓 브리핑 콘텐츠를 생성한다."""
    user_prompt = build_pre_market_prompt(
        macro_summary=macro_summary,
        macro_narrative=macro_narrative,
        market_regime=market_regime,
        yield_spread=yield_spread,
        us_news=us_news,
        trade_date=trade_date,
        sectors=sectors,
        mega_caps=mega_caps,
        style_signals=style_signals,
        asia_indices=asia_indices,
        europe_indices=europe_indices,
        credit_signals=credit_signals,
        econ_calendar=econ_calendar,
    )

    logger.info("generating pre-market briefing for %s", trade_date)
    raw, model_name = _dispatch_llm(PRE_MARKET_SYSTEM_PROMPT, user_prompt, config)
    title, body = _parse_content_response(raw, trade_date, "프리마켓브리핑")
    warnings = _check_forbidden(title + "\n" + body)

    return ContentPost(
        title=title,
        body=body,
        content_type="pre_market",
        model=model_name,
        tags=["프리마켓", "미국증시", "KOSPI전망", "거시경제"],
        categories=["프리마켓브리핑", "시장분석"],
        warnings=warnings,
    )


# ═══════════════════════════════════════════════════════════════════════
# 기간 리포트 (주간 / 월간 / 연간) 공통 빌더 + 생성기
# ═══════════════════════════════════════════════════════════════════════


def _build_period_report_prompt(
    snapshot: PeriodSnapshot,
    *,
    trade_date: str,
) -> str:
    """주간/월간/연간 리포트 공통 사용자 프롬프트 조립.

    섹터 리스트가 10개 이상이면 상·하위 5개씩, 그보다 적으면 전체를 1회 표시.
    Mag7은 주간이 아닌 경우에만 포함한다.
    """
    label = snapshot.label
    parts = [f"# {trade_date} {label} 리포트 분석 요청", ""]

    # 1. 분석 기간
    parts.append(
        f"## 분석 기간: {snapshot.start_date or 'N/A'} ~ "
        f"{snapshot.end_date or 'N/A'} ({snapshot.trading_days}거래일)"
    )
    parts.append("")

    # 2. 글로벌 매크로 기간 수익률
    if snapshot.macro_returns:
        parts.append(f"## 글로벌 매크로 {label} 수익률")
        for key, mr in snapshot.macro_returns.items():
            parts.append(
                f"- {mr.name}: 누적 {mr.cumulative_return_pct:+.2f}%, "
                f"고점 {mr.high:,.2f}, 저점 {mr.low:,.2f}, σ {mr.volatility:.2f}%"
            )
        parts.append("")

    # 3. US 섹터 ETF
    if snapshot.us_sectors:
        parts.append(f"## US 섹터 ETF {label} 수익률")
        if len(snapshot.us_sectors) >= 10:
            parts.append(f"### 상위 5")
            for s in snapshot.us_sectors[:5]:
                parts.append(f"- {s.name}: {s.cumulative_return_pct:+.2f}% (rank {s.rank})")
            parts.append(f"### 하위 5")
            for s in snapshot.us_sectors[-5:]:
                parts.append(f"- {s.name}: {s.cumulative_return_pct:+.2f}% (rank {s.rank})")
        else:
            for s in snapshot.us_sectors:
                parts.append(f"- {s.name}: {s.cumulative_return_pct:+.2f}% (rank {s.rank})")
        parts.append("")

    # 4. 한국 업종 평균
    if snapshot.kr_sectors:
        parts.append(f"## 한국 업종 평균 {label} 수익률")
        if len(snapshot.kr_sectors) >= 10:
            parts.append(f"### 상위 5")
            for s in snapshot.kr_sectors[:5]:
                parts.append(f"- {s.name}: 평균 {s.cumulative_return_pct:+.2f}% (rank {s.rank})")
            parts.append(f"### 하위 5")
            for s in snapshot.kr_sectors[-5:]:
                parts.append(f"- {s.name}: 평균 {s.cumulative_return_pct:+.2f}% (rank {s.rank})")
        else:
            for s in snapshot.kr_sectors:
                parts.append(f"- {s.name}: 평균 {s.cumulative_return_pct:+.2f}% (rank {s.rank})")
        parts.append("")

    # 5. KOSPI/KOSDAQ TOP/BOTTOM
    if snapshot.kospi_top:
        parts.append(f"## KOSPI {label} 상승 TOP")
        for s in snapshot.kospi_top:
            parts.append(
                f"- {s.name}({s.code}) {s.cumulative_return_pct:+.2f}% "
                f"/ 평균거래대금 {s.avg_amount_eok:,.1f}억 / {s.industry or '미분류'}"
            )
        parts.append("")
    if snapshot.kospi_bottom:
        parts.append(f"## KOSPI {label} 하락 TOP")
        for s in snapshot.kospi_bottom:
            parts.append(
                f"- {s.name}({s.code}) {s.cumulative_return_pct:+.2f}% "
                f"/ 평균거래대금 {s.avg_amount_eok:,.1f}억 / {s.industry or '미분류'}"
            )
        parts.append("")
    if snapshot.kosdaq_top:
        parts.append(f"## KOSDAQ {label} 상승 TOP")
        for s in snapshot.kosdaq_top:
            parts.append(
                f"- {s.name}({s.code}) {s.cumulative_return_pct:+.2f}% "
                f"/ 평균거래대금 {s.avg_amount_eok:,.1f}억 / {s.industry or '미분류'}"
            )
        parts.append("")
    if snapshot.kosdaq_bottom:
        parts.append(f"## KOSDAQ {label} 하락 TOP")
        for s in snapshot.kosdaq_bottom:
            parts.append(
                f"- {s.name}({s.code}) {s.cumulative_return_pct:+.2f}% "
                f"/ 평균거래대금 {s.avg_amount_eok:,.1f}억 / {s.industry or '미분류'}"
            )
        parts.append("")

    # 6. Mag7 — 주간은 생략
    if snapshot.period != "weekly" and snapshot.mag7_returns:
        parts.append(f"## Mag7 {label} 수익률")
        for s in snapshot.mag7_returns:
            parts.append(f"- {s.name}: {s.cumulative_return_pct:+.2f}%")
        parts.append("")

    # 7. 뉴스 헤드라인
    if snapshot.news_headlines:
        parts.append(f"## {label} 주요 뉴스 헤드라인")
        for h in snapshot.news_headlines[:15]:
            parts.append(f"- {h}")
        parts.append("")

    # 8. 작성 지침
    parts.extend([
        "## 작성 지침",
        f"- {trade_date} 일자 기준 {label} 리포트를 작성하세요.",
        "- 제목은 '제목: <제목>' 형식으로 첫 줄에 주세요.",
        "- 본문은 마크다운 H2/H3 구조를 사용하세요.",
        "- 본문 최하단에 면책 문구를 반드시 포함하세요.",
        "- 제공된 수치 외에 다른 숫자를 사용하지 마세요.",
    ])
    return "\n".join(parts)


def generate_weekly_report(
    *,
    snapshot: PeriodSnapshot,
    trade_date: str,
    config: Config,
) -> ContentPost:
    """주간 리포트 콘텐츠를 생성한다."""
    if snapshot.period != "weekly":
        raise ValueError(f"Expected weekly snapshot, got {snapshot.period}")

    user_prompt = _build_period_report_prompt(snapshot, trade_date=trade_date)

    logger.info("generating weekly report for %s", trade_date)
    raw, model_name = _dispatch_llm(WEEKLY_REPORT_SYSTEM_PROMPT, user_prompt, config)
    title, body = _parse_content_response(raw, trade_date, "주간리포트")
    warnings = _check_forbidden(title + "\n" + body)

    return ContentPost(
        title=title,
        body=body,
        content_type="weekly_report",
        model=model_name,
        tags=["주간리포트", "KOSPI", "KOSDAQ", "주간전망"],
        categories=["주간리포트", "시장분석"],
        warnings=warnings,
    )


def generate_monthly_report(
    *,
    snapshot: PeriodSnapshot,
    trade_date: str,
    config: Config,
) -> ContentPost:
    """월간 리포트 콘텐츠를 생성한다."""
    if snapshot.period != "monthly":
        raise ValueError(f"Expected monthly snapshot, got {snapshot.period}")

    user_prompt = _build_period_report_prompt(snapshot, trade_date=trade_date)

    logger.info("generating monthly report for %s", trade_date)
    raw, model_name = _dispatch_llm(MONTHLY_REPORT_SYSTEM_PROMPT, user_prompt, config)
    title, body = _parse_content_response(raw, trade_date, "월간리포트")
    warnings = _check_forbidden(title + "\n" + body)

    return ContentPost(
        title=title,
        body=body,
        content_type="monthly_report",
        model=model_name,
        tags=["월간리포트", "KOSPI", "KOSDAQ", "월간전망"],
        categories=["월간리포트", "시장분석"],
        warnings=warnings,
    )


def generate_yearly_report(
    *,
    snapshot: PeriodSnapshot,
    trade_date: str,
    config: Config,
) -> ContentPost:
    """연간 리포트 콘텐츠를 생성한다."""
    if snapshot.period != "yearly":
        raise ValueError(f"Expected yearly snapshot, got {snapshot.period}")

    user_prompt = _build_period_report_prompt(snapshot, trade_date=trade_date)

    logger.info("generating yearly report for %s", trade_date)
    raw, model_name = _dispatch_llm(YEARLY_REPORT_SYSTEM_PROMPT, user_prompt, config)
    title, body = _parse_content_response(raw, trade_date, "연간리포트")
    warnings = _check_forbidden(title + "\n" + body)

    return ContentPost(
        title=title,
        body=body,
        content_type="yearly_report",
        model=model_name,
        tags=["연간리포트", "KOSPI", "KOSDAQ", "연간결산"],
        categories=["연간리포트", "시장분석"],
        warnings=warnings,
    )
