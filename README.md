# stock_daily_blog

매일 장 마감 후 KOSPI/KOSDAQ 급등·급락 종목 TOP N을 자동 탐지하고,
거시경제 맥락 + Google News + 기술적 지표 + ML 예측 + LLM으로
"왜 움직였는가" + "앞으로의 시사점"까지 다루는 탑다운 분석 글을 마크다운으로 생성한다.

추가로 **프리마켓 브리핑** 파이프라인을 통해 미국 증시 마감 후(KST 06:00)
한국 시장에 미칠 영향을 예측하는 별도 글을 자동 생성한다.

WordPress.com에 자동 발행(draft)하여 애드센스(금융 키워드 CPC 최상위권) 수익화 가능.

---

## 빠른 시작

### 1) 의존 패키지 설치
```bash
py -3 -m pip install -r requirements.txt
```

### 2) `.env` 설정
`.env.example`을 `.env`로 복사한 뒤 키 입력:
```env
LLM_PROVIDER=gemini                 # gemini | claude
GEMINI_API_KEY=AIzaSy...            # https://aistudio.google.com/apikey 에서 무료 발급
ANTHROPIC_API_KEY=sk-ant-...        # 선택 (LLM_PROVIDER=claude 일 때)
OUTPUT_DIR=./output
TIMEZONE=Asia/Seoul
MOVER_THRESHOLD_PCT=5.0
TOP_N_MOVERS=5
```

### 3) 실행
```bash
# ── 장 마감 파이프라인 (15:30 이후) ──────────────────────

# 데이터만 수집 (LLM 호출 없음, 키 없어도 동작)
py -3 -m src.main --dry-run --top 3

# 전체 파이프라인 (급등 3 + 급락 3 → 6개 글 생성)
py -3 -m src.main --top 3

# 급등주만
py -3 -m src.main --top 5 --only-surges

# 임계값 변경 (10% 이상만)
py -3 -m src.main --top 3 --threshold 10

# ── 프리마켓 브리핑 (05:50~06:00, 미국 장 마감 후) ────

py -3 -m src.main --pre-market

# ── 개별 모듈 실행 ───────────────────────────────────────

py -3 -m src.technical 005930       # 삼성전자 기술적 지표 출력
py -3 -m src.predictor 005930 10.0  # 삼성전자 +10% 전망 데이터
```

생성 결과는 `output/{YYYY-MM-DD}_{종목명}_{급등|급락}.md` 형식으로 저장된다.
프리마켓 브리핑은 `output/{YYYY-MM-DD}_프리마켓브리핑.md`로 저장된다.

---

## 파이프라인 구조

### A. 장 마감 파이프라인 (7단계) — `py -3 -m src.main`

```
[STEP 1] fetch_macro.py      전일 미국 증시·환율·유가·VIX·미국채 스냅샷
             │
             ▼
[STEP 2] fetch_market.py     KOSPI/KOSDAQ 전 종목 + 지수 스냅샷
             │
             ▼
[STEP 3] detect_movers.py    threshold% 이상 등락 종목 TOP N 탐지
                              + 힌트 태깅 (상한가/거래량급증/거래대금상위)
             │
             ▼
[STEP 4] fetch_news.py       종목별 Google News RSS 5건 수집
             │
             ▼
[STEP 5] technical.py        기술적 지표 계산 (RSI/MACD/BB/MA/거래량비율)
         predictor.py         과거 유사 패턴 통계 + ML 방향 예측
             │
             ▼
[STEP 6] analyzer.py         LLM(Gemini/Claude) 탑다운 분석 글 생성
         content_generator.py 데일리시황 / 섹터리포트 / 퀀트인사이트
             │
             ▼
[STEP 7] main.py             output/ 폴더에 마크다운 저장
```

### B. 프리마켓 브리핑 (3단계) — `py -3 -m src.main --pre-market`

```
[STEP 1] fetch_macro.py        미국 증시 마감 데이터 (S&P500/NASDAQ/DOW/SOXX/VIX/채권/환율/원자재)
             │
             ▼
[STEP 2] content_generator.py  LLM이 한국 시장 영향 분석 (프리마켓 브리핑 생성)
             │
             ▼
[STEP 3] main.py               output/ 저장 + WordPress draft 발행
```

> 장 마감 파이프라인과 **완전 독립**. `--pre-market` 플래그로 분리되며,
> 한국 시장 데이터(fetch_market)는 수집하지 않는다.

---

## 파일 구조

| 파일 | 역할 |
|---|---|
| `src/config.py` | `Config` 데이터클래스, `.env` 로드, LLM provider 분기 |
| `src/fetch_macro.py` | `MacroSnapshot` — S&P500/NASDAQ/다우/SOXX/환율/WTI/VIX/미국채10Y |
| `src/fetch_market.py` | `MarketSnapshot` — KOSPI/KOSDAQ 종목 + KS11/KQ11/KS200 지수 |
| `src/detect_movers.py` | `Mover`, `MoverReport` — 급등/급락 탐지 + 힌트 부착 |
| `src/fetch_news.py` | `NewsItem` — Google News RSS 우선, 네이버 API 폴백 |
| `src/technical.py` | `TechnicalIndicators` — RSI(14), MACD, 볼린저밴드, MA(5/20/60), 거래량비율 |
| `src/predictor.py` | `PatternStats`, `DirectionPrediction`, `OutlookData` — 통계 패턴 + ML 예측 |
| `src/analyzer.py` | `Article`, `generate_article()` — Gemini/Claude 분기 + 금지어 검사 |
| `src/content_generator.py` | 데일리시황 / 섹터리포트 / 퀀트인사이트 / **프리마켓브리핑** 생성 |
| `src/content_post.py` | `ContentPost` 데이터클래스 — 종목 비귀속 콘텐츠 |
| `src/main.py` | 전체 오케스트레이터 + CLI (`--top`, `--dry-run`, `--pre-market` 등) |
| `tests/` | 11개 테스트 파일, **259/259 PASS** |
| `output/` | 생성된 마크다운 글 + dry-run JSON |

---

## 전망 기능 (기술적 분석 + 통계 패턴 + ML 예측)

### 기술적 지표 (`src/technical.py`)

| 지표 | 설명 | 판단 기준 |
|---|---|---|
| RSI(14) | Wilder 방식 상대강도지수 | >70 과매수, <30 과매도 |
| MACD | EMA(12)-EMA(26), 시그널 EMA(9) | 시그널 상회=매수 시그널, 하회=매도 시그널 |
| 볼린저밴드 | MA(20) ± 2σ | 상단돌파/상단근접/중립/하단근접/하단이탈 |
| 이동평균 | SMA 5/20/60일 | 정배열(5>20>60), 역배열, 혼조 |
| 거래량비율 | 당일 / 20일 평균 | 3배 이상이면 거래량 급증 |

### 과거 패턴 통계 (`src/predictor.py`)

해당 종목의 최근 2년 히스토리에서 유사 등락률 이벤트를 추출하고,
이벤트 후 1일/5일 수익률 평균 및 양수 비율을 집계한다.

```
예시 — 광전자 상한가(+30%) 패턴:
  과거 6건 중 익일 평균 +18.18%, 상승 확률 80%
  5일 평균 +1.19%, 상승 확률 50%
```

- 사례 5건 미만이면 통계를 생략 (신뢰도 부족)

### ML 방향 예측 (`src/predictor.py`)

- 모델: `sklearn.LogisticRegression`
- 피처: RSI, MACD histogram, BB %B, Volume ratio, 등락률
- 라벨: 익일 수익률 > 0 → 상승, 아니면 하락
- 신뢰도 55% 미만이면 "중립" 반환 (과적합 방지)
- **sklearn 미설치 시 ML 예측만 건너뜀** (기술적 지표·패턴 통계는 정상 동작)

---

## LLM 공급자 전환

`.env`의 `LLM_PROVIDER`만 바꾸면 된다. `analyzer.py` 내부 모델명:
- **Gemini**: `gemini-2.0-flash` (무료 tier 분당 15건)
- **Claude**: `claude-sonnet-4-6` (유료, 분당 제한 훨씬 관대)

> Gemini 2.5-flash는 무료 tier가 분당 5건으로 너무 빡빡해서 2.0-flash 사용.
> Claude는 글당 약 30~50원 수준으로 하루 6~9건이면 월 1만원 미만.

---

## 안전장치 (법적 컴플라이언스)

본 프로젝트는 사실 전달 목적의 기사 생성기이며, **유사투자자문업 신고 대상이 아니다.**
다만 다음 안전장치는 절대 제거하지 말 것:

1. **시스템 프롬프트** (`analyzer.py:SYSTEM_PROMPT`)
   - "추천", "매수하세요", "목표가", "손절", "익절" 등 사용 금지
   - "~으로 관측된다", "~로 풀이된다" 중립 표현 강제
   - 뉴스에 없는 정보 임의 생성 금지
   - 전망 섹션에 "기술적 지표와 통계는 참고 자료일 뿐 미래를 보장하지 않습니다" 필수

2. **금지어 후처리 필터** (`analyzer.py:FORBIDDEN_WORDS`)
   - "매수 추천", "목표가", "손절", "반드시 오른다", "확실히 떨어진다", "무조건" 등 16개
   - 생성된 글에 금지어가 포함되면 `Article.warnings`에 기록
   - 운영 시 warnings 있는 글은 발행 보류 권장

3. **면책 고지** — 시스템 프롬프트에서 본문 하단 필수 삽입 지시

---

## CLI 옵션

```
--top N              방향별 최대 종목 수 (기본 3)
--threshold PCT      최소 등락률 % (기본 5.0)
--dry-run            LLM 호출 없이 데이터만 수집 → JSON 저장
--only-surges        급등주만 분석 (급락 제외)
--news-per-stock N   종목당 수집할 뉴스 개수 (기본 5)
--pre-market         프리마켓 브리핑만 생성 (미국 증시 마감 후, 장 마감 파이프라인과 독립)
```

---

## 테스트

```bash
py -3 -m pytest tests/ -v    # 259/259 PASS
```

| 테스트 파일 | 테스트 수 | 내용 |
|---|---|---|
| `test_config.py` | 4 | Config 로드, frozen, 디폴트 |
| `test_fetch_macro.py` | 21 | MacroSnapshot, 지표 수집, 파생 지표 |
| `test_fetch_market.py` | 7 | 시장 브레드스, 섹터 브레드스 |
| `test_fetch_news.py` | 22 | RSS 파싱, 뉴스 수집, 감성/신뢰도 |
| `test_detect_movers.py` | 17 | 급등/급락 탐지, 힌트, 상대강도 |
| `test_analyzer.py` | 25 | 프롬프트, 파싱, 금지어, LLM 모킹 |
| `test_content_generator.py` | 44 | 데일리시황/섹터/퀀트/프리마켓 프롬프트+생성 |
| `test_pipeline.py` | 14 | 파이프라인 E2E, dry-run, 프리마켓 |
| `test_technical.py` | 30 | RSI/MACD/BB/MA/거래량비율/OBV |
| `test_predictor.py` | 26 | 패턴 통계, ML 예측, OutlookData |
| `test_wordpress_publisher.py` | 15 | WordPress 발행, 중복 체크 |

---

## 운영 예시

### dry-run JSON (outlook 포함)
```json
{
  "code": "017900",
  "name": "광전자",
  "change_pct": 30.0,
  "outlook": {
    "technical": {
      "rsi_14": 93.36,
      "macd_histogram": 739.77,
      "bb_position": "상단돌파",
      "ma_trend": "정배열",
      "volume_ratio": 2.55
    },
    "pattern": {
      "event_type": "상한가",
      "sample_count": 6,
      "avg_return_1d": 18.18,
      "positive_rate_1d": 80.0
    }
  }
}
```

### 생성되는 글 구조
- 제목 (SEO 친화적)
- 30초 요약 (3줄 불릿)
- 글로벌 매크로 동향
- 주가 현황 (종가/등락률/거래대금/시총)
- 급등(급락) 배경 (뉴스 기반 추정)
- 주요 뉴스 요약
- 업종 및 시장 맥락
- **전망 및 시사점** (기술적 지표 해석 + 과거 패턴 통계 + 종합 시사점)
- 면책 고지

---

## 스케줄 운영

| 시각 | 파이프라인 | 명령 |
|---|---|---|
| **05:50 KST** | 프리마켓 브리핑 | `py -3 -m src.main --pre-market` |
| **15:40 KST** | 장 마감 분석 | `py -3 -m src.main --top 3` |

Windows 작업 스케줄러 또는 cron으로 등록.

---

## 로드맵

- [x] 1단계: 데이터 수집 + 탐지 + LLM 분석 + 마크다운 저장
- [x] 2단계: 거시경제(미국 증시·환율·유가·VIX) 탑다운 분석
- [x] 3단계: 기술적 지표 + 통계 패턴 + ML 예측 → 전망 섹션
- [x] 4단계: 멀티 콘텐츠 (데일리시황 / 섹터리포트 / 퀀트인사이트)
- [x] 5단계: WordPress 자동 발행 (draft)
- [x] 6단계: 프리마켓 브리핑 (미국 증시 → 한국 영향 예측)
- [ ] 7단계: 매일 자동 실행 (Windows 작업 스케줄러)
- [ ] 8단계: 차트 이미지 첨부 (matplotlib/plotly)
- [ ] 9단계: 스레드 분석 (3일 연속 급등 종목 추적)

---

## 수익 모델 (참고)

- 플랫폼: **WordPress.com + 애드센스** (금융 키워드 CPC 국내 최상위권)
- 비용: Gemini 무료 또는 Claude 월 7,000원 미만
- 법적 위험: 유사투자자문업 신고 X (사실 전달만), 금지어 필터 + 면책 고지로 방어
- 콘텐츠 차별점: 뉴스 단순 요약이 아닌 "왜 움직였는지" + "앞으로의 시사점" 분석
