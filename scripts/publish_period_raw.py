"""LLM 없이 주간/월간/연간 스냅샷을 마크다운 테이블로 렌더링 후 WP draft로 발행.

Gemini 쿼터 소진 등으로 LLM 생성이 불가할 때 사용하는 데이터 덤프 리포트.

실행:
    py -3 scripts/publish_period_raw.py --weekly
    py -3 scripts/publish_period_raw.py --monthly
    py -3 scripts/publish_period_raw.py --yearly
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import Config
from src.content_post import ContentPost
from src.fetch_history import (
    PeriodSnapshot,
    MacroPeriodReturn,
    StockPeriodReturn,
    SectorPeriodReturn,
    fetch_period_snapshot,
)
from src.main import _save_content_post
from src.wordpress_publisher import publish_content_post

logger = logging.getLogger("publish_period_raw")


_DISCLAIMER = (
    "> **면책 고지**: 본 리포트는 공개 시장 데이터를 자동 집계한 자료이며, "
    "특정 종목의 매매를 권유하지 않습니다. 투자 판단과 책임은 전적으로 투자자 본인에게 있습니다.\n"
)


def _fmt_pct(v: float) -> str:
    sign = "+" if v >= 0 else ""
    return f"{sign}{v:.2f}%"


def _fmt_price(v: float) -> str:
    if v >= 10000:
        return f"{v:,.0f}"
    if v >= 100:
        return f"{v:,.2f}"
    return f"{v:.2f}"


def _render_macro_section(snap: PeriodSnapshot) -> str:
    if not snap.macro_returns:
        return ""
    rows = ["| 지표 | 시작가 | 종가 | 누적수익률 | 고가 | 저가 | 변동성(%) |",
            "|---|---:|---:|---:|---:|---:|---:|"]
    # 누적수익률 내림차순
    sorted_macros = sorted(
        snap.macro_returns.values(),
        key=lambda m: m.cumulative_return_pct,
        reverse=True,
    )
    for m in sorted_macros:
        rows.append(
            f"| {m.name} | {_fmt_price(m.start_close)} | {_fmt_price(m.end_close)} | "
            f"{_fmt_pct(m.cumulative_return_pct)} | {_fmt_price(m.high)} | "
            f"{_fmt_price(m.low)} | {m.volatility:.2f} |"
        )
    return f"## 글로벌 매크로 {snap.label} 수익률\n\n" + "\n".join(rows) + "\n"


def _render_sector_section(title: str, sectors: list[SectorPeriodReturn]) -> str:
    if not sectors:
        return ""
    rows = ["| 순위 | 섹터 | 코드 | 누적수익률 |",
            "|---:|---|---|---:|"]
    for s in sectors:
        rows.append(f"| {s.rank} | {s.name} | {s.code} | {_fmt_pct(s.cumulative_return_pct)} |")
    return f"## {title}\n\n" + "\n".join(rows) + "\n"


def _render_stocks_section(title: str, stocks: list[StockPeriodReturn]) -> str:
    if not stocks:
        return ""
    rows = ["| 순위 | 종목명 | 코드 | 업종 | 시작가 | 종가 | 누적수익률 | 평균거래대금(억) |",
            "|---:|---|---|---|---:|---:|---:|---:|"]
    for i, s in enumerate(stocks, 1):
        rows.append(
            f"| {i} | {s.name} | {s.code} | {s.industry or '-'} | "
            f"{_fmt_price(s.start_close)} | {_fmt_price(s.end_close)} | "
            f"{_fmt_pct(s.cumulative_return_pct)} | {s.avg_amount_eok:,.0f} |"
        )
    return f"## {title}\n\n" + "\n".join(rows) + "\n"


def _render_news_section(snap: PeriodSnapshot) -> str:
    if not snap.news_headlines:
        return ""
    lines = [f"## {snap.label} 주요 뉴스 헤드라인", ""]
    for h in snap.news_headlines[:15]:
        lines.append(f"- {h}")
    return "\n".join(lines) + "\n"


def _render_summary_header(snap: PeriodSnapshot, trade_date: str) -> str:
    return (
        f"**분석 기간**: {snap.start_date} ~ {snap.end_date} "
        f"({snap.trading_days}거래일) · **작성일**: {trade_date}\n\n"
        f"본 리포트는 자동 수집된 시장 데이터를 정리한 것으로, "
        f"해석이나 전망은 포함하지 않습니다.\n"
    )


def _render_period_report(snap: PeriodSnapshot, trade_date: str) -> tuple[str, str]:
    """스냅샷을 (title, body) 마크다운으로 렌더링."""
    label = snap.label
    title = f"[{label} 리포트] {trade_date} 최근 {snap.trading_days}거래일 시장 데이터 요약"

    parts: list[str] = []
    parts.append(_render_summary_header(snap, trade_date))
    parts.append(_render_macro_section(snap))

    # US 섹터
    if len(snap.us_sectors) >= 10:
        parts.append(_render_sector_section(
            f"US 섹터 ETF {label} 수익률 · 상위 5", snap.us_sectors[:5]
        ))
        parts.append(_render_sector_section(
            f"US 섹터 ETF {label} 수익률 · 하위 5", snap.us_sectors[-5:]
        ))
    elif snap.us_sectors:
        parts.append(_render_sector_section(
            f"US 섹터 ETF {label} 수익률", snap.us_sectors
        ))

    # 한국 업종
    if len(snap.kr_sectors) >= 10:
        parts.append(_render_sector_section(
            f"한국 업종 평균 {label} 수익률 · 상위 5", snap.kr_sectors[:5]
        ))
        parts.append(_render_sector_section(
            f"한국 업종 평균 {label} 수익률 · 하위 5", snap.kr_sectors[-5:]
        ))
    elif snap.kr_sectors:
        parts.append(_render_sector_section(
            f"한국 업종 평균 {label} 수익률", snap.kr_sectors
        ))

    # KOSPI/KOSDAQ TOP/BOTTOM
    parts.append(_render_stocks_section(
        f"KOSPI {label} 상승 TOP 10", snap.kospi_top,
    ))
    parts.append(_render_stocks_section(
        f"KOSPI {label} 하락 TOP 10", snap.kospi_bottom,
    ))
    parts.append(_render_stocks_section(
        f"KOSDAQ {label} 상승 TOP 10", snap.kosdaq_top,
    ))
    parts.append(_render_stocks_section(
        f"KOSDAQ {label} 하락 TOP 10", snap.kosdaq_bottom,
    ))

    # Mag7 (월간/연간)
    if snap.mag7_returns:
        parts.append(_render_stocks_section(
            f"Mag7 {label} 수익률", snap.mag7_returns,
        ))

    parts.append(_render_news_section(snap))
    parts.append(_DISCLAIMER)

    body = "\n".join(p for p in parts if p)
    return title, body


def _label_to_tags(period: str) -> tuple[list[str], list[str]]:
    label_map = {"weekly": "주간리포트", "monthly": "월간리포트", "yearly": "연간리포트"}
    label = label_map[period]
    tags = [label, "KOSPI", "KOSDAQ", "시장데이터"]
    categories = [label, "시장분석"]
    return tags, categories


def run(period: str) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    cfg = Config.load()

    logger.info("=" * 60)
    logger.info("RAW %s REPORT (NO LLM) — START", period.upper())
    logger.info("=" * 60)

    logger.info("[1/3] fetching %s snapshot...", period)
    snap = fetch_period_snapshot(period)
    if snap.is_empty():
        logger.error("empty snapshot — aborting")
        return
    logger.info(
        "snapshot: macro=%d, us_sec=%d, kr_sec=%d, "
        "kospi_top=%d, kospi_bot=%d, kosdaq_top=%d, kosdaq_bot=%d, mag7=%d, news=%d",
        len(snap.macro_returns), len(snap.us_sectors), len(snap.kr_sectors),
        len(snap.kospi_top), len(snap.kospi_bottom),
        len(snap.kosdaq_top), len(snap.kosdaq_bottom),
        len(snap.mag7_returns), len(snap.news_headlines),
    )

    today_str = datetime.now().strftime("%Y-%m-%d")
    title, body = _render_period_report(snap, today_str)
    tags, categories = _label_to_tags(period)

    post = ContentPost(
        title=title,
        body=body,
        content_type=f"{period}_report",
        model="raw-data-dump",
        tags=tags,
        categories=categories,
        warnings=[],
    )

    logger.info("[2/3] saving markdown...")
    path = _save_content_post(post, cfg.output_dir, today_str)
    logger.info("  saved: %s (%d chars)", path.name, len(body))

    if cfg.wp_auto_publish and cfg.wp_access_token and cfg.wp_site_id:
        logger.info("[3/3] publishing to WordPress.com as DRAFT...")
        result = publish_content_post(post, today_str, cfg)
        if result:
            logger.info("  [WP OK] post_id=%s → %s", result.post_id, result.url)
        else:
            logger.warning("  [WP] skipped or failed")
    else:
        logger.info("[3/3] WP auto_publish disabled — skipping")

    logger.info("=" * 60)
    logger.info("DONE")
    logger.info("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM-free period report publisher")
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--weekly", action="store_true")
    grp.add_argument("--monthly", action="store_true")
    grp.add_argument("--yearly", action="store_true")
    args = parser.parse_args()

    if args.weekly:
        run("weekly")
    elif args.monthly:
        run("monthly")
    else:
        run("yearly")


if __name__ == "__main__":
    main()
