"""전체 파이프라인 오케스트레이터.

흐름:
1. 거시경제·해외 증시 스냅샷 수집
2. 국내 시장 스냅샷 수집 (KOSPI/KOSDAQ 전 종목 + 지수)
3. 급등/급락 종목 탐지
4. 각 종목별 뉴스 수집 (Google News RSS)
5. 전망 데이터 계산 (기술적 지표 + 통계 패턴 + ML 예측)
6. LLM으로 탑다운 분석 글 생성 (거시→섹터→종목→전망)
7. output/ 폴더에 마크다운 저장

실행:
    py -3 -m src.main
    py -3 -m src.main --top 3 --threshold 10   # TOP 3, 10% 이상만
    py -3 -m src.main --dry-run                # LLM 호출 생략 (데이터만 수집)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.config import Config
from src.content_post import ContentPost
from src.fetch_macro import MacroSnapshot, fetch_macro_snapshot

logger = logging.getLogger(__name__)


def _setup_logging() -> None:
    # Windows 콘솔 UTF-8 강제 (basicConfig 보다 먼저 수행해야 stream 핸들러에 반영됨)
    for stream in (sys.stdout, sys.stderr):
        if stream.encoding and stream.encoding.lower() != "utf-8":
            try:
                stream.reconfigure(encoding="utf-8")
            except Exception:
                pass
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def _save_article(article: Article, output_dir: Path, trade_date: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / article.filename(trade_date)
    path.write_text(article.to_markdown(), encoding="utf-8")
    return path


def _save_content_post(post: ContentPost, output_dir: Path, trade_date: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / post.filename(trade_date)
    path.write_text(post.to_markdown(), encoding="utf-8")
    return path


def _outlook_to_dict(outlook: OutlookData) -> dict:
    """OutlookData를 JSON-serializable dict로 변환."""
    result: dict = {}
    if outlook.technical:
        t = outlook.technical
        result["technical"] = {
            "rsi_14": t.rsi_14,
            "macd": t.macd,
            "macd_signal": t.macd_signal,
            "macd_histogram": t.macd_histogram,
            "bb_upper": t.bb_upper,
            "bb_middle": t.bb_middle,
            "bb_lower": t.bb_lower,
            "bb_position": t.bb_position,
            "ma_5": t.ma_5,
            "ma_20": t.ma_20,
            "ma_60": t.ma_60,
            "ma_trend": t.ma_trend,
            "volume_ratio": t.volume_ratio,
            "obv_trend": t.obv_trend,
            "rsi_divergence": t.rsi_divergence,
            "signal_summary": t.signal_summary,
        }
    if outlook.pattern:
        p = outlook.pattern
        result["pattern"] = {
            "event_type": p.event_type,
            "sample_count": p.sample_count,
            "avg_return_1d": p.avg_return_1d,
            "avg_return_5d": p.avg_return_5d,
            "positive_rate_1d": p.positive_rate_1d,
            "positive_rate_5d": p.positive_rate_5d,
        }
    if outlook.prediction:
        pr = outlook.prediction
        result["prediction"] = {
            "direction": pr.direction,
            "confidence": pr.confidence,
            "confidence_grade": pr.confidence_grade,
            "cv_accuracy": pr.cv_accuracy,
            "features_used": pr.features_used,
            "model_type": pr.model_type,
        }
    return result


def _save_dry_run_summary(
    report: MoverReport,
    news_map: dict[str, list],
    macro: MacroSnapshot,
    output_dir: Path,
    outlook_map: Optional[dict[str, OutlookData]] = None,
    breadth: Optional[dict] = None,
) -> Path:
    """dry-run 모드: 수집된 데이터를 JSON으로 저장."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{report.trade_date}_dry_run.json"

    def mover_dict(m: Mover) -> dict:
        d = {
            "code": m.code,
            "name": m.name,
            "market": m.market,
            "industry": m.industry,
            "move_type": m.move_type,
            "close": m.close,
            "change_pct": m.change_pct,
            "relative_strength": m.relative_strength,
            "amount_eok": m.amount / 1e8,
            "hints": m.reason_hints,
            "news_count": len(news_map.get(m.code, [])),
            "news_titles": [n["title"] for n in news_map.get(m.code, [])],
        }
        if outlook_map and m.code in outlook_map:
            d["outlook"] = _outlook_to_dict(outlook_map[m.code])
        return d

    macro_dict = {}
    for key, ind in macro.all_indicators.items():
        macro_dict[ind.name] = {
            "close": ind.close,
            "change_pct": ind.change_pct,
            "date": ind.date,
        }

    data = {
        "trade_date": report.trade_date,
        "macro": macro_dict,
        "market_regime": macro.market_regime,
        "yield_spread": macro.yield_spread,
        "macro_narrative": macro.to_narrative() if not macro.is_empty() else "",
        "market_breadth": breadth if breadth else {},
        "surges": [mover_dict(m) for m in report.surges],
        "plunges": [mover_dict(m) for m in report.plunges],
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def run_pipeline(
    top_n: int = 3,
    threshold_pct: float = 5.0,
    dry_run: bool = False,
    news_per_stock: int = 5,
    only_surges: bool = False,
) -> None:
    """전체 파이프라인 실행."""
    # 포스트마켓 전용 모듈 (FinanceDataReader, scikit-learn 필요)
    from src.analyzer import Article, generate_article
    from src.content_generator import (
        generate_daily_market,
        generate_sector_report,
        generate_quant_insight,
    )
    from src.detect_movers import Mover, MoverReport, detect_movers
    from src.fetch_market import fetch_market_snapshot
    from src.fetch_news import NewsItem, fetch_news_for_stock
    from src.predictor import OutlookData, compute_outlook

    started = time.time()
    cfg = Config.load()

    logger.info("=" * 60)
    logger.info("STOCK DAILY BLOG PIPELINE START")
    logger.info("dry_run=%s top_n=%d threshold=%.1f%%", dry_run, top_n, threshold_pct)
    logger.info("=" * 60)

    # 1) 거시경제 스냅샷
    logger.info("[STEP 1/7] fetching macro snapshot...")
    macro = fetch_macro_snapshot()
    logger.info("macro: %d indicators", len(macro.all_indicators))

    # 2) 국내 시장 스냅샷
    logger.info("[STEP 2/7] fetching market snapshot...")
    snapshot = fetch_market_snapshot()
    logger.info("market snapshot: %d KOSPI + %d KOSDAQ stocks on %s",
                len(snapshot.kospi), len(snapshot.kosdaq), snapshot.trade_date)

    # 시장 브레드스 (시장 체온)
    breadth = snapshot.market_breadth()
    sector_data = snapshot.sector_breadth()
    logger.info("market breadth: %d up / %d down (%.1f%% up ratio)",
                breadth["total_up"], breadth["total_down"], breadth["up_ratio"])

    # 3) 급등/급락 탐지
    logger.info("[STEP 3/7] detecting movers...")
    report = detect_movers(snapshot, threshold_pct=threshold_pct, top_n=top_n)
    logger.info("found %d surges, %d plunges", len(report.surges), len(report.plunges))

    targets: list[Mover] = list(report.surges)
    if not only_surges:
        targets += list(report.plunges)

    if not targets:
        logger.warning("no movers found above threshold, exiting")
        return

    # 4) 종목별 뉴스 수집
    logger.info("[STEP 4/7] fetching news for %d stocks...", len(targets))
    news_map: dict[str, list] = {}
    for m in targets:
        items = fetch_news_for_stock(m.name, cfg, display=news_per_stock)
        news_map[m.code] = [n.to_dict() for n in items]
        logger.info("  %s (%s): %d news", m.name, m.industry or "미분류", len(items))
        time.sleep(0.5)

    # 5) 전망 데이터 계산
    logger.info("[STEP 5/7] computing outlook data for %d stocks...", len(targets))
    outlook_map: dict[str, OutlookData] = {}
    for m in targets:
        try:
            outlook_map[m.code] = compute_outlook(m)
            parts = []
            ol = outlook_map[m.code]
            if ol.technical:
                parts.append("TA")
            if ol.pattern:
                parts.append(f"패턴({ol.pattern.sample_count}건)")
            if ol.prediction:
                parts.append(f"ML({ol.prediction.direction})")
            logger.info("  %s: %s", m.name, " + ".join(parts) if parts else "데이터 부족")
        except Exception as exc:
            logger.warning("  %s outlook failed: %s", m.name, str(exc)[:100])
            outlook_map[m.code] = OutlookData()

    # dry-run은 여기까지
    if dry_run:
        logger.info("[STEP 6/7] SKIPPED (dry-run)")
        logger.info("[STEP 7/7] saving dry-run summary...")
        path = _save_dry_run_summary(report, news_map, macro, cfg.output_dir,
                                     outlook_map=outlook_map, breadth=breadth)
        logger.info("dry-run summary saved to %s", str(path))
        elapsed = time.time() - started
        logger.info("PIPELINE DONE in %.1fs", elapsed)
        return

    # 6) LLM 분석 글 생성
    logger.info("[STEP 6/7] generating articles via %s...", cfg.llm_provider)
    key_ok = (
        (cfg.llm_provider == "gemini" and cfg.gemini_api_key)
        or (cfg.llm_provider == "claude" and cfg.anthropic_api_key)
    )
    if not key_ok:
        logger.error("LLM API key not set for provider=%s — skipping article generation",
                     cfg.llm_provider)
        logger.error("add the key to .env and re-run, or use --dry-run to test data pipeline only")
        return

    index_summary = {
        k: {"Close": float(v["Close"]), "ChangePct": float(v["ChangePct"])}
        for k, v in snapshot.indices.items()
    }
    macro_summary = macro.to_summary_dict() if not macro.is_empty() else None

    # Gemini 2.0-flash 무료 tier는 분당 15건 — 5초 간격이면 안전.
    # Claude는 훨씬 관대해서 1초만.
    call_interval = 5.0 if cfg.llm_provider == "gemini" else 1.0

    articles: list[Article] = []
    quota_exhausted = False
    for idx, m in enumerate(targets):
        news_items = [
            NewsItem(
                title=n["title"], description=n["description"],
                link=n["link"], press=n["press"],
                pub_date=datetime.fromisoformat(n["pub_date"]) if n["pub_date"] else None,
            )
            for n in news_map[m.code]
        ]

        # 재시도 로직: 일시적 rate-limit만 재시도, quota 소진이면 나머지 전부 스킵
        for attempt in range(3):
            try:
                article = generate_article(
                    m, news_items, cfg, index_summary,
                    macro_summary=macro_summary,
                    outlook=outlook_map.get(m.code),
                    macro_narrative=macro.to_narrative() if not macro.is_empty() else None,
                    market_breadth=breadth,
                    sector_breadth=sector_data,
                )
                articles.append(article)
                logger.info("  [OK] %s — %s", m.name, article.title[:50])
                break
            except Exception as exc:
                msg = str(exc).lower()
                is_429 = "429" in msg
                is_quota = "quota" in msg or "resource_exhausted" in msg
                is_rate = "rate" in msg

                if is_quota:
                    logger.error("  [QUOTA EXHAUSTED] %s — 일일 한도 소진, 나머지 종목 스킵", m.name)
                    quota_exhausted = True
                    break
                elif (is_429 or is_rate) and attempt < 2:
                    wait = 30.0 * (attempt + 1)  # 30s → 60s
                    logger.warning("  [RATE LIMIT] %s, retrying in %.0fs (attempt %d/3)",
                                   m.name, wait, attempt + 1)
                    time.sleep(wait)
                    continue
                else:
                    logger.error("  [FAIL] %s: %s", m.name, str(exc)[:200])
                    break

        if quota_exhausted:
            logger.warning("quota exhausted — skipping remaining %d stocks",
                           len(targets) - idx - 1)
            break

        # 마지막 호출이 아니면 간격 유지
        if idx < len(targets) - 1:
            time.sleep(call_interval)

    # 6b) 콘텐츠 포스트 생성 (데일리시황 / 섹터리포트 / 퀀트인사이트)
    logger.info("[STEP 6b] generating content posts...")
    content_posts: list[ContentPost] = []

    # mover 요약 문자열 (데일리시황 / 섹터리포트용)
    surges_summary = [
        f"{m.name} {m.change_pct:+.1f}% ({m.industry or '미분류'})"
        for m in report.surges[:3]
    ]
    plunges_summary = [
        f"{m.name} {m.change_pct:+.1f}% ({m.industry or '미분류'})"
        for m in report.plunges[:3]
    ]

    # 업종별 급등/급락 종목 그룹핑 (섹터리포트용)
    surges_by_sector: dict[str, list[str]] = {}
    for m in report.surges:
        key = m.industry or "미분류"
        surges_by_sector.setdefault(key, []).append(f"{m.name} {m.change_pct:+.1f}%")
    plunges_by_sector: dict[str, list[str]] = {}
    for m in report.plunges:
        key = m.industry or "미분류"
        plunges_by_sector.setdefault(key, []).append(f"{m.name} {m.change_pct:+.1f}%")

    content_generators = [
        ("데일리시황", generate_daily_market, dict(
            macro_summary=macro_summary,
            macro_narrative=macro.to_narrative() if not macro.is_empty() else None,
            breadth=breadth,
            index_summary=index_summary,
            sector_breadth=sector_data,
            surges_summary=surges_summary,
            plunges_summary=plunges_summary,
            trade_date=report.trade_date,
            config=cfg,
        )),
        ("섹터리포트", generate_sector_report, dict(
            sector_breadth=sector_data,
            macro_narrative=macro.to_narrative() if not macro.is_empty() else None,
            breadth=breadth,
            surges_by_sector=surges_by_sector,
            plunges_by_sector=plunges_by_sector,
            trade_date=report.trade_date,
            config=cfg,
        )),
        ("퀀트인사이트", generate_quant_insight, dict(
            outlook_map=outlook_map,
            trade_date=report.trade_date,
            config=cfg,
        )),
    ]

    for c_idx, (label, gen_fn, kwargs) in enumerate(content_generators):
        if quota_exhausted:
            logger.warning("  [SKIP] %s — quota exhausted", label)
            break

        for attempt in range(3):
            try:
                post = gen_fn(**kwargs)
                content_posts.append(post)
                logger.info("  [OK] %s — %s", label, post.title[:50])
                break
            except Exception as exc:
                msg = str(exc).lower()
                is_quota = "quota" in msg or "resource_exhausted" in msg
                is_rate = "429" in msg or "rate" in msg

                if is_quota:
                    logger.error("  [QUOTA EXHAUSTED] %s — 일일 한도 소진", label)
                    quota_exhausted = True
                    break
                elif is_rate and attempt < 2:
                    wait = 30.0 * (attempt + 1)
                    logger.warning("  [RATE LIMIT] %s, retrying in %.0fs (attempt %d/3)",
                                   label, wait, attempt + 1)
                    time.sleep(wait)
                    continue
                else:
                    logger.error("  [FAIL] %s: %s", label, str(exc)[:200])
                    break

        # 마지막 호출이 아니면 간격 유지
        if c_idx < len(content_generators) - 1:
            time.sleep(call_interval)

    # 7) 저장
    total_count = len(articles) + len(content_posts)
    logger.info("[STEP 7/7] saving %d articles + %d content posts...",
                len(articles), len(content_posts))
    for article in articles:
        path = _save_article(article, cfg.output_dir, report.trade_date)
        logger.info("  saved: %s", path.name)
        if article.warnings:
            logger.warning("    warnings: %s", article.warnings)
    for post in content_posts:
        path = _save_content_post(post, cfg.output_dir, report.trade_date)
        logger.info("  saved: %s", path.name)
        if post.warnings:
            logger.warning("    warnings: %s", post.warnings)

    # 7b) WordPress 발행
    if cfg.wp_auto_publish and cfg.wp_access_token and cfg.wp_site_id:
        from src.wordpress_publisher import publish_articles, publish_content_posts
        logger.info("[STEP 7b] publishing to WordPress.com as DRAFT...")
        wp_results = publish_articles(articles, report.trade_date, cfg)
        for r in wp_results:
            logger.info("  [WP] %s → %s", r.title[:40], r.url)
        wp_content_results = publish_content_posts(content_posts, report.trade_date, cfg)
        for r in wp_content_results:
            logger.info("  [WP] %s → %s", r.title[:40], r.url)

    elapsed = time.time() - started
    logger.info("=" * 60)
    logger.info("PIPELINE DONE in %.1fs — %d items (%d articles + %d content posts) in %s",
                elapsed, total_count, len(articles), len(content_posts), cfg.output_dir)
    logger.info("=" * 60)


def _fetch_us_market_news() -> list[str]:
    """미국 증시 관련 뉴스 헤드라인을 수집한다."""
    from src.fetch_news import _fetch_via_google_rss

    queries = ["미국 증시", "나스닥 뉴욕증시", "월가 Fed"]
    seen_titles: set[str] = set()
    headlines: list[str] = []
    for q in queries:
        items = _fetch_via_google_rss(q, display=5)
        for item in items:
            if item.title not in seen_titles:
                seen_titles.add(item.title)
                tag = f"[{item.press}]" if item.press != "unknown" else ""
                headlines.append(f"{tag} {item.title}".strip())
    return headlines[:10]


def _fetch_econ_calendar_news() -> list[str]:
    """오늘 예정된 경제 이벤트 뉴스 수집."""
    from src.fetch_news import _fetch_via_google_rss

    queries = ["오늘 경제지표 발표", "FOMC 금리", "미국 CPI 고용"]
    seen_titles: set[str] = set()
    headlines: list[str] = []
    for q in queries:
        items = _fetch_via_google_rss(q, display=3)
        for item in items:
            if item.title not in seen_titles:
                seen_titles.add(item.title)
                tag = f"[{item.press}]" if item.press != "unknown" else ""
                headlines.append(f"{tag} {item.title}".strip())
    return headlines[:5]


def run_pre_market_pipeline() -> None:
    """프리마켓 브리핑 파이프라인. 미국 증시 마감 후 실행."""
    from src.content_generator import generate_pre_market

    started = time.time()
    cfg = Config.load()

    logger.info("=" * 60)
    logger.info("PRE-MARKET BRIEFING PIPELINE START")
    logger.info("=" * 60)

    # 1) 거시경제 수집 (미국 지표)
    logger.info("[STEP 1/4] fetching macro snapshot...")
    macro = fetch_macro_snapshot()
    logger.info("macro: %d indicators", len(macro.all_indicators))

    # 2) 미국 증시 뉴스 + 경제 캘린더 수집
    logger.info("[STEP 2/4] fetching US market news + econ calendar...")
    us_news = _fetch_us_market_news()
    econ_news = _fetch_econ_calendar_news()
    logger.info("us news: %d headlines, econ calendar: %d headlines",
                len(us_news), len(econ_news))

    # 3) 프리마켓 브리핑 생성
    logger.info("[STEP 3/4] generating pre-market briefing via %s...", cfg.llm_provider)
    key_ok = (
        (cfg.llm_provider == "gemini" and cfg.gemini_api_key)
        or (cfg.llm_provider == "claude" and cfg.anthropic_api_key)
    )
    if not key_ok:
        logger.error("LLM API key not set for provider=%s — cannot generate briefing",
                     cfg.llm_provider)
        return

    today_str = datetime.now().strftime("%Y-%m-%d")

    # 새 데이터 변환
    sectors_dict = {ind.name: {"Close": ind.close, "ChangePct": ind.change_pct}
                    for ind in macro.sectors.values()} if macro.sectors else None

    mega_dict = {ind.name: {"Close": ind.close, "ChangePct": ind.change_pct}
                 for ind in macro.mega_caps.values()} if macro.mega_caps else None

    style_dict = None
    if macro.style:
        style_dict = {
            "growth_value_ratio": macro.growth_value_ratio,
            "items": {ind.name: {"ChangePct": ind.change_pct}
                      for ind in macro.style.values()},
        }

    asia_dict = {ind.name: {"Close": ind.close, "ChangePct": ind.change_pct}
                 for ind in macro.asia.values()} if macro.asia else None

    europe_dict = {ind.name: {"Close": ind.close, "ChangePct": ind.change_pct}
                   for ind in macro.europe.values()} if macro.europe else None

    credit_dict = None
    if macro.credit:
        credit_dict = {
            "stress": macro.credit_stress,
            "items": {ind.name: {"ChangePct": ind.change_pct}
                      for ind in macro.credit.values()},
        }

    post = None
    for attempt in range(3):
        try:
            post = generate_pre_market(
                macro_summary=macro.to_summary_dict() if not macro.is_empty() else None,
                macro_narrative=macro.to_narrative() if not macro.is_empty() else None,
                market_regime=macro.market_regime,
                yield_spread=macro.yield_spread,
                us_news=us_news if us_news else None,
                trade_date=today_str,
                config=cfg,
                sectors=sectors_dict,
                mega_caps=mega_dict,
                style_signals=style_dict,
                asia_indices=asia_dict,
                europe_indices=europe_dict,
                credit_signals=credit_dict,
                econ_calendar=econ_news if econ_news else None,
            )
            logger.info("  [OK] %s", post.title[:60])
            if post.warnings:
                logger.warning("  warnings: %s", post.warnings)
            break
        except Exception as exc:
            msg = str(exc).lower()
            is_rate = "429" in msg or "rate" in msg or "quota" in msg or "resource_exhausted" in msg
            if is_rate and attempt < 2:
                wait = 30.0 * (attempt + 1)
                logger.warning("  [RATE LIMIT] retrying in %.0fs (attempt %d/3)", wait, attempt + 1)
                time.sleep(wait)
                continue
            else:
                logger.error("  [FAIL] pre-market generation: %s", str(exc)[:200])
                return

    if post is None:
        logger.error("  [FAIL] pre-market generation failed after retries")
        return

    # 4) 저장 + WordPress 발행
    logger.info("[STEP 4/4] saving pre-market briefing...")
    path = _save_content_post(post, cfg.output_dir, today_str)
    logger.info("  saved: %s", path.name)

    if cfg.wp_auto_publish and cfg.wp_access_token and cfg.wp_site_id:
        from src.wordpress_publisher import publish_content_post
        logger.info("  publishing to WordPress.com as DRAFT...")
        result = publish_content_post(post, today_str, cfg)
        if result:
            logger.info("  [WP] %s → %s", result.title[:40], result.url)

    elapsed = time.time() - started
    logger.info("=" * 60)
    logger.info("PRE-MARKET PIPELINE DONE in %.1fs — %s", elapsed, path.name)
    logger.info("=" * 60)


def _run_period_pipeline(period: str) -> None:
    """주간/월간/연간 공통 파이프라인.

    4단계:
      1) API 키 선체크
      2) fetch_period_snapshot
      3) LLM 생성 (rate-limit 3회 재시도 30s→60s)
      4) 저장 + WP 발행
    """
    from src.fetch_history import fetch_period_snapshot
    from src.content_generator import (
        generate_weekly_report,
        generate_monthly_report,
        generate_yearly_report,
    )

    started = time.time()
    cfg = Config.load()

    label_map = {"weekly": "주간", "monthly": "월간", "yearly": "연간"}
    label = label_map.get(period, period)
    generators = {
        "weekly": generate_weekly_report,
        "monthly": generate_monthly_report,
        "yearly": generate_yearly_report,
    }
    gen_fn = generators[period]

    logger.info("=" * 60)
    logger.info("%s REPORT PIPELINE START", label.upper())
    logger.info("=" * 60)

    # 1) API 키 선체크
    key_ok = (
        (cfg.llm_provider == "gemini" and cfg.gemini_api_key)
        or (cfg.llm_provider == "claude" and cfg.anthropic_api_key)
    )
    if not key_ok:
        logger.error("LLM API key not set for provider=%s — skipping %s report",
                     cfg.llm_provider, label)
        return

    # 2) 스냅샷 수집
    logger.info("[STEP 1/4] fetching %s snapshot...", label)
    snapshot = fetch_period_snapshot(period)  # type: ignore[arg-type]
    if snapshot.is_empty():
        logger.warning("%s snapshot is empty — aborting", label)
        return
    logger.info("%s snapshot: macro=%d, us_sec=%d, kr_sec=%d, kospi_top=%d, kosdaq_top=%d, news=%d",
                label, len(snapshot.macro_returns), len(snapshot.us_sectors),
                len(snapshot.kr_sectors), len(snapshot.kospi_top),
                len(snapshot.kosdaq_top), len(snapshot.news_headlines))

    # 3) LLM 생성
    today_str = datetime.now().strftime("%Y-%m-%d")
    logger.info("[STEP 2/4] generating %s report via %s...", label, cfg.llm_provider)

    post: Optional[ContentPost] = None
    for attempt in range(3):
        try:
            post = gen_fn(snapshot=snapshot, trade_date=today_str, config=cfg)
            logger.info("  [OK] %s — %s", label, post.title[:60])
            if post.warnings:
                logger.warning("  warnings: %s", post.warnings)
            break
        except Exception as exc:
            msg = str(exc).lower()
            is_quota = "quota" in msg or "resource_exhausted" in msg
            is_rate = "429" in msg or "rate" in msg
            if is_quota:
                logger.error("  [QUOTA EXHAUSTED] %s report — 일일 한도 소진", label)
                return
            if is_rate and attempt < 2:
                wait = 30.0 * (attempt + 1)
                logger.warning("  [RATE LIMIT] retrying in %.0fs (attempt %d/3)",
                               wait, attempt + 1)
                time.sleep(wait)
                continue
            logger.error("  [FAIL] %s report: %s", label, str(exc)[:200])
            return

    if post is None:
        logger.error("  [FAIL] %s report failed after retries", label)
        return

    # 4) 저장 + WP 발행
    logger.info("[STEP 3/4] saving %s report...", label)
    path = _save_content_post(post, cfg.output_dir, today_str)
    logger.info("  saved: %s", path.name)

    if cfg.wp_auto_publish and cfg.wp_access_token and cfg.wp_site_id:
        from src.wordpress_publisher import publish_content_post
        logger.info("[STEP 4/4] publishing to WordPress.com as DRAFT...")
        result = publish_content_post(post, today_str, cfg)
        if result:
            logger.info("  [WP] %s → %s", result.title[:40], result.url)

    elapsed = time.time() - started
    logger.info("=" * 60)
    logger.info("%s REPORT PIPELINE DONE in %.1fs — %s",
                label.upper(), elapsed, path.name)
    logger.info("=" * 60)


def run_weekly_pipeline() -> None:
    """주간 리포트 파이프라인."""
    _run_period_pipeline("weekly")


def run_monthly_pipeline() -> None:
    """월간 리포트 파이프라인."""
    _run_period_pipeline("monthly")


def run_yearly_pipeline() -> None:
    """연간 리포트 파이프라인."""
    _run_period_pipeline("yearly")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stock daily blog pipeline")
    parser.add_argument("--top", type=int, default=3,
                        help="방향별 최대 종목 수 (default: 3)")
    parser.add_argument("--threshold", type=float, default=5.0,
                        help="최소 등락률 (default: 5.0)")
    parser.add_argument("--dry-run", action="store_true",
                        help="LLM 호출 생략 (데이터 수집만)")
    parser.add_argument("--only-surges", action="store_true",
                        help="급등주만 분석 (급락 제외)")
    parser.add_argument("--news-per-stock", type=int, default=5,
                        help="종목당 수집할 뉴스 개수 (default: 5)")
    parser.add_argument("--pre-market", action="store_true",
                        help="프리마켓 브리핑만 생성 (미국 증시 마감 후)")
    parser.add_argument("--weekly", action="store_true",
                        help="주간 리포트 생성 (5거래일)")
    parser.add_argument("--monthly", action="store_true",
                        help="월간 리포트 생성 (21거래일)")
    parser.add_argument("--yearly", action="store_true",
                        help="연간 리포트 생성 (252거래일)")
    args = parser.parse_args()

    _setup_logging()

    if args.pre_market:
        run_pre_market_pipeline()
    elif args.weekly:
        run_weekly_pipeline()
    elif args.monthly:
        run_monthly_pipeline()
    elif args.yearly:
        run_yearly_pipeline()
    else:
        run_pipeline(
            top_n=args.top,
            threshold_pct=args.threshold,
            dry_run=args.dry_run,
            news_per_stock=args.news_per_stock,
            only_surges=args.only_surges,
        )


if __name__ == "__main__":
    main()
