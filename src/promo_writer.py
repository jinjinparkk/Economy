"""커뮤니티 홍보 텍스트 자동 생성.

파이프라인 실행 후 output/YYYY-MM-DD_홍보.txt 파일을 생성한다.
디시인사이드, 네이버 카페 등에 복사-붙여넣기용.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from src.analyzer import Article
from src.config import Config

logger = logging.getLogger(__name__)


def _find_wp_digest_url(trade_date: str, output_dir: Path) -> str:
    """wp_posted.json에서 다이제스트 URL을 찾는다."""
    wp_path = output_dir / "wp_posted.json"
    if not wp_path.exists():
        return ""
    try:
        posted = json.loads(wp_path.read_text(encoding="utf-8"))
        return posted.get(f"{trade_date}_digest", {}).get("url", "")
    except (json.JSONDecodeError, OSError):
        return ""


def generate_promo_text(
    articles: list[Article],
    trade_date: str,
    config: Config,
    telegram_url: str = "https://t.me/daily_stock_kr",
) -> Optional[Path]:
    """커뮤니티 홍보 텍스트를 생성하고 파일로 저장한다.

    Returns:
        저장된 파일 Path, 또는 articles가 비어있으면 None
    """
    if not articles:
        logger.info("  [PROMO] no articles — skipping promo text")
        return None

    surges = [a for a in articles if a.mover.change_pct > 0]
    plunges = [a for a in articles if a.mover.change_pct <= 0]

    lines: list[str] = []
    lines.append("오늘 급등주 정리해봄")
    lines.append("")

    for a in surges:
        m = a.mover
        reason = _extract_reason(a.title)
        reason_part = f" — {reason[:30]}" if reason else ""
        lines.append(f"🔺 {m.name} +{m.change_pct:.1f}%{reason_part}")

    if surges and plunges:
        lines.append("")

    for a in plunges:
        m = a.mover
        reason = _extract_reason(a.title)
        reason_part = f" — {reason[:30]}" if reason else ""
        lines.append(f"🔻 {m.name} {m.change_pct:.1f}%{reason_part}")

    lines.append("")

    wp_url = _find_wp_digest_url(trade_date, config.output_dir)
    if wp_url:
        lines.append(f"전문 분석 👉 {wp_url}")

    lines.append(f"텔레그램 채널 👉 {telegram_url}")

    text = "\n".join(lines)

    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{trade_date}_홍보.txt"
    path.write_text(text, encoding="utf-8")

    logger.info("  [PROMO] saved: %s (%d chars)", path.name, len(text))
    return path


def _extract_reason(title: str) -> str:
    """글 제목에서 '—' 또는 '–' 뒤의 이유 부분을 추출한다."""
    if "—" in title:
        return title.split("—", 1)[1].strip()
    if "–" in title:
        return title.split("–", 1)[1].strip()
    return ""
