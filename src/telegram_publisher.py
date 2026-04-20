"""Telegram Bot API를 통한 채널 자동 발송.

- 제목 + 30초 요약 + 전문 보기 링크를 HTML 파싱 모드로 전송
- telegram_posted.json으로 중복 방지
- 429 rate-limit 시 retry_after 대기 후 재시도
- 모든 에러는 로그만 남기고 파이프라인을 중단하지 않음
"""
from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Optional

import requests

from src.analyzer import Article
from src.config import Config
from src.content_post import ContentPost
from src.wordpress_publisher import PublishResult

logger = logging.getLogger(__name__)

_TG_API_BASE = "https://api.telegram.org"
_MAX_MESSAGE_LEN = 4096

# ── 텔레그램 발행 기록 ───────────────────────────────────────────────


def _load_posted(output_dir: Path) -> dict:
    """telegram_posted.json을 로드한다."""
    path = output_dir / "telegram_posted.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("  [TG] telegram_posted.json load failed: %s", exc)
            return {}
    return {}


def _save_posted(output_dir: Path, posted: dict) -> None:
    """telegram_posted.json을 저장한다."""
    path = output_dir / "telegram_posted.json"
    path.write_text(json.dumps(posted, ensure_ascii=False, indent=2), encoding="utf-8")


# ── 요약 추출 ────────────────────────────────────────────────────────


def _extract_summary(body: str) -> str:
    """본문에서 '30초 요약' 섹션 또는 첫 불릿 목록을 추출한다."""
    lines = body.split("\n")

    # 1) "30초 요약" 섹션 찾기
    summary_start = -1
    for i, line in enumerate(lines):
        if re.search(r"30초\s*요약", line, re.IGNORECASE):
            summary_start = i + 1
            break

    if summary_start >= 0:
        summary_lines: list[str] = []
        for line in lines[summary_start:]:
            stripped = line.strip()
            # 다음 헤딩이 나오면 중단
            if stripped.startswith("#") and not stripped.startswith("##"):
                break
            if stripped.startswith("## "):
                break
            if stripped:
                summary_lines.append(stripped)
            # 빈 줄이 나오고 이미 내용이 있으면 중단
            elif summary_lines:
                break
        if summary_lines:
            return "\n".join(summary_lines[:5])

    # 2) 첫 불릿 목록 추출 (- 또는 * 시작)
    bullet_lines: list[str] = []
    in_bullets = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(("- ", "* ", "• ")):
            in_bullets = True
            bullet_lines.append(stripped)
        elif in_bullets and stripped:
            # 들여쓴 줄은 이전 불릿의 연장
            if line.startswith(("  ", "\t")):
                bullet_lines.append(stripped)
            else:
                break
        elif in_bullets and not stripped:
            break
    if bullet_lines:
        return "\n".join(bullet_lines[:5])

    # 3) 폴백: 첫 비헤딩, 비빈 줄 3개
    content_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            content_lines.append(stripped)
            if len(content_lines) >= 3:
                break
    return "\n".join(content_lines)


# ── 메시지 포맷팅 ────────────────────────────────────────────────────


def _escape_html(text: str) -> str:
    """Telegram HTML 파싱에 필요한 특수문자 이스케이프."""
    return (
        text
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _format_message(
    title: str,
    body: str,
    wp_url: str = "",
) -> str:
    """텔레그램 메시지를 HTML 포맷으로 구성한다."""
    summary = _extract_summary(body)

    # 마크다운 불릿을 이모지로 교체하고 HTML 이스케이프
    summary_escaped = _escape_html(summary)
    # 불릿 포인트를 보기 좋게
    summary_escaped = re.sub(r"^[-*•]\s*", "▸ ", summary_escaped, flags=re.MULTILINE)

    parts = [
        f"📊 <b>{_escape_html(title)}</b>",
        "",
        summary_escaped,
    ]

    if wp_url:
        parts.append("")
        parts.append(f'📎 <a href="{wp_url}">전문 보기</a>')

    parts.append("")
    parts.append("🤖 자동 생성 | 투자 판단은 본인 책임")

    message = "\n".join(parts)

    # 4096자 제한 처리
    if len(message) > _MAX_MESSAGE_LEN:
        # 요약 부분을 줄여서 맞춤
        overflow = len(message) - _MAX_MESSAGE_LEN + 4  # "..." 여유
        summary_escaped = summary_escaped[: len(summary_escaped) - overflow] + "..."
        parts_trimmed = [
            f"📊 <b>{_escape_html(title)}</b>",
            "",
            summary_escaped,
        ]
        if wp_url:
            parts_trimmed.append("")
            parts_trimmed.append(f'📎 <a href="{wp_url}">전문 보기</a>')
        parts_trimmed.append("")
        parts_trimmed.append("🤖 자동 생성 | 투자 판단은 본인 책임")
        message = "\n".join(parts_trimmed)

    return message


# ── Telegram API 호출 ────────────────────────────────────────────────


def send_to_telegram(
    title: str,
    body: str,
    config: Config,
    wp_url: str = "",
) -> Optional[PublishResult]:
    """텔레그램 채널에 메시지를 전송한다.

    Returns:
        PublishResult(post_id=message_id, url="", title=title, status="sent")
        또는 실패 시 None
    """
    if not config.telegram_bot_token or not config.telegram_channel_id:
        return None

    message = _format_message(title, body, wp_url=wp_url)

    url = f"{_TG_API_BASE}/bot{config.telegram_bot_token}/sendMessage"
    payload = {
        "chat_id": config.telegram_channel_id,
        "text": message,
        "parse_mode": "HTML",
        "disable_web_page_preview": False,
    }

    # 최대 2회 재시도 (429 rate-limit)
    for attempt in range(3):
        try:
            resp = requests.post(url, json=payload, timeout=10)

            if resp.status_code == 429:
                data = resp.json()
                retry_after = data.get("parameters", {}).get("retry_after", 5)
                if attempt < 2:
                    logger.warning("  [TG] rate limited, waiting %ds (attempt %d/3)",
                                   retry_after, attempt + 1)
                    time.sleep(retry_after)
                    continue
                else:
                    logger.error("  [TG] rate limited after 3 attempts: %s", title[:40])
                    return None

            if resp.status_code != 200:
                logger.error("  [TG] failed to send '%s': HTTP %d — %s",
                             title[:40], resp.status_code, resp.text[:200])
                return None

            data = resp.json()
            if not data.get("ok"):
                logger.error("  [TG] API error for '%s': %s",
                             title[:40], data.get("description", "unknown"))
                return None

            message_id = data.get("result", {}).get("message_id", 0)
            return PublishResult(
                post_id=message_id,
                url="",
                title=title,
                status="sent",
            )

        except requests.RequestException as exc:
            logger.error("  [TG] request failed for '%s': %s", title[:40], str(exc)[:200])
            return None

    return None


# ── WordPress URL 조회 ───────────────────────────────────────────────


def _find_wp_url(filename: str, output_dir: Path) -> str:
    """wp_posted.json에서 해당 파일의 WordPress URL을 찾는다."""
    wp_path = output_dir / "wp_posted.json"
    if not wp_path.exists():
        return ""
    try:
        posted = json.loads(wp_path.read_text(encoding="utf-8"))
        return posted.get(filename, {}).get("url", "")
    except (json.JSONDecodeError, OSError):
        return ""


# ── Article 발행 ─────────────────────────────────────────────────────


def publish_article_to_telegram(
    article: Article,
    trade_date: str,
    config: Config,
) -> Optional[PublishResult]:
    """단일 Article을 텔레그램 채널에 발송한다."""
    if not config.telegram_bot_token or not config.telegram_channel_id:
        return None

    filename = article.filename(trade_date)
    posted = _load_posted(config.output_dir)
    if filename in posted:
        logger.info("  [TG] already posted: %s (msg_id=%s)", filename, posted[filename].get("msg_id"))
        return None

    wp_url = _find_wp_url(filename, config.output_dir)
    result = send_to_telegram(article.title, article.body, config, wp_url=wp_url)

    if result:
        posted[filename] = {"msg_id": result.post_id, "title": result.title}
        _save_posted(config.output_dir, posted)

    return result


def publish_articles_to_telegram(
    articles: list[Article],
    trade_date: str,
    config: Config,
) -> list[PublishResult]:
    """여러 Article을 일괄 발송한다."""
    results: list[PublishResult] = []
    for article in articles:
        result = publish_article_to_telegram(article, trade_date, config)
        if result:
            results.append(result)
    return results


# ── ContentPost 발행 ─────────────────────────────────────────────────


def publish_content_post_to_telegram(
    post: ContentPost,
    trade_date: str,
    config: Config,
) -> Optional[PublishResult]:
    """단일 ContentPost를 텔레그램 채널에 발송한다."""
    if not config.telegram_bot_token or not config.telegram_channel_id:
        return None

    filename = post.filename(trade_date)
    posted = _load_posted(config.output_dir)
    if filename in posted:
        logger.info("  [TG] already posted: %s (msg_id=%s)", filename, posted[filename].get("msg_id"))
        return None

    wp_url = _find_wp_url(filename, config.output_dir)
    result = send_to_telegram(post.title, post.body, config, wp_url=wp_url)

    if result:
        posted[filename] = {"msg_id": result.post_id, "title": result.title}
        _save_posted(config.output_dir, posted)

    return result


def publish_content_posts_to_telegram(
    posts: list[ContentPost],
    trade_date: str,
    config: Config,
) -> list[PublishResult]:
    """여러 ContentPost를 일괄 발송한다."""
    results: list[PublishResult] = []
    for post in posts:
        result = publish_content_post_to_telegram(post, trade_date, config)
        if result:
            results.append(result)
    return results


# ── 일일 다이제스트 (단일 메시지) ─────────────────────────────────────


def _build_digest_message(
    articles: list[Article],
    content_posts: list[ContentPost],
    trade_date: str,
    output_dir: Path,
) -> str:
    """articles + content_posts를 하나의 텔레그램 메시지로 합친다."""
    # 날짜 포맷: 2026-04-20 → 2026년 04월 20일
    parts_date = trade_date.split("-")
    date_str = f"{parts_date[0]}년 {parts_date[1]}월 {parts_date[2]}일"

    parts: list[str] = [f"📊 <b>{date_str} 증시 리포트</b>", ""]

    surges = [a for a in articles if a.mover.change_pct > 0]
    plunges = [a for a in articles if a.mover.change_pct <= 0]

    if surges:
        parts.append("🔺 <b>급등</b>")
        for a in surges:
            m = a.mover
            reason = ""
            if "—" in a.title:
                reason = a.title.split("—", 1)[1].strip()
            elif "–" in a.title:
                reason = a.title.split("–", 1)[1].strip()
            line = f"▸ {_escape_html(m.name)} +{m.change_pct:.1f}%"
            if reason:
                line += f" — {_escape_html(reason[:30])}"
            parts.append(line)
        parts.append("")

    if plunges:
        parts.append("🔻 <b>급락</b>")
        for a in plunges:
            m = a.mover
            reason = ""
            if "—" in a.title:
                reason = a.title.split("—", 1)[1].strip()
            elif "–" in a.title:
                reason = a.title.split("–", 1)[1].strip()
            line = f"▸ {_escape_html(m.name)} {m.change_pct:.1f}%"
            if reason:
                line += f" — {_escape_html(reason[:30])}"
            parts.append(line)
        parts.append("")

    # 통합 WordPress 링크
    digest_url = _find_wp_url(f"{trade_date}_digest", output_dir)
    if digest_url:
        parts.append(f'📎 <a href="{digest_url}">전문 보기</a>')
        parts.append("")

    parts.append("🤖 자동 생성 | 투자 판단은 본인 책임")

    message = "\n".join(parts)

    if len(message) > _MAX_MESSAGE_LEN:
        message = message[:_MAX_MESSAGE_LEN - 3] + "..."

    return message


def publish_daily_digest_to_telegram(
    articles: list[Article],
    content_posts: list[ContentPost],
    trade_date: str,
    config: Config,
) -> Optional[PublishResult]:
    """모든 articles + content_posts를 하나의 다이제스트 메시지로 발송한다."""
    if not config.telegram_bot_token or not config.telegram_channel_id:
        return None

    digest_key = f"{trade_date}_digest"
    posted = _load_posted(config.output_dir)
    if digest_key in posted:
        logger.info("  [TG] already posted digest: %s (msg_id=%s)",
                     digest_key, posted[digest_key].get("msg_id"))
        return None

    message = _build_digest_message(articles, content_posts, trade_date, config.output_dir)

    url = f"{_TG_API_BASE}/bot{config.telegram_bot_token}/sendMessage"
    payload = {
        "chat_id": config.telegram_channel_id,
        "text": message,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }

    for attempt in range(3):
        try:
            resp = requests.post(url, json=payload, timeout=10)

            if resp.status_code == 429:
                data = resp.json()
                retry_after = data.get("parameters", {}).get("retry_after", 5)
                if attempt < 2:
                    logger.warning("  [TG] rate limited, waiting %ds (attempt %d/3)",
                                   retry_after, attempt + 1)
                    time.sleep(retry_after)
                    continue
                else:
                    logger.error("  [TG] rate limited after 3 attempts: digest")
                    return None

            if resp.status_code != 200:
                logger.error("  [TG] failed to send digest: HTTP %d — %s",
                             resp.status_code, resp.text[:200])
                return None

            data = resp.json()
            if not data.get("ok"):
                logger.error("  [TG] API error for digest: %s",
                             data.get("description", "unknown"))
                return None

            message_id = data.get("result", {}).get("message_id", 0)
            result = PublishResult(
                post_id=message_id,
                url="",
                title=f"{trade_date} daily digest",
                status="sent",
            )

            # 다이제스트 + 개별 파일 모두 posted 기록
            posted[digest_key] = {"msg_id": message_id, "title": result.title}
            for a in articles:
                posted[a.filename(trade_date)] = {"msg_id": message_id, "title": a.title}
            for p in content_posts:
                posted[p.filename(trade_date)] = {"msg_id": message_id, "title": p.title}
            _save_posted(config.output_dir, posted)

            return result

        except requests.RequestException as exc:
            logger.error("  [TG] request failed for digest: %s", str(exc)[:200])
            return None

    return None
