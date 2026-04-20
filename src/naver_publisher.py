"""네이버 블로그 자동 포스팅 (OpenAPI writePost).

- MD -> HTML 변환 후 발행
- naver_posted.json으로 중복 방지
- 모든 에러는 로그만 남기고 파이프라인을 중단하지 않음
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import requests

from src.analyzer import Article
from src.config import Config
from src.content_post import ContentPost
from src.wordpress_publisher import PublishResult, _md_to_html, _build_daily_digest_body

logger = logging.getLogger(__name__)

_NAVER_BLOG_API = "https://openapi.naver.com/blog/writePost.json"

# ── 발행 기록 (중복 방지) ─────────────────────────────────────────────


def _load_posted(output_dir: Path) -> dict:
    """naver_posted.json을 로드한다."""
    path = output_dir / "naver_posted.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("  [NAVER] naver_posted.json load failed: %s", exc)
            return {}
    return {}


def _save_posted(output_dir: Path, posted: dict) -> None:
    """naver_posted.json을 저장한다."""
    path = output_dir / "naver_posted.json"
    path.write_text(json.dumps(posted, ensure_ascii=False, indent=2), encoding="utf-8")


# ── API 호출 ──────────────────────────────────────────────────────────


def _post_to_naver(
    title: str,
    html_content: str,
    config: Config,
) -> Optional[dict]:
    """네이버 블로그 API로 글을 발행한다.

    Returns:
        {"blogNo": "...", "logNo": "...", "url": "..."} 또는 None
    """
    headers = {
        "Authorization": f"Bearer {config.naver_access_token}",
    }
    payload = {
        "title": title,
        "contents": html_content,
    }

    try:
        resp = requests.post(_NAVER_BLOG_API, headers=headers, data=payload, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as exc:
        logger.error("  [NAVER] failed to publish '%s': %s", title[:40], str(exc)[:200])
        return None

    data = resp.json()
    if data.get("message") and data["message"].get("error"):
        logger.error("  [NAVER] API error: %s", data["message"].get("error", "unknown"))
        return None

    return data


# ── Article 발행 ──────────────────────────────────────────────────────


def publish_to_naver(
    article: Article,
    trade_date: str,
    config: Config,
) -> Optional[PublishResult]:
    """단일 Article을 네이버 블로그에 발행한다."""
    if not config.naver_access_token:
        return None

    filename = article.filename(trade_date)
    posted = _load_posted(config.output_dir)
    if filename in posted:
        logger.info("  [NAVER] already posted: %s", filename)
        return None

    html_content = _md_to_html(article.body)
    data = _post_to_naver(article.title, html_content, config)
    if not data:
        return None

    log_no = str(data.get("logNo", ""))
    post_url = data.get("url", "")

    posted[filename] = {"logNo": log_no, "url": post_url}
    _save_posted(config.output_dir, posted)

    return PublishResult(
        post_id=int(log_no) if log_no.isdigit() else 0,
        url=post_url,
        title=article.title,
        status="published",
    )


# ── ContentPost 발행 ─────────────────────────────────────────────────


def publish_content_post_to_naver(
    post: ContentPost,
    trade_date: str,
    config: Config,
) -> Optional[PublishResult]:
    """단일 ContentPost를 네이버 블로그에 발행한다."""
    if not config.naver_access_token:
        return None

    filename = post.filename(trade_date)
    posted = _load_posted(config.output_dir)
    if filename in posted:
        logger.info("  [NAVER] already posted: %s", filename)
        return None

    html_content = _md_to_html(post.body)
    data = _post_to_naver(post.title, html_content, config)
    if not data:
        return None

    log_no = str(data.get("logNo", ""))
    post_url = data.get("url", "")

    posted[filename] = {"logNo": log_no, "url": post_url}
    _save_posted(config.output_dir, posted)

    return PublishResult(
        post_id=int(log_no) if log_no.isdigit() else 0,
        url=post_url,
        title=post.title,
        status="published",
    )


# ── 일일 통합 포스트 ─────────────────────────────────────────────────


def publish_daily_digest_to_naver(
    articles: list[Article],
    content_posts: list[ContentPost],
    trade_date: str,
    config: Config,
) -> Optional[PublishResult]:
    """모든 articles + content_posts를 하나의 통합 포스트로 네이버에 발행한다."""
    if not config.naver_access_token:
        return None

    digest_key = f"{trade_date}_digest"
    posted = _load_posted(config.output_dir)
    if digest_key in posted:
        logger.info("  [NAVER] already posted digest: %s", digest_key)
        return None

    parts_date = trade_date.split("-")
    date_str = f"{parts_date[0]}년 {parts_date[1]}월 {parts_date[2]}일"
    title = f"{date_str} 증시 리포트 — 오늘의 급등주·급락주 정리"

    body = _build_daily_digest_body(articles, content_posts)
    html_content = _md_to_html(body)

    data = _post_to_naver(title, html_content, config)
    if not data:
        return None

    log_no = str(data.get("logNo", ""))
    post_url = data.get("url", "")

    posted[digest_key] = {"logNo": log_no, "url": post_url, "title": title}
    for a in articles:
        posted[a.filename(trade_date)] = {"logNo": log_no, "url": post_url, "title": a.title}
    for p in content_posts:
        posted[p.filename(trade_date)] = {"logNo": log_no, "url": post_url, "title": p.title}
    _save_posted(config.output_dir, posted)

    return PublishResult(
        post_id=int(log_no) if log_no.isdigit() else 0,
        url=post_url,
        title=title,
        status="published",
    )
