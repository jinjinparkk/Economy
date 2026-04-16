"""WordPress.com REST API v1.1을 통한 비공개(draft) 자동 포스팅.

- MD → HTML 변환 후 draft 상태로 발행
- wp_posted.json으로 중복 방지
- 모든 에러는 로그만 남기고 파이프라인을 중단하지 않음
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import markdown
import requests

from src.analyzer import Article
from src.config import Config
from src.content_post import ContentPost

logger = logging.getLogger(__name__)

_WP_API_BASE = "https://public-api.wordpress.com/rest/v1.1"

# 본문 내 제목 크기 축소 (WP 기본 테마 대비 ~10px 작게).
# 글 제목(post title)은 테마 CSS로 결정되어 여기서 건드리지 않음.
_HEADING_STYLES: dict[str, str] = {
    "h1": "font-size:20px;margin-top:1.2em;margin-bottom:0.6em;",
    "h2": "font-size:16px;margin-top:1em;margin-bottom:0.5em;",
    "h3": "font-size:15px;margin-top:0.8em;margin-bottom:0.4em;",
    "h4": "font-size:14px;margin-top:0.7em;margin-bottom:0.3em;",
}


@dataclass
class PublishResult:
    """WordPress 발행 결과."""
    post_id: int
    url: str
    title: str
    status: str


def _md_to_html(body: str) -> str:
    """마크다운 본문을 HTML로 변환하고 제목 태그에 축소된 font-size를 주입한다."""
    html = markdown.markdown(
        body,
        extensions=["tables", "fenced_code"],
    )
    for tag, style in _HEADING_STYLES.items():
        html = html.replace(f"<{tag}>", f'<{tag} style="{style}">')
    return html


def _build_tags(article: Article) -> list[str]:
    """WordPress 태그 목록을 구성한다."""
    tags = [article.mover.name, "주식", "증시분석"]
    if article.mover.move_type == "surge":
        tags.append("급등")
    else:
        tags.append("급락")
    if article.mover.market:
        tags.append(article.mover.market)
    if article.mover.industry:
        tags.append(article.mover.industry)
    return tags


def _build_categories(article: Article) -> list[str]:
    """WordPress 카테고리 목록을 구성한다."""
    categories = ["종목분석"]
    if article.mover.move_type == "surge":
        categories.append("급등분석")
    else:
        categories.append("급락분석")
    if article.mover.market:
        categories.append(article.mover.market)
    return categories


def _load_posted(output_dir: Path) -> dict:
    """wp_posted.json을 로드한다."""
    path = output_dir / "wp_posted.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_posted(output_dir: Path, posted: dict) -> None:
    """wp_posted.json을 저장한다."""
    path = output_dir / "wp_posted.json"
    path.write_text(json.dumps(posted, ensure_ascii=False, indent=2), encoding="utf-8")


def publish_to_wordpress(
    article: Article,
    trade_date: str,
    config: Config,
) -> Optional[PublishResult]:
    """단일 글을 WordPress.com에 draft로 발행한다.

    - 크리덴셜 없으면 None 반환 (graceful skip)
    - 이미 발행된 글이면 None 반환 (중복 방지)
    - HTTP 에러 시 로그만 남기고 None 반환
    """
    if not config.wp_access_token or not config.wp_site_id:
        return None

    # 중복 체크
    filename = article.filename(trade_date)
    posted = _load_posted(config.output_dir)
    if filename in posted:
        logger.info("  [WP] already posted: %s (post_id=%s)", filename, posted[filename].get("post_id"))
        return None

    # MD → HTML 변환
    html_content = _md_to_html(article.body)

    # 태그 & 카테고리
    tags = _build_tags(article)
    categories = _build_categories(article)

    # WordPress API 호출
    url = f"{_WP_API_BASE}/sites/{config.wp_site_id}/posts/new"
    headers = {
        "Authorization": f"Bearer {config.wp_access_token}",
        "Content-Type": "application/json",
    }
    payload = {
        "title": article.title,
        "content": html_content,
        "status": "draft",
        "tags": ",".join(tags),
        "categories": ",".join(categories),
        "format": "standard",
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as exc:
        logger.error("  [WP] failed to publish '%s': %s", article.title[:40], str(exc)[:200])
        return None

    data = resp.json()
    post_id = data.get("ID", 0)
    post_url = data.get("URL", "")

    # 발행 기록 저장
    posted[filename] = {"post_id": post_id, "url": post_url}
    _save_posted(config.output_dir, posted)

    return PublishResult(
        post_id=post_id,
        url=post_url,
        title=article.title,
        status="draft",
    )


def publish_articles(
    articles: list[Article],
    trade_date: str,
    config: Config,
) -> list[PublishResult]:
    """여러 글을 일괄 발행한다."""
    results: list[PublishResult] = []
    for article in articles:
        result = publish_to_wordpress(article, trade_date, config)
        if result:
            results.append(result)
    return results


def publish_content_post(
    post: ContentPost,
    trade_date: str,
    config: Config,
) -> Optional[PublishResult]:
    """ContentPost를 WordPress.com에 draft로 발행한다.

    Article 버전과 동일한 로직이지만,
    tags/categories를 post 자체에서 가져온다.
    """
    if not config.wp_access_token or not config.wp_site_id:
        return None

    filename = post.filename(trade_date)
    posted = _load_posted(config.output_dir)
    if filename in posted:
        logger.info("  [WP] already posted: %s (post_id=%s)", filename, posted[filename].get("post_id"))
        return None

    html_content = _md_to_html(post.body)

    url = f"{_WP_API_BASE}/sites/{config.wp_site_id}/posts/new"
    headers = {
        "Authorization": f"Bearer {config.wp_access_token}",
        "Content-Type": "application/json",
    }
    payload = {
        "title": post.title,
        "content": html_content,
        "status": "draft",
        "tags": ",".join(post.tags),
        "categories": ",".join(post.categories),
        "format": "standard",
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as exc:
        logger.error("  [WP] failed to publish '%s': %s", post.title[:40], str(exc)[:200])
        return None

    data = resp.json()
    post_id = data.get("ID", 0)
    post_url = data.get("URL", "")

    posted[filename] = {"post_id": post_id, "url": post_url}
    _save_posted(config.output_dir, posted)

    return PublishResult(
        post_id=post_id,
        url=post_url,
        title=post.title,
        status="draft",
    )


def publish_content_posts(
    posts: list[ContentPost],
    trade_date: str,
    config: Config,
) -> list[PublishResult]:
    """여러 ContentPost를 일괄 발행한다."""
    results: list[PublishResult] = []
    for post in posts:
        result = publish_content_post(post, trade_date, config)
        if result:
            results.append(result)
    return results
