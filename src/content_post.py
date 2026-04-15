"""멀티 콘텐츠 포스트 데이터클래스.

Article이 종목 분석 전용이라면, ContentPost는 시황·섹터·퀀트 등
종목에 귀속되지 않는 콘텐츠를 표현한다.

content_type:
- "daily_market"   → 데일리 시황
- "sector_report"  → 섹터 리포트
- "quant_insight"  → 퀀트 인사이트
- "pre_market"     → 프리마켓 브리핑
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

_CONTENT_LABELS: dict[str, str] = {
    "daily_market": "데일리시황",
    "sector_report": "섹터리포트",
    "quant_insight": "퀀트인사이트",
    "pre_market": "프리마켓브리핑",
}


@dataclass
class ContentPost:
    """종목 비귀속 콘텐츠 포스트."""

    title: str
    body: str
    content_type: str          # "daily_market" | "sector_report" | "quant_insight" | "pre_market"
    model: str
    tags: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_markdown(self) -> str:
        """제목 + 본문을 한 마크다운 문서로."""
        return f"# {self.title}\n\n{self.body}\n"

    def filename(self, trade_date: str) -> str:
        """파일명 생성: YYYY-MM-DD_데일리시황.md"""
        label = _CONTENT_LABELS.get(self.content_type, self.content_type)
        safe = re.sub(r"[^\w가-힣]+", "_", label)
        return f"{trade_date}_{safe}.md"
