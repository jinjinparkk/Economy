"""설정 로더 — .env 파일을 읽어 환경변수로 노출한다."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# 프로젝트 루트의 .env 로드
_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_ROOT / ".env")


@dataclass(frozen=True)
class Config:
    # LLM 공급자 선택: "gemini" | "claude"
    llm_provider: str
    anthropic_api_key: str
    gemini_api_key: str
    naver_client_id: str
    naver_client_secret: str
    dart_api_key: str
    output_dir: Path
    timezone: str
    mover_threshold_pct: float
    volume_surge_multiplier: float
    top_n_movers: int
    wp_access_token: str
    wp_site_id: str
    wp_auto_publish: bool

    @classmethod
    def load(cls) -> "Config":
        output_dir = Path(os.getenv("OUTPUT_DIR", "./output"))
        if not output_dir.is_absolute():
            output_dir = _ROOT / output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        return cls(
            llm_provider=os.getenv("LLM_PROVIDER", "gemini").lower(),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
            gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
            naver_client_id=os.getenv("NAVER_CLIENT_ID", ""),
            naver_client_secret=os.getenv("NAVER_CLIENT_SECRET", ""),
            dart_api_key=os.getenv("DART_API_KEY", ""),
            output_dir=output_dir,
            timezone=os.getenv("TIMEZONE", "Asia/Seoul"),
            mover_threshold_pct=float(os.getenv("MOVER_THRESHOLD_PCT", "5.0")),
            volume_surge_multiplier=float(os.getenv("VOLUME_SURGE_MULTIPLIER", "3.0")),
            top_n_movers=int(os.getenv("TOP_N_MOVERS", "5")),
            wp_access_token=os.getenv("WP_ACCESS_TOKEN", ""),
            wp_site_id=os.getenv("WP_SITE_ID", ""),
            wp_auto_publish=os.getenv("WP_AUTO_PUBLISH", "true").lower() in ("true", "1", "yes"),
        )
