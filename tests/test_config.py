"""config 모듈 단위 테스트."""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from src.config import Config


class TestConfig:
    def test_defaults(self):
        env = {
            "LLM_PROVIDER": "gemini",
            "GEMINI_API_KEY": "test-key",
        }
        with patch.dict(os.environ, env, clear=False):
            cfg = Config.load()
        assert cfg.llm_provider == "gemini"
        assert cfg.gemini_api_key == "test-key"
        assert cfg.timezone == "Asia/Seoul"
        assert cfg.mover_threshold_pct == 5.0
        assert cfg.top_n_movers == 5

    def test_provider_case_insensitive(self):
        with patch.dict(os.environ, {"LLM_PROVIDER": "CLAUDE"}, clear=False):
            cfg = Config.load()
        assert cfg.llm_provider == "claude"

    def test_output_dir_created(self, tmp_path):
        out = tmp_path / "sub" / "out"
        with patch.dict(os.environ, {"OUTPUT_DIR": str(out)}, clear=False):
            cfg = Config.load()
        assert cfg.output_dir.exists()

    def test_frozen(self):
        cfg = Config.load()
        with pytest.raises(Exception):
            cfg.llm_provider = "claude"  # type: ignore[misc]
