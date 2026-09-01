"""Environment-driven configuration.

Every setting has a default so the agent runs with nothing but an LLM API
key set. All of these can be overridden by environment variables (loaded
from a .env file if present) or by CLI flags in main.py.
"""

from __future__ import annotations

import logging
import os
import warnings
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Silence two cosmetic, harmless messages from the google-genai stack:
# 1. UserWarning when a model ignores our `temperature` (some Gemini models
#    use fixed sampling and can't be tuned) -- informational only.
# 2. A logged suggestion to use the Chat API instead of one-shot
#    generate_content calls -- doesn't apply to how LangChain drives it.
warnings.filterwarnings("ignore", message=".*uses fixed sampling defaults.*")
logging.getLogger("google_genai.models").setLevel(logging.ERROR)

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
CHUNKS_PATH = DATA_DIR / "index" / "chunks.jsonl"


@dataclass
class Settings:
    mode: str = os.getenv("AGENT_MODE", "offline")
    llm_provider: str = os.getenv("LLM_PROVIDER", "gemini")

    google_api_key: str | None = os.getenv("GOOGLE_API_KEY")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

    openrouter_api_key: str | None = os.getenv("OPENROUTER_API_KEY")
    openrouter_model: str = os.getenv(
        "OPENROUTER_MODEL", "google/gemini-2.0-flash-exp:free"
    )

    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.1")
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    tavily_api_key: str | None = os.getenv("TAVILY_API_KEY")

    retrieval_k: int = int(os.getenv("RETRIEVAL_K", "5"))

    # Without an explicit timeout, a stalled connection to the LLM API hangs
    # forever instead of failing. Cap it so the CLI always terminates.
    llm_timeout_seconds: float = float(os.getenv("LLM_TIMEOUT_SECONDS", "45"))
    llm_max_retries: int = int(os.getenv("LLM_MAX_RETRIES", "2"))

    def validate_mode(self, mode: str) -> str:
        mode = (mode or self.mode).lower()
        if mode not in ("offline", "online"):
            raise ValueError(f"AGENT_MODE/--mode must be 'offline' or 'online', got {mode!r}")
        return mode


settings = Settings()
