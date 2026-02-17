from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    hf_token: str | None = os.getenv("HF_TOKEN")
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")


settings = Settings()
