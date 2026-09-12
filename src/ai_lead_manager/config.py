from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "AI Lead Manager 3.0"
    app_env: str = "local"
    app_debug: bool = False
    database_url: str = Field(
        default="postgresql+psycopg://ai_lead_manager:local_dev_password@localhost:5432/ai_lead_manager"
    )
    ai_execution_mode: Literal["MOCK", "LIVE"] = "MOCK"
    allow_live_ai: bool = False
    ai_kill_switch: bool = False
    ai_default_qualification_provider: str = "mock"
    ai_default_qualification_model: str = "deterministic-keyword-v1"
    ai_default_draft_provider: str = "mock"
    ai_default_draft_model: str = "deterministic-template-v1"
    ai_connect_timeout_seconds: float = Field(default=3.0, gt=0, le=30)
    ai_read_timeout_seconds: float = Field(default=15.0, gt=0, le=120)
    ai_total_timeout_seconds: float = Field(default=20.0, gt=0, le=180)
    ai_max_attempts: int = Field(default=2, ge=1, le=5)
    ai_max_messages: int = Field(default=20, ge=1, le=100)
    ai_max_input_chars: int = Field(default=20_000, ge=100, le=200_000)
    ai_max_input_tokens: int = Field(default=8_000, ge=100, le=100_000)
    ai_max_output_tokens: int = Field(default=1_000, ge=1, le=20_000)
    ai_openai_compatible_providers_json: str = "[]"
    ai_model_pricing_json: str = "[]"


@lru_cache
def get_settings() -> Settings:
    return Settings()
