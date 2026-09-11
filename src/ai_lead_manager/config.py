from functools import lru_cache

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


@lru_cache
def get_settings() -> Settings:
    return Settings()

