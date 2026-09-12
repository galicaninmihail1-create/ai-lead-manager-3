from functools import lru_cache

from ai_lead_manager.application.ai.executor import AIExecutor
from ai_lead_manager.application.ai.registry import ProviderRegistry
from ai_lead_manager.config import get_settings
from ai_lead_manager.infrastructure.ai.composition import (
    build_ai_executor,
    build_provider_registry,
)
from ai_lead_manager.infrastructure.database import SessionFactory


@lru_cache
def get_provider_registry() -> ProviderRegistry:
    return build_provider_registry(get_settings())


@lru_cache
def get_ai_executor() -> AIExecutor:
    settings = get_settings()
    return build_ai_executor(settings, SessionFactory)
