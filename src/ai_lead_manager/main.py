from fastapi import FastAPI

from ai_lead_manager.api.routes import router
from ai_lead_manager.config import get_settings


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.app_name, debug=settings.app_debug)
    app.include_router(router)
    return app


app = create_app()
