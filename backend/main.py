"""IntelliChat FastAPI application with injected services and safe errors.

Run with: python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
"""

import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException

from backend.api import routes_chat, routes_documents, routes_health
from backend.core.config import Settings
from backend.core.exceptions import AppError
from backend.core.logging_config import configure_logging
from backend.core.state import SessionRegistry
from backend.schemas.common import ErrorResponse
from backend.services.chat_service import ChatService
from backend.services.document_service import DocumentService

logger = logging.getLogger(__name__)


def create_app(
    settings: Settings | None = None, llm_factory=None, embedding_factory=None
) -> FastAPI:
    """Build an isolated app; tests inject deterministic models and embeddings."""
    settings = settings or Settings()
    configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        """Log startup and release all in-memory user data at shutdown."""
        logger.info(
            "Backend startup: application=%s version=%s", settings.app_name, settings.app_version
        )
        yield
        application.state.registry.sessions.clear()

    application = FastAPI(
        title="AI-Powered Chatbot",
        description="IntelliChat – AI-Powered Knowledge Assistant. Session-scoped REST API.",
        version=settings.app_version,
        lifespan=lifespan,
        responses={
            code: {"model": ErrorResponse} for code in (409, 413, 415, 422, 429, 500, 502, 503, 504)
        },
    )
    application.state.settings = settings
    registry = SessionRegistry(settings)
    application.state.registry = registry
    application.state.chat_service = ChatService(settings, registry, llm_factory)
    application.state.document_service = DocumentService(settings, registry, embedding_factory)
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type"],
    )

    @application.exception_handler(AppError)
    async def app_error_handler(request: Request, error: AppError) -> JSONResponse:
        """Translate known service failures without exposing provider details."""
        return JSONResponse(
            status_code=error.status, content={"code": error.code, "message": error.message}
        )

    @application.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request, error: RequestValidationError
    ) -> JSONResponse:
        """Avoid echoing rejected message bodies, uploaded data, or secret inputs."""
        return JSONResponse(
            status_code=422,
            content={
                "code": "invalid_request",
                "message": (
                    "Invalid request. Check required fields, session ID, "
                    "and allowed values in /docs."
                ),
            },
        )

    @application.exception_handler(HTTPException)
    async def http_error_handler(request: Request, error: HTTPException) -> JSONResponse:
        """Normalize routing and multipart errors to the same safe contract."""
        return JSONResponse(
            status_code=error.status_code,
            content={"code": "http_error", "message": "The HTTP request could not be completed."},
        )

    @application.exception_handler(Exception)
    async def unexpected_error_handler(request: Request, error: Exception) -> JSONResponse:
        """Log only the exception type; return no raw stack or document content."""
        logger.error("Unexpected application failure: type=%s", type(error).__name__)
        return JSONResponse(
            status_code=500,
            content={
                "code": "internal_error",
                "message": "An unexpected error occurred. Please retry.",
            },
        )

    for router in (routes_health.router, routes_chat.router, routes_documents.router):
        application.include_router(router)
    return application


app = create_app()

if __name__ == "__main__":
    runtime_settings = Settings()
    uvicorn.run(
        "backend.main:app", host=runtime_settings.fastapi_host, port=runtime_settings.fastapi_port
    )
