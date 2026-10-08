"""Expose liveness and safe frontend settings without testing paid providers."""

from fastapi import APIRouter, Request

from backend.schemas.common import ConfigResponse, HealthResponse

router = APIRouter(tags=["System"])


@router.get("/", response_model=HealthResponse)
@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    """Report process health even when API credentials have not been configured."""
    settings = request.app.state.settings
    return HealthResponse(application=settings.app_name, version=settings.app_version)


@router.get("/api/config", response_model=ConfigResponse)
def config(request: Request) -> ConfigResponse:
    """Return an explicit allowlist of settings; never serialize the Settings object."""
    settings = request.app.state.settings
    chat_key = bool(settings.llm_api_key.get_secret_value())
    embed_key = bool(settings.embedding_api_key.get_secret_value())
    return ConfigResponse(
        application=settings.app_name,
        version=settings.app_version,
        llm_provider=settings.llm_provider,
        llm_model=settings.llm_model,
        embedding_provider=settings.embedding_provider,
        embedding_model=settings.embedding_model,
        chat_configured=settings.llm_provider == "ollama" or chat_key,
        embeddings_configured=(
            settings.embedding_provider == "ollama"
            or embed_key
            or (settings.llm_provider == "openai" and chat_key)
        ),
        top_k=settings.top_k,
        max_upload_mb=settings.max_upload_mb,
        max_files_per_session=settings.max_files_per_session,
        temperature_supported=settings.llm_supports_temperature,
    )
