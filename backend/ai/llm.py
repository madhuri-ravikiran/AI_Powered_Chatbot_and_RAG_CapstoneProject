"""Isolate model construction for OpenAI and local Ollama providers."""

from langchain_core.language_models import BaseChatModel

from backend.core.config import Settings
from backend.core.exceptions import AppError


def create_llm(settings: Settings, temperature: float) -> BaseChatModel:
    """Create a bounded-time chat model; no network call happens until invoke."""
    if settings.llm_provider == "openai":
        from langchain_openai import ChatOpenAI

        key = settings.llm_api_key.get_secret_value()
        if not key:
            raise AppError(503, "missing_api_key", "Set LLM_API_KEY in the backend .env file.")
        return ChatOpenAI(
            model=settings.llm_model,
            api_key=key,
            temperature=temperature if settings.llm_supports_temperature else None,
            timeout=settings.provider_timeout_seconds,
            max_retries=0,
            max_completion_tokens=settings.max_output_tokens,
        )
    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=settings.llm_model,
        base_url=settings.ollama_base_url,
        temperature=temperature,
        num_predict=settings.max_output_tokens,
        client_kwargs={"timeout": settings.provider_timeout_seconds},
    )
