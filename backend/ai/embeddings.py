"""Construct configurable embedding providers independently of the chat model."""

from langchain_core.embeddings import Embeddings

from backend.core.config import Settings
from backend.core.exceptions import AppError


def create_embeddings(settings: Settings) -> Embeddings:
    """Create the embedding adapter; document/query calls use the same model."""
    if settings.embedding_provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        key = settings.embedding_api_key.get_secret_value()
        if not key and settings.llm_provider == "openai":
            key = settings.llm_api_key.get_secret_value()
        if not key:
            raise AppError(503, "missing_embedding_key", "Set EMBEDDING_API_KEY in .env.")
        return OpenAIEmbeddings(
            model=settings.embedding_model,
            api_key=key,
            request_timeout=settings.provider_timeout_seconds,
            max_retries=0,
        )
    from langchain_ollama import OllamaEmbeddings

    return OllamaEmbeddings(
        model=settings.embedding_model,
        base_url=settings.ollama_base_url,
        client_kwargs={"timeout": settings.provider_timeout_seconds},
    )
