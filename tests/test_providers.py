"""Construct real provider adapters without invoking network or model inference."""

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import SecretStr

from backend.ai.embeddings import create_embeddings
from backend.ai.llm import create_llm


def test_openai_construction(settings):
    """Verify installed OpenAI adapter imports and constructor parameters."""
    settings.llm_api_key = SecretStr("test-placeholder-not-a-real-key")
    llm = create_llm(settings, 0.2)
    embeddings = create_embeddings(settings)
    assert isinstance(llm, ChatOpenAI)
    assert isinstance(embeddings, OpenAIEmbeddings)
    assert llm.model_name == settings.llm_model


def test_ollama_construction(settings):
    """Verify local-provider construction using its dedicated integration package."""
    settings.llm_provider = "ollama"
    settings.embedding_provider = "ollama"
    settings.llm_model = "llama3.2"
    settings.embedding_model = "nomic-embed-text"
    assert isinstance(create_llm(settings, 0.2), ChatOllama)
    assert isinstance(create_embeddings(settings), OllamaEmbeddings)
