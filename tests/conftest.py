"""Supply deterministic test-only models and embeddings; never call paid APIs."""

import re

import pytest
from fastapi.testclient import TestClient
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from backend.core.config import Settings
from backend.main import create_app


class KeywordEmbeddings(Embeddings):
    """Map topic words to small vectors to test real FAISS operations offline."""

    vocabulary = ("leave", "remote", "expense", "password", "training", "ceo", "join", "salary")

    def __init__(self):
        """Track document/query embedding counts to detect redundant embedding work."""
        self.document_calls = 0
        self.query_calls = 0

    def vector(self, text: str) -> list[float]:
        """Count fixed topic stems and add a small nonzero component."""
        words = re.findall(r"\w+", text.lower())
        return [
            float(sum(word.startswith(topic) for word in words)) for topic in self.vocabulary
        ] + [0.01]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return deterministic document vectors and record the batch call."""
        self.document_calls += 1
        return [self.vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        """Return the same vector space for the search query."""
        self.query_calls += 1
        return self.vector(text)


@pytest.fixture
def settings():
    """Ignore environment files and explicitly remove provider credentials in tests."""
    return Settings(
        _env_file=None,
        llm_api_key="",
        embedding_api_key="",
        chunk_size=150,
        chunk_overlap=20,
        min_similarity=0.2,
    )


@pytest.fixture
def embeddings():
    """Provide a fresh counter-bearing offline embedding adapter."""
    return KeywordEmbeddings()


@pytest.fixture
def llm():
    """Provide a predictable model whose responses individual tests can replace."""
    return FakeListChatModel(responses=["A helpful general answer."])


@pytest.fixture
def app(settings, embeddings, llm):
    """Construct an isolated app instance with fake providers and real services."""
    return create_app(settings, lambda temperature: llm, lambda: embeddings)


@pytest.fixture
def client(app):
    """Run TestClient with lifespan startup/shutdown enabled."""
    with TestClient(app) as instance:
        yield instance


def upload_and_process(client, session_id="test-session", text="Annual leave is 24 days."):
    """Create a processed knowledge base through public HTTP endpoints."""
    uploaded = client.post(
        "/api/documents/upload",
        data={"session_id": session_id},
        files={"files": ("policy.txt", text.encode(), "text/plain")},
    )
    assert uploaded.status_code == 201, uploaded.text
    processed = client.post("/api/documents/process", json={"session_id": session_id})
    assert processed.status_code == 200, processed.text
    return processed.json()
