"""Test real FAISS creation, search, incremental updates, and failure atomicity."""

import pytest
from langchain_core.documents import Document

from backend.ai.vector_store import VectorStore
from backend.core.exceptions import AppError


def test_cosine_retrieval_and_reuse(embeddings):
    """Query vectors match the expected topic without re-embedding stored documents."""
    store = VectorStore(embeddings)
    store.add_documents(
        [
            Document(page_content="Annual leave: 24 days.", metadata={"source": "leave.txt"}),
            Document(page_content="Remote work: two days.", metadata={"source": "remote.txt"}),
        ]
    )
    result = store.search("What is the remote policy?", 1, 0.2)
    assert result[0][0].metadata["source"] == "remote.txt"
    assert 0.99 <= result[0][1] <= 1
    store.search("Leave policy?", 2, 0.2)
    assert embeddings.document_calls == 1
    assert embeddings.query_calls == 2
    store.add_documents([Document(page_content="Password policy")])
    assert len(store.documents) == 3
    store.clear()
    assert store.search("Leave", 4, 0.2) == []


def test_embedding_failure_preserves_index(embeddings, monkeypatch):
    """A failed second batch cannot corrupt the existing index."""
    store = VectorStore(embeddings)
    store.add_documents([Document(page_content="Annual leave")])

    def fail(texts):
        """Simulate a private provider failure without contacting a provider."""
        raise RuntimeError("private provider details")

    monkeypatch.setattr(embeddings, "embed_documents", fail)
    with pytest.raises(AppError):
        store.add_documents([Document(page_content="Remote")])
    assert len(store.documents) == 1
    assert store.index.ntotal == 1


def test_invalid_vectors(embeddings, monkeypatch):
    """Reject nonfinite vectors before handing them to FAISS."""
    store = VectorStore(embeddings)
    monkeypatch.setattr(embeddings, "embed_documents", lambda texts: [[float("nan")]])
    with pytest.raises(AppError):
        store.add_documents([Document(page_content="Annual leave")])
    assert store.index is None
