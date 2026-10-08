"""Retrieve bounded, relevance-filtered context from a session's FAISS store."""

import logging

from langchain_core.documents import Document

from backend.ai.vector_store import VectorStore

logger = logging.getLogger(__name__)


def retrieve(
    store: VectorStore, query: str, top_k: int, threshold: float
) -> list[tuple[Document, float]]:
    """Return at most top_k real chunks meeting a cosine-similarity threshold."""
    results = store.search(query, top_k, threshold)
    logger.info("Retrieval completed: chunks=%d", len(results))
    return results
