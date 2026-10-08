"""Provide atomic FAISS indexing and cosine retrieval for LangChain documents.

Normalized inner product equals cosine similarity. Index updates are staged
before publication so a failed embedding batch cannot damage existing data.
"""

import logging

import faiss
import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from backend.core.exceptions import AppError, provider_call

logger = logging.getLogger(__name__)


class VectorStore:
    """Keep vectors and their corresponding metadata in one session-local object."""

    def __init__(self, embeddings: Embeddings):
        """Store an embedding adapter; initialize the FAISS dimension lazily."""
        self.embeddings = embeddings
        self.index = None
        self.documents: list[Document] = []

    @staticmethod
    def _normalize(vectors: list[list[float]]) -> np.ndarray:
        """Reject malformed provider vectors and normalize valid float32 rows."""
        matrix = np.asarray(vectors, dtype=np.float32)
        if (
            matrix.ndim != 2
            or matrix.shape[1] == 0
            or not np.isfinite(matrix).all()
            or (np.linalg.norm(matrix, axis=1) == 0).any()
        ):
            raise AppError(
                502, "invalid_embeddings", "The embedding provider returned invalid vectors."
            )
        matrix = np.ascontiguousarray(matrix)
        faiss.normalize_L2(matrix)
        return matrix

    def add_documents(self, documents: list[Document]) -> None:
        """Embed only new chunks, then atomically replace the staged FAISS index."""
        if not documents:
            return
        logger.info("Embedding generation started: chunks=%d", len(documents))
        vectors = provider_call(
            "embedding",
            lambda: self.embeddings.embed_documents(
                [document.page_content for document in documents]
            ),
        )
        matrix = self._normalize(vectors)
        if len(matrix) != len(documents):
            raise AppError(502, "invalid_embeddings", "Embedding count did not match chunk count.")
        try:
            if self.index is None:
                staged = faiss.IndexFlatIP(matrix.shape[1])
            else:
                if self.index.d != matrix.shape[1]:
                    raise AppError(
                        409, "embedding_dimension", "Clear the index after changing models."
                    )
                staged = faiss.clone_index(self.index)
            staged.add(matrix)
        except AppError:
            raise
        except Exception:
            raise AppError(
                500, "vector_store_error", "Unable to update the vector index."
            ) from None
        self.documents = self.documents + documents
        self.index = staged
        logger.info(
            "Embedding generation completed; vector store updated: total=%d", len(self.documents)
        )

    def search(self, query: str, top_k: int, min_similarity: float) -> list[tuple[Document, float]]:
        """Embed the query once and retrieve relevant chunks without re-embedding documents."""
        if self.index is None:
            return []
        vector = provider_call("embedding", lambda: self.embeddings.embed_query(query))
        matrix = self._normalize([vector])
        if matrix.shape[1] != self.index.d:
            raise AppError(409, "embedding_dimension", "Clear the index after changing models.")
        try:
            scores, positions = self.index.search(matrix, min(top_k, len(self.documents)))
        except Exception:
            raise AppError(
                500, "vector_store_error", "Unable to search the vector index."
            ) from None
        return [
            (self.documents[int(position)], float(np.clip(score, -1, 1)))
            for score, position in zip(scores[0], positions[0], strict=True)
            if position >= 0 and score >= min_similarity
        ]

    def clear(self) -> None:
        """Discard vectors and source documents without loading serialized objects."""
        self.index = None
        self.documents = []
