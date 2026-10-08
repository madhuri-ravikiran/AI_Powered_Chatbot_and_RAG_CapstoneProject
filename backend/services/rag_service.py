"""Coordinate contextual retrieval, grounded generation, and verified citations."""

import re

from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable

from backend.ai.prompts import FALLBACK
from backend.ai.rag_chain import generate_grounded_answer, rewrite_query
from backend.ai.retriever import retrieve
from backend.ai.vector_store import VectorStore
from backend.core.config import Settings
from backend.schemas.chat import Source


def normalized(text: str) -> str:
    """Collapse extraction whitespace for quote matching without inventing content."""
    return re.sub(r"\s+", " ", text).strip()


class RagService:
    """Execute the RAG workflow independently from HTTP and session bookkeeping."""

    def __init__(self, settings: Settings):
        """Retain retrieval configuration for every question."""
        self.settings = settings

    def answer(
        self,
        llm: Runnable,
        question: str,
        history: list[BaseMessage],
        store: VectorStore,
        top_k: int,
    ) -> tuple[str, list[Source]]:
        """Rewrite, retrieve, generate, and reject unverifiable answer provenance."""
        query = rewrite_query(llm, question, history)
        matches = retrieve(store, query, top_k, self.settings.min_similarity)
        if not matches:
            return FALLBACK, []
        generated = generate_grounded_answer(
            llm, question, query, history, [document for document, _ in matches]
        )
        if generated.answer.strip() == FALLBACK or not generated.evidence:
            return FALLBACK, []
        lookup = {doc.metadata["chunk_id"]: (doc, score) for doc, score in matches}
        sources = []
        seen = set()
        for evidence in generated.evidence:
            match = lookup.get(evidence.chunk_id)
            if (
                not match
                or not normalized(evidence.quote)
                or normalized(evidence.quote) not in normalized(match[0].page_content)
            ):
                # Fail closed: a fabricated identifier or quote invalidates the answer.
                return FALLBACK, []
            if evidence.chunk_id not in seen:
                document, score = match
                sources.append(
                    Source(
                        file_name=document.metadata["file_name"],
                        page=document.metadata.get("page"),
                        chunk_id=evidence.chunk_id,
                        excerpt=evidence.quote,
                        similarity=round(score, 4),
                    )
                )
                seen.add(evidence.chunk_id)
        return generated.answer, sources
