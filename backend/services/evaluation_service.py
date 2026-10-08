"""Calculate transparent lexical evaluation proxies from real RAG responses.

These metrics assist a capstone demonstration; they are not semantic correctness
or entailment proofs. Human review must verify each factual claim.
"""

import re
from collections import Counter

from pydantic import BaseModel

from backend.schemas.chat import ChatRequest
from backend.services.chat_service import ChatService


class EvaluationCase(BaseModel):
    """Define an expected answer and source for one independently evaluated question."""

    question: str
    expected_answer: str
    expected_source: str


def tokens(text: str) -> list[str]:
    """Normalize text into case-insensitive word tokens for reproducible overlap metrics."""
    return re.findall(r"\b\w+\b", text.lower())


def token_f1(expected: str, actual: str) -> float:
    """Measure multiset token overlap; this is a lexical correctness proxy only."""
    reference, candidate = Counter(tokens(expected)), Counter(tokens(actual))
    overlap = sum((reference & candidate).values())
    if not overlap:
        return 0.0
    precision = overlap / sum(candidate.values())
    recall = overlap / sum(reference.values())
    return 2 * precision * recall / (precision + recall)


class EvaluationService:
    """Run isolated questions and report calculated source/answer/evidence proxies."""

    def __init__(self, chat_service: ChatService):
        """Use the same chat path that serves API requests."""
        self.chat_service = chat_service

    def evaluate(self, session_id: str, cases: list[EvaluationCase]) -> dict:
        """Evaluate against a processed KB, clearing history between independent cases."""
        results = []
        for case in cases:
            self.chat_service.clear(session_id)
            response = self.chat_service.chat(
                ChatRequest(session_id=session_id, message=case.question, mode="rag", temperature=0)
            )
            # Actual retrieval metrics inspect search results, not only model-selected citations.
            with self.chat_service.registry.session(session_id) as state:
                matches = state.vector_store.search(
                    case.question,
                    self.chat_service.settings.top_k,
                    self.chat_service.settings.min_similarity,
                )
            relevant = sum(doc.metadata["file_name"] == case.expected_source for doc, _ in matches)
            answer_tokens = set(tokens(response.answer))
            evidence_tokens = set(tokens(" ".join(source.excerpt for source in response.sources)))
            results.append(
                {
                    **case.model_dump(),
                    "actual_answer": response.answer,
                    "retrieval_source_hit_at_k": bool(relevant),
                    "retrieval_source_precision_at_k": relevant / len(matches) if matches else 0.0,
                    "answer_token_f1": token_f1(case.expected_answer, response.answer),
                    "evidence_token_coverage": (
                        len(answer_tokens & evidence_tokens) / len(answer_tokens)
                        if answer_tokens
                        else 0.0
                    ),
                    "processing_time_ms": response.processing_time_ms,
                    "sources": [source.model_dump() for source in response.sources],
                    "human_correctness": None,
                    "human_groundedness": None,
                }
            )
        metrics = (
            "retrieval_source_hit_at_k",
            "retrieval_source_precision_at_k",
            "answer_token_f1",
            "evidence_token_coverage",
            "processing_time_ms",
        )
        return {
            "notice": (
                "Lexical/source proxies only. Human semantic and groundedness review is required."
            ),
            "cases": results,
            "averages": {
                metric: sum(row[metric] for row in results) / len(results) if results else None
                for metric in metrics
            },
        }
