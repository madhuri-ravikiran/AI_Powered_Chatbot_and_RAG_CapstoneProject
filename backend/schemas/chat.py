"""Validated chat requests, real retrieval sources, and generated answers."""

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from backend.schemas.common import SessionRequest


class ChatRequest(SessionRequest):
    """Validate message length, mode, and retrieval/generation settings."""

    message: str = Field(min_length=1, max_length=4000)
    mode: Literal["general", "rag"] = "general"
    top_k: int | None = Field(default=None, ge=1, le=10)
    temperature: float = Field(default=0.2, ge=0, le=1)

    @field_validator("message")
    @classmethod
    def nonempty_message(cls, value: str) -> str:
        """Reject whitespace-only messages before invoking a provider."""
        if not value.strip():
            raise ValueError("Message must not be blank")
        return value.strip()


class Source(BaseModel):
    """Describe a retrieved chunk whose evidence was used in the answer."""

    file_name: str
    page: int | None = None
    chunk_id: str
    excerpt: str
    similarity: float


class ChatResponse(BaseModel):
    """Return the generated answer, citations, and measured wall-clock latency."""

    answer: str
    session_id: str
    mode: Literal["general", "rag"]
    sources: list[Source] = Field(default_factory=list)
    processing_time_ms: float


class Evidence(BaseModel):
    """Require a verbatim quotation tied to a retrieved chunk identifier."""

    chunk_id: str
    quote: str = Field(min_length=1, max_length=1500)


class GroundedAnswer(BaseModel):
    """Internal JSON output contract validated before exposing RAG citations."""

    answer: str = Field(min_length=1, max_length=12000)
    evidence: list[Evidence] = Field(default_factory=list, max_length=10)
