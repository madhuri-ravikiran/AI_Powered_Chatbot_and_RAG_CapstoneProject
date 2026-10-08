"""Common response models and reusable session identifier validation."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

SessionId = Annotated[str, Field(min_length=1, max_length=80, pattern=r"^[a-zA-Z0-9_-]+$")]


class StrictModel(BaseModel):
    """Reject misspelled request options instead of silently ignoring them."""

    model_config = ConfigDict(extra="forbid")


class SessionRequest(StrictModel):
    """Identify an isolated conversation and its document collection."""

    session_id: SessionId


class MessageResponse(BaseModel):
    """Acknowledge a successful state-changing operation."""

    message: str


class HealthResponse(BaseModel):
    """Report process liveness, independent of provider connectivity."""

    status: str = "healthy"
    application: str
    version: str


class ErrorResponse(BaseModel):
    """Return a stable machine code and a safe explanation."""

    code: str
    message: str


class ConfigResponse(BaseModel):
    """Expose only non-secret settings needed by the frontend."""

    application: str
    version: str
    llm_provider: str
    llm_model: str
    embedding_provider: str
    embedding_model: str
    chat_configured: bool
    embeddings_configured: bool
    top_k: int
    max_upload_mb: int
    max_files_per_session: int
    temperature_supported: bool
