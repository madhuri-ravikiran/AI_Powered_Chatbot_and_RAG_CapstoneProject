"""Load and validate environment settings without initializing remote providers.

Settings is injected into services so tests can run without credentials or .env.
"""

from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Central configuration; secrets are excluded from the public API."""

    model_config = SettingsConfigDict(
        env_file=ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )
    app_name: str = "IntelliChat"
    app_version: str = "1.0.0"
    llm_provider: Literal["openai", "ollama"] = "openai"
    llm_api_key: SecretStr = SecretStr("")
    llm_model: str = "gpt-4.1-mini"
    embedding_provider: Literal["openai", "ollama"] = "openai"
    embedding_api_key: SecretStr = SecretStr("")
    embedding_model: str = "text-embedding-3-small"
    ollama_base_url: str = "http://127.0.0.1:11434"
    llm_supports_temperature: bool = True
    provider_timeout_seconds: float = Field(default=60, ge=1, le=300)
    max_output_tokens: int = Field(default=1200, ge=100, le=8000)
    chunk_size: int = Field(default=1000, ge=100, le=4000)
    chunk_overlap: int = Field(default=150, ge=0)
    top_k: int = Field(default=4, ge=1, le=10)
    min_similarity: float = Field(default=0.2, ge=-1, le=1)
    max_upload_mb: int = Field(default=10, ge=1, le=25)
    max_files_per_session: int = Field(default=20, ge=1, le=50)
    max_extracted_chars: int = Field(default=500000, ge=1000, le=2000000)
    max_history_turns: int = Field(default=10, ge=1, le=25)
    max_sessions: int = Field(default=32, ge=1, le=100)
    session_ttl_seconds: int = Field(default=3600, ge=60)
    fastapi_host: str = "127.0.0.1"
    fastapi_port: int = Field(default=8000, ge=1, le=65535)
    backend_url: str = "http://127.0.0.1:8000"
    cors_origins: list[str] = ["http://localhost:8501", "http://127.0.0.1:8501"]
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    @model_validator(mode="after")
    def check_chunk_overlap(self) -> "Settings":
        """Reject invalid overlap before the splitter is constructed."""
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        if not self.llm_model.strip() or not self.embedding_model.strip():
            raise ValueError("Model names must not be blank")
        return self
