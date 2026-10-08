"""Bound in-memory sessions and serialize each session's state changes.

The registry holds a short lock for lookups; unrelated sessions can run in
parallel. Idle sessions expire. Active sessions are never evicted mid-request.
"""

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from threading import Lock, RLock

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage

from backend.ai.vector_store import VectorStore
from backend.core.config import Settings
from backend.core.exceptions import AppError
from backend.schemas.documents import DocumentInfo


@dataclass
class StoredDocument:
    """Store extracted content and public metadata, without retaining binary uploads."""

    info: DocumentInfo
    pages: list[Document]


@dataclass
class SessionState:
    """Own one session's documents, mode-specific memory, and vector index."""

    documents: dict[str, StoredDocument] = field(default_factory=dict)
    history: dict[str, list[BaseMessage]] = field(
        default_factory=lambda: {"general": [], "rag": []}
    )
    vector_store: VectorStore | None = None
    touched: float = field(default_factory=time.monotonic)
    lock: RLock = field(default_factory=RLock)


class SessionRegistry:
    """Limit memory lifetime and isolate session operations with per-session locks."""

    def __init__(self, settings: Settings):
        """Initialize an empty, dependency-injectable session registry."""
        self.settings = settings
        self.sessions: dict[str, SessionState] = {}
        self.lock = Lock()

    @contextmanager
    def session(self, session_id: str) -> Iterator[SessionState]:
        """Lease a session; reject concurrent same-session operations with HTTP 409."""
        with self.lock:
            now = time.monotonic()
            for key, state in list(self.sessions.items()):
                if state.lock.acquire(blocking=False):
                    try:
                        expired = now - state.touched > self.settings.session_ttl_seconds
                        empty = not state.documents and not any(state.history.values())
                        # Reclaim cleared chats immediately; do not exhaust capacity on New Chat.
                        if expired or (key != session_id and empty):
                            del self.sessions[key]
                    finally:
                        state.lock.release()
            if session_id not in self.sessions:
                if len(self.sessions) >= self.settings.max_sessions:
                    raise AppError(
                        503, "session_capacity", "Session capacity reached. Try again later."
                    )
                self.sessions[session_id] = SessionState()
            state = self.sessions[session_id]
            if not state.lock.acquire(blocking=False):
                raise AppError(
                    409, "session_busy", "This conversation is busy. Please retry shortly."
                )
            state.touched = now
        try:
            yield state
        finally:
            state.touched = time.monotonic()
            state.lock.release()
