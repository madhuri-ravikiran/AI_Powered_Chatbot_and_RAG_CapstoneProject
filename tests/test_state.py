"""Verify bounded session lifetime, capacity, and parallel-request behavior."""

from concurrent.futures import ThreadPoolExecutor

import pytest
from langchain_core.messages import HumanMessage

from backend.core.exceptions import AppError
from backend.core.state import SessionRegistry


def test_session_capacity_and_expiry(settings):
    """Expire idle state before admitting a new session at capacity."""
    settings.max_sessions = 1
    registry = SessionRegistry(settings)
    with registry.session("one") as state:
        state.history["general"].append(HumanMessage(content="Hello"))
    with pytest.raises(AppError) as error:
        with registry.session("two"):
            pass
    assert error.value.code == "session_capacity"
    registry.sessions["one"].touched -= settings.session_ttl_seconds + 1
    with registry.session("two"):
        assert "one" not in registry.sessions


def test_same_session_concurrency_returns_conflict(settings):
    """Reject concurrent mutation while allowing unrelated sessions to proceed."""
    registry = SessionRegistry(settings)

    def use_session(identifier):
        """Attempt a lease from another thread."""
        try:
            with registry.session(identifier):
                return "ok"
        except AppError as error:
            return error.code

    with registry.session("busy"), ThreadPoolExecutor(max_workers=1) as pool:
        assert pool.submit(use_session, "busy").result() == "session_busy"
        assert pool.submit(use_session, "other").result() == "ok"


def test_cleared_sessions_release_capacity(settings):
    """New empty chats should not consume capacity until the idle TTL elapses."""
    settings.max_sessions = 1
    registry = SessionRegistry(settings)
    for index in range(5):
        with registry.session(f"chat-{index}"):
            assert len(registry.sessions) == 1
