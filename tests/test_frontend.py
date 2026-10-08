"""Validate the reusable HTTP client and Streamlit UI without a browser or paid API."""

from pathlib import Path

import httpx
import pytest
from streamlit.testing.v1 import AppTest

from frontend.api_client import UNAVAILABLE, ApiClient, ApiError


def test_http_client_against_real_api(client):
    """Use the frontend adapter against real FastAPI routes and services."""
    api = ApiClient("http://testserver", client=client)
    assert api.health_check()["status"] == "healthy"
    assert api.get_config()["application"] == "IntelliChat"
    api.upload_documents("s", [("policy.txt", b"Annual leave is 24 days.")])
    assert api.process_documents("s")["total_chunks"] > 0
    assert api.get_documents("s")["documents"][0]["status"] == "processed"
    assert api.send_chat_message("s", "Hello", "general", 4, 0.2)["answer"]
    api.clear_chat("s")
    api.clear_documents("s")
    assert api.get_documents("s")["documents"] == []


@pytest.mark.parametrize(
    ("handler", "expected"),
    [
        (
            lambda request: httpx.Response(502, json={"code": "error", "message": "Try later."}),
            "Try later.",
        ),
        (lambda request: httpx.Response(200, text="<html>bad gateway</html>"), "unreadable"),
        (lambda request: httpx.Response(200, json=[]), "unexpected"),
    ],
)
def test_frontend_response_errors(handler, expected):
    """Render safe errors for backend failures and malformed responses."""
    with httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test") as client:
        api = ApiClient("http://test", client=client)
        with pytest.raises(ApiError, match=expected):
            api.health_check()


def test_frontend_backend_unavailable():
    """Show the required friendly message when no backend is reachable."""

    def offline(request):
        """Simulate a disconnected backend."""
        raise httpx.ConnectError("connection refused")

    with httpx.Client(transport=httpx.MockTransport(offline), base_url="http://test") as client:
        with pytest.raises(ApiError, match=UNAVAILABLE):
            ApiClient("http://test", client=client).health_check()


def test_streamlit_initial_and_general_chat(client, monkeypatch):
    """Render the actual UI and submit a chat via the actual frontend HTTP adapter."""
    config = client.get("/api/config").json()
    config["chat_configured"] = True  # This app instance uses a test-only fake LLM.
    monkeypatch.setattr(ApiClient, "health_check", lambda self: client.get("/health").json())
    monkeypatch.setattr(ApiClient, "get_config", lambda self: config)
    monkeypatch.setattr(
        ApiClient,
        "send_chat_message",
        lambda self, sid, msg, mode, k, temp: client.post(
            "/api/chat",
            json={
                "session_id": sid,
                "message": msg,
                "mode": mode,
                "top_k": k,
                "temperature": temp,
            },
        ).json(),
    )
    ui = AppTest.from_file(
        Path(__file__).resolve().parents[1] / "frontend/app.py", default_timeout=20
    ).run()
    assert not ui.exception
    assert any("IntelliChat" in item.value for item in ui.title)
    ui.chat_input[0].set_value("Explain AI").run()
    assert not ui.exception
    assert len(ui.chat_message) == 2
    assert "A helpful general answer." in [item.value for item in ui.markdown]


def test_streamlit_offline_state(monkeypatch):
    """An offline backend produces a friendly UI error and a retry control."""

    def offline(self):
        """Replace the network with the same error produced by ApiClient."""
        raise ApiError(UNAVAILABLE)

    monkeypatch.setattr(ApiClient, "health_check", offline)
    ui = AppTest.from_file(
        Path(__file__).resolve().parents[1] / "frontend/app.py", default_timeout=20
    ).run()
    assert not ui.exception
    assert ui.error[0].value == UNAVAILABLE
    assert ui.button[0].label == "Retry connection"


def test_streamlit_rag_sources_and_clear_controls(client, app, llm, monkeypatch):
    """Exercise document chat, source rendering, and clearing through the HTTP adapter."""
    from pydantic import SecretStr

    from tests.conftest import upload_and_process
    from tests.test_rag import grounded_response

    app.state.settings.llm_api_key = SecretStr("test-placeholder-not-a-real-key")
    processed = upload_and_process(client)
    llm.responses = [grounded_response(app)]

    def use_test_transport(self, base_url, **kwargs):
        """Connect the actual frontend adapter to the real test backend."""
        self.client = client

    monkeypatch.setattr(ApiClient, "__init__", use_test_transport)
    ui = AppTest.from_file(
        Path(__file__).resolve().parents[1] / "frontend/app.py", default_timeout=20
    ).run()
    ui.session_state["session_id"] = "test-session"
    ui.session_state["documents"] = processed["documents"]
    ui.radio[0].set_value("Document / RAG Chat").run()
    ui.chat_input[0].set_value("How much annual leave?").run()
    assert not ui.exception
    assert any("Sources" in expander.label for expander in ui.expander)
    assert any(text.value == "Annual leave is 24 days." for text in ui.text)
    next(button for button in ui.button if button.label == "Clear Conversation").click().run()
    assert not ui.exception
    assert len(ui.chat_message) == 0
    assert app.state.registry.sessions["test-session"].vector_store is not None
    next(button for button in ui.button if button.label == "Clear Knowledge Base").click().run()
    assert not ui.exception
    assert app.state.registry.sessions["test-session"].vector_store is None
    assert ui.chat_input[0].disabled
    next(button for button in ui.button if button.label == "New Chat").click().run()
    assert not ui.exception
    assert ui.session_state["session_id"] != "test-session"
