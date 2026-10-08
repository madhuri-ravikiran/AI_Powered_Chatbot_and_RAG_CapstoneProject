"""Test general conversation memory, mode separation, and failure semantics."""

import httpx
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from openai import AuthenticationError, RateLimitError

from tests.conftest import upload_and_process


def test_general_followup_and_clear(client, app):
    """Pass earlier turns to the LLM and clear memory without removing documents."""
    seen = []

    def reply(prompt):
        """Capture the complete prompt used for general chat."""
        seen.append(prompt.to_messages())
        return AIMessage(content="AI applications include language translation.")

    app.state.chat_service.llm_factory = lambda temperature: RunnableLambda(reply)
    for question in ("What is AI?", "What are its applications?"):
        assert (
            client.post("/api/chat", json={"session_id": "s", "message": question}).status_code
            == 200
        )
    assert any(message.content == "What is AI?" for message in seen[1])
    assert len(app.state.registry.sessions["s"].history["general"]) == 4
    upload_and_process(client, "s")
    assert client.post("/api/chat/clear", json={"session_id": "s"}).status_code == 200
    assert app.state.registry.sessions["s"].history["general"] == []
    assert app.state.registry.sessions["s"].vector_store is not None


def test_failed_call_does_not_commit_history(client, app):
    """Do not save partial turns or leak private provider exceptions."""

    def fail(prompt):
        """Simulate an unavailable provider."""
        raise RuntimeError("private content and secret")

    app.state.chat_service.llm_factory = lambda temperature: RunnableLambda(fail)
    response = client.post("/api/chat", json={"session_id": "s", "message": "Hello"})
    assert response.status_code == 502
    assert "private" not in response.text
    assert app.state.registry.sessions["s"].history["general"] == []


def test_history_is_bounded_and_modes_are_separate(client, app, settings):
    """Trim complete pairs while avoiding unsupported general-chat context in RAG."""
    settings.max_history_turns = 2
    for _ in range(4):
        assert (
            client.post("/api/chat", json={"session_id": "s", "message": "Hello"}).status_code
            == 200
        )
    assert len(app.state.registry.sessions["s"].history["general"]) == 4
    assert app.state.registry.sessions["s"].history["rag"] == []


def test_provider_auth_rate_and_timeout(client, app):
    """Map credential, rate limit, and timeout failures to actionable HTTP status codes."""
    response = httpx.Response(401, request=httpx.Request("POST", "https://example.test"))
    errors = [
        (AuthenticationError("secret", response=response, body={}), 503, "provider_auth"),
        (RateLimitError("secret", response=response, body={}), 429, "provider_rate_limit"),
        (httpx.ReadTimeout("secret"), 504, "provider_timeout"),
    ]
    for error, status, code in errors:

        def fail(prompt, failure=error):
            """Raise the test provider failure."""
            raise failure

        app.state.chat_service.llm_factory = lambda temperature: RunnableLambda(fail)
        result = client.post("/api/chat", json={"session_id": "s", "message": "Hello"})
        assert result.status_code == status
        assert result.json()["code"] == code
        assert "secret" not in result.text


def test_failed_processing_remains_retryable(client, app, embeddings, monkeypatch):
    """An embedding failure leaves documents uploaded and allows a later retry."""
    client.post(
        "/api/documents/upload",
        data={"session_id": "s"},
        files={"files": ("policy.txt", b"Annual leave is 24 days.")},
    )
    original = embeddings.embed_documents

    def fail(texts):
        """Raise a simulated embedding outage."""
        raise httpx.ReadTimeout("network down")

    monkeypatch.setattr(embeddings, "embed_documents", fail)
    failed = client.post("/api/documents/process", json={"session_id": "s"})
    assert failed.status_code == 504
    assert app.state.registry.sessions["s"].vector_store is None
    docs = client.get("/api/documents", params={"session_id": "s"}).json()["documents"]
    assert docs[0]["status"] == "uploaded"
    monkeypatch.setattr(embeddings, "embed_documents", original)
    assert client.post("/api/documents/process", json={"session_id": "s"}).status_code == 200
