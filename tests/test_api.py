"""Exercise API validation, isolation, document lifecycle, and safe error contracts."""

import pytest
from fastapi.testclient import TestClient

from backend.main import create_app
from tests.conftest import upload_and_process


def test_health_docs_and_config(client):
    """Verify public API documentation and ensure secrets are never returned."""
    assert client.get("/health").json() == {
        "status": "healthy",
        "application": "IntelliChat",
        "version": "1.0.0",
    }
    assert client.get("/").status_code == 200
    assert client.get("/docs").status_code == 200
    assert client.get("/redoc").status_code == 200
    assert client.get("/openapi.json").json()["info"]["title"] == "AI-Powered Chatbot"
    config = client.get("/api/config").json()
    assert "llm_api_key" not in config
    assert config["chat_configured"] is False


@pytest.mark.parametrize(
    "override",
    [
        {"message": ""},
        {"message": "   "},
        {"message": "x" * 4001},
        {"mode": "unknown"},
        {"top_k": 0},
        {"top_k": 11},
        {"temperature": 1.1},
        {"session_id": "../unsafe"},
        {"session_id": ""},
        {"unknown": "field"},
    ],
)
def test_invalid_requests(client, override):
    """Reject invalid inputs before either embedding or generation occurs."""
    response = client.post(
        "/api/chat",
        json={
            "session_id": "test-session",
            "message": "Hello",
            **override,
        },
    )
    assert response.status_code == 422
    assert response.json()["code"] == "invalid_request"


def test_document_lifecycle_and_no_reembedding(client, embeddings):
    """Upload/process/list/clear work and processing is idempotent."""
    first = upload_and_process(client)
    assert first["total_chunks"] >= 1
    again = client.post("/api/documents/process", json={"session_id": "test-session"}).json()
    assert again["added_chunks"] == 0
    assert embeddings.document_calls == 1
    listed = client.get("/api/documents", params={"session_id": "test-session"}).json()
    assert listed["documents"][0]["status"] == "processed"
    assert client.delete("/api/documents", params={"session_id": "test-session"}).status_code == 200
    assert (
        client.get("/api/documents", params={"session_id": "test-session"}).json()["documents"]
        == []
    )


def test_missing_knowledge_and_session_isolation(client):
    """A second session cannot read or query the first session's documents."""
    upload_and_process(client, "alice")
    assert client.get("/api/documents", params={"session_id": "bob"}).json()["documents"] == []
    response = client.post(
        "/api/chat", json={"session_id": "bob", "mode": "rag", "message": "Leave?"}
    )
    assert response.status_code == 409
    assert response.json()["code"] == "missing_knowledge_base"


def test_upload_batch_is_atomic(client):
    """A corrupt second file must not leave a partially uploaded first file."""
    response = client.post(
        "/api/documents/upload",
        data={"session_id": "s"},
        files=[
            ("files", ("good.txt", b"Annual leave is 24 days.")),
            ("files", ("bad.pdf", b"not a pdf")),
        ],
    )
    assert response.status_code == 422
    assert client.get("/api/documents", params={"session_id": "s"}).json()["documents"] == []


def test_duplicate_and_conflicting_name(client):
    """Identical retries are safe; changed content under the same name is explicit."""
    for _ in range(2):
        response = client.post(
            "/api/documents/upload",
            data={"session_id": "s"},
            files={"files": ("policy.txt", b"Annual leave.")},
        )
        assert response.status_code == 201
        assert len(response.json()["documents"]) == 1
    conflict = client.post(
        "/api/documents/upload",
        data={"session_id": "s"},
        files={"files": ("policy.txt", b"Changed annual leave.")},
    )
    assert conflict.status_code == 409


def test_missing_key_is_safe(settings):
    """A real provider factory without a key must fail clearly, without a network call."""
    with TestClient(create_app(settings)) as client:
        response = client.post("/api/chat", json={"session_id": "s", "message": "Hello"})
    assert response.status_code == 503
    assert response.json()["code"] == "missing_api_key"


def test_cors_restricts_origins(client):
    """Only explicit local Streamlit origins receive CORS permission."""
    allowed = client.options(
        "/api/chat",
        headers={
            "Origin": "http://localhost:8501",
            "Access-Control-Request-Method": "POST",
        },
    )
    denied = client.options(
        "/api/chat",
        headers={
            "Origin": "https://untrusted.example",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert allowed.headers["access-control-allow-origin"] == "http://localhost:8501"
    assert "access-control-allow-origin" not in denied.headers


def test_upload_size_limit(client, settings):
    """Cap oversized content and return an actionable error."""
    settings.max_upload_mb = 1
    response = client.post(
        "/api/documents/upload",
        data={"session_id": "s"},
        files={"files": ("large.txt", b"x" * (1024 * 1024 + 1))},
    )
    assert response.status_code == 413


def test_invalid_upload_and_query_session_ids(client):
    """Validate IDs consistently for multipart and query-string endpoints."""
    assert client.get("/api/documents", params={"session_id": "../bad"}).status_code == 422
    response = client.post(
        "/api/documents/upload",
        data={"session_id": "../bad"},
        files={"files": ("policy.txt", b"Annual leave.")},
    )
    assert response.status_code == 422
