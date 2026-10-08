"""Check contextual RAG end to end with real extraction/FAISS and a mocked LLM."""

import json

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from backend.ai.prompts import FALLBACK
from tests.conftest import upload_and_process


def grounded_response(app, session_id="test-session", quote="Annual leave is 24 days."):
    """Build a fake response using an identifier that genuinely exists in the index."""
    store = app.state.registry.sessions[session_id].vector_store
    return json.dumps(
        {
            "answer": "Employees receive 24 days of annual leave.",
            "evidence": [{"chunk_id": store.documents[0].metadata["chunk_id"], "quote": quote}],
        }
    )


def test_rag_answer_sources_and_followup(client, app, llm, embeddings):
    """A follow-up rewrites its query; all citations originate in the retrieved store."""
    upload_and_process(client)
    llm.responses = [grounded_response(app)]
    response = client.post(
        "/api/chat",
        json={"session_id": "test-session", "mode": "rag", "message": "How much annual leave?"},
    )
    assert response.status_code == 200
    answer = response.json()
    assert answer["sources"][0]["file_name"] == "policy.txt"
    assert answer["sources"][0]["page"] is None
    assert answer["sources"][0]["excerpt"] == "Annual leave is 24 days."
    assert answer["processing_time_ms"] >= 0
    seen = []

    def followup(prompt):
        """Record both calls and provide a standalone query before a grounded answer."""
        messages = prompt.to_messages()
        seen.append(messages)
        if "Rewrite" in messages[0].content:
            return AIMessage(content="How many annual leave days do employees receive?")
        return AIMessage(content=grounded_response(app))

    app.state.chat_service.llm_factory = lambda temperature: RunnableLambda(followup)
    second = client.post(
        "/api/chat",
        json={"session_id": "test-session", "mode": "rag", "message": "How many are there?"},
    )
    assert second.status_code == 200
    assert len(second.json()["sources"]) == 1
    assert len(seen) == 2
    assert any("How much annual leave?" == message.content for message in seen[0])
    assert "How many annual leave days" in seen[1][-1].content
    assert embeddings.document_calls == 1


def test_no_relevant_retrieval_returns_fallback(client, llm):
    """An unrelated query is answered safely without asking the LLM to invent facts."""
    upload_and_process(client)
    response = client.post(
        "/api/chat",
        json={"session_id": "test-session", "mode": "rag", "message": "What salary is paid?"},
    )
    assert response.status_code == 200
    assert response.json()["answer"] == FALLBACK
    assert response.json()["sources"] == []


def test_unsupported_answer_with_retrieved_context(client, llm):
    """Even when retrieval finds context, the model can report insufficient evidence."""
    upload_and_process(client)
    llm.responses = [json.dumps({"answer": FALLBACK, "evidence": []})]
    response = client.post(
        "/api/chat",
        json={
            "session_id": "test-session",
            "mode": "rag",
            "message": "What leave is offered in 2099?",
        },
    )
    assert response.json()["answer"] == FALLBACK
    assert response.json()["sources"] == []


def test_fabricated_source_or_quote_is_rejected(client, app, llm):
    """Reject hallucinated quotations and IDs, even if the generated answer looks plausible."""
    upload_and_process(client)
    for output in [
        grounded_response(app, quote="Employees have 999 leave days."),
        grounded_response(app, quote="   "),
        json.dumps(
            {
                "answer": "999 leave days.",
                "evidence": [{"chunk_id": "invented", "quote": "Annual leave is 24 days."}],
            }
        ),
    ]:
        app.state.chat_service.clear("test-session")
        llm.responses = [output]
        response = client.post(
            "/api/chat",
            json={"session_id": "test-session", "mode": "rag", "message": "Annual leave?"},
        )
        assert response.json()["answer"] == FALLBACK
        assert response.json()["sources"] == []


def test_bad_model_json_is_safe(client, app, llm):
    """Malformed model output is reported as a provider failure, not a fabricated answer."""
    upload_and_process(client)
    llm.responses = ["This is not JSON."]
    response = client.post(
        "/api/chat", json={"session_id": "test-session", "mode": "rag", "message": "Annual leave?"}
    )
    assert response.status_code == 502
    assert response.json()["code"] == "invalid_model_output"
    assert app.state.registry.sessions["test-session"].history["rag"] == []


def test_clear_knowledge_removes_rag_memory(client, app):
    """Removing documents also eliminates conversation references to their content."""
    upload_and_process(client)
    state = app.state.registry.sessions["test-session"]
    state.history["rag"] = [AIMessage(content="Old document information.")]
    client.delete("/api/documents", params={"session_id": "test-session"})
    assert state.history["rag"] == []
    assert state.vector_store is None
