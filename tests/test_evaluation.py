"""Check evaluation arithmetic without presenting fake responses as quality measurements."""

from backend.services.evaluation_service import EvaluationCase, EvaluationService, token_f1
from tests.conftest import upload_and_process
from tests.test_rag import grounded_response


def test_token_f1():
    """Verify exact matching and completely unrelated answers behave as expected."""
    assert token_f1("24 days annual leave", "24 days annual leave") == 1
    assert token_f1("24 days", "unknown") == 0


def test_measured_evaluation_structure(client, app, llm):
    """Calculate metrics on a test response; human quality fields remain unmeasured."""
    upload_and_process(client)
    llm.responses = [grounded_response(app)]
    result = EvaluationService(app.state.chat_service).evaluate(
        "test-session",
        [
            EvaluationCase(
                question="How much annual leave?",
                expected_answer="24 days of annual leave",
                expected_source="policy.txt",
            )
        ],
    )
    row = result["cases"][0]
    assert row["retrieval_source_hit_at_k"] is True
    assert row["retrieval_source_precision_at_k"] == 1
    assert 0 < row["answer_token_f1"] <= 1
    assert row["processing_time_ms"] >= 0
    assert row["human_groundedness"] is None
