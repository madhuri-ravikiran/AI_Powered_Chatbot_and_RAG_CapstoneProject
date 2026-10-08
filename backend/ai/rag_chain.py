"""Compose modern LangChain prompts, models, and parsers using Runnable pipes."""

import json

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable

from backend.ai.prompts import RAG_PROMPT, REWRITE_PROMPT
from backend.core.exceptions import AppError, provider_call
from backend.schemas.chat import GroundedAnswer


def rewrite_query(llm: Runnable, question: str, history: list[BaseMessage]) -> str:
    """Resolve follow-ups into standalone retrieval queries only when history exists."""
    if not history:
        return question
    chain = REWRITE_PROMPT | llm | StrOutputParser()
    query = provider_call(
        "query rewrite", lambda: chain.invoke({"question": question, "history": history})
    ).strip()
    if not query or len(query) > 4000:
        raise AppError(502, "invalid_rewrite", "The AI provider returned an invalid search query.")
    return query


def generate_grounded_answer(
    llm: Runnable,
    question: str,
    query: str,
    history: list[BaseMessage],
    documents: list[Document],
) -> GroundedAnswer:
    """Generate a JSON answer; validate the output before checking its citations."""
    context = json.dumps(
        [{"chunk_id": doc.metadata["chunk_id"], "text": doc.page_content} for doc in documents],
        ensure_ascii=False,
    )
    chain = RAG_PROMPT | llm | StrOutputParser()
    raw = provider_call(
        "LLM",
        lambda: chain.invoke(
            {"question": question, "query": query, "history": history, "context": context}
        ),
    )
    try:
        # Some local models add fences despite the prompt; accept that harmless wrapper.
        cleaned = raw.strip()
        if cleaned.startswith("```") and cleaned.endswith("```"):
            cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        return GroundedAnswer.model_validate_json(cleaned)
    except Exception:
        raise AppError(
            502, "invalid_model_output", "The model returned an invalid answer format. Retry."
        ) from None
