"""Select chat mode, preserve bounded memory, and delegate RAG to its service."""

import logging
from time import perf_counter

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser

from backend.ai.llm import create_llm
from backend.ai.prompts import GENERAL_PROMPT
from backend.core.config import Settings
from backend.core.exceptions import AppError, provider_call
from backend.core.state import SessionRegistry
from backend.schemas.chat import ChatRequest, ChatResponse
from backend.services.rag_service import RagService

logger = logging.getLogger(__name__)


class ChatService:
    """Commit history only after a successful response; separate general/RAG memory."""

    def __init__(self, settings: Settings, registry: SessionRegistry, llm_factory=None):
        """Accept model factory injection for tests without exposing a fake runtime provider."""
        self.settings = settings
        self.registry = registry
        self.llm_factory = llm_factory or (lambda temperature: create_llm(settings, temperature))
        self.rag = RagService(settings)

    def chat(self, request: ChatRequest) -> ChatResponse:
        """Generate a general or document answer, update memory, and measure elapsed time."""
        started = perf_counter()
        logger.info("Chat request received: mode=%s", request.mode)
        with self.registry.session(request.session_id) as state:
            if request.mode == "rag" and state.vector_store is None:
                raise AppError(
                    409,
                    "missing_knowledge_base",
                    "Upload and process documents before using RAG chat.",
                )
            history = state.history[request.mode]
            llm = self.llm_factory(request.temperature)
            sources = []
            if request.mode == "rag":
                answer, sources = self.rag.answer(
                    llm,
                    request.message,
                    history,
                    state.vector_store,
                    request.top_k or self.settings.top_k,
                )
            else:
                chain = GENERAL_PROMPT | llm | StrOutputParser()
                answer = provider_call(
                    "LLM", lambda: chain.invoke({"question": request.message, "history": history})
                ).strip()
                if not answer or len(answer) > 20000:
                    raise AppError(
                        502, "invalid_model_output", "The AI provider returned an invalid answer."
                    )
            # Store complete pairs only; failed calls must not poison future context.
            history.extend([HumanMessage(content=request.message), AIMessage(content=answer)])
            state.history[request.mode] = history[-2 * self.settings.max_history_turns :]
            return ChatResponse(
                answer=answer,
                session_id=request.session_id,
                mode=request.mode,
                sources=sources,
                processing_time_ms=round((perf_counter() - started) * 1000, 2),
            )

    def clear(self, session_id: str) -> None:
        """Clear both modes' conversation history while retaining indexed documents."""
        with self.registry.session(session_id) as state:
            state.history = {"general": [], "rag": []}
