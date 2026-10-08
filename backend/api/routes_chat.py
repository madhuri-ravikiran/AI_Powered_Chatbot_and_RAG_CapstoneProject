"""Validate chat and clear requests, delegate to ChatService, return typed JSON."""

from fastapi import APIRouter, Request

from backend.schemas.chat import ChatRequest, ChatResponse
from backend.schemas.common import MessageResponse, SessionRequest

router = APIRouter(prefix="/api/chat", tags=["Chat"])


@router.post("", response_model=ChatResponse)
def chat(payload: ChatRequest, request: Request) -> ChatResponse:
    """Generate an answer in a worker thread so blocking SDK calls do not block ASGI."""
    return request.app.state.chat_service.chat(payload)


@router.post("/clear", response_model=MessageResponse)
def clear_chat(payload: SessionRequest, request: Request) -> MessageResponse:
    """Clear conversation memory while leaving the session's documents available."""
    request.app.state.chat_service.clear(payload.session_id)
    return MessageResponse(message="Conversation cleared.")
