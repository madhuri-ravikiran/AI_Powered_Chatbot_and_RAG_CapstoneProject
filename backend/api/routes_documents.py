"""Bound multipart uploads, validate session IDs, and delegate document operations."""

from typing import Annotated

from fastapi import APIRouter, File, Form, Query, Request, UploadFile

from backend.core.exceptions import AppError
from backend.schemas.common import MessageResponse, SessionId, SessionRequest
from backend.schemas.documents import DocumentList, ProcessResponse

router = APIRouter(prefix="/api/documents", tags=["Documents"])


@router.post("/upload", response_model=DocumentList, status_code=201)
def upload_documents(
    request: Request,
    session_id: Annotated[SessionId, Form()],
    files: Annotated[list[UploadFile], File()],
) -> DocumentList:
    """Read each spooled file up to its cap; always close handles after extraction."""
    settings = request.app.state.settings
    try:
        if len(files) > settings.max_files_per_session:
            raise AppError(413, "too_many_files", "Too many files in one upload.")
        contents = []
        for file in files:
            data = file.file.read(settings.max_upload_mb * 1024 * 1024 + 1)
            if len(data) > settings.max_upload_mb * 1024 * 1024:
                raise AppError(413, "file_too_large", "A file exceeds the upload size limit.")
            contents.append((file.filename or "unnamed", data))
        return request.app.state.document_service.upload(session_id, contents)
    finally:
        for file in files:
            file.file.close()


@router.post("/process", response_model=ProcessResponse)
def process_documents(payload: SessionRequest, request: Request) -> ProcessResponse:
    """Embed pending documents and return exact chunk counts and elapsed time."""
    return request.app.state.document_service.process(payload.session_id)


@router.get("", response_model=DocumentList)
def get_documents(request: Request, session_id: Annotated[SessionId, Query()]) -> DocumentList:
    """List sanitized metadata for the specified session."""
    return request.app.state.document_service.list_documents(session_id)


@router.delete("", response_model=MessageResponse)
def clear_documents(request: Request, session_id: Annotated[SessionId, Query()]) -> MessageResponse:
    """Clear this session's knowledge base and its RAG conversation memory."""
    request.app.state.document_service.clear(session_id)
    return MessageResponse(message="Knowledge base and document conversation cleared.")
