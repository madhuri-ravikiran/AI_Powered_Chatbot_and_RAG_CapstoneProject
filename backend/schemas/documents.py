"""Document status and processing responses for the upload API."""

from typing import Literal

from pydantic import BaseModel


class DocumentInfo(BaseModel):
    """Describe a sanitized document and whether it has been embedded."""

    document_id: str
    file_name: str
    file_type: str
    size: int
    status: Literal["uploaded", "processed"]
    chunks: int = 0


class DocumentList(BaseModel):
    """List documents belonging only to the requested session."""

    documents: list[DocumentInfo]


class ProcessResponse(DocumentList):
    """Report how much new work was performed by the process operation."""

    processed_documents: int
    added_chunks: int
    total_chunks: int
    processing_time_ms: float
