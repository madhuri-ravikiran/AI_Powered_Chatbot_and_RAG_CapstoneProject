"""Validate uploads, stage extraction, and process each document only once."""

import hashlib
import logging
from pathlib import PurePosixPath
from time import perf_counter

from backend.ai.document_loader import load_document, safe_filename
from backend.ai.embeddings import create_embeddings
from backend.ai.text_splitter import split_documents
from backend.ai.vector_store import VectorStore
from backend.core.config import Settings
from backend.core.exceptions import AppError
from backend.core.state import SessionRegistry, StoredDocument
from backend.schemas.documents import DocumentInfo, DocumentList, ProcessResponse

logger = logging.getLogger(__name__)


class DocumentService:
    """Keep upload/extraction separate from embedding and index construction."""

    def __init__(self, settings: Settings, registry: SessionRegistry, embedding_factory=None):
        """Inject provider creation so tests use deterministic, offline embeddings."""
        self.settings = settings
        self.registry = registry
        self.embedding_factory = embedding_factory or (lambda: create_embeddings(settings))

    def upload(self, session_id: str, files: list[tuple[str, bytes]]) -> DocumentList:
        """Validate the entire batch before committing; identical uploads are idempotent."""
        if not files:
            raise AppError(422, "no_files", "Select at least one document.")
        with self.registry.session(session_id) as state:
            staged = dict(state.documents)
            for original_name, data in files:
                if len(data) > self.settings.max_upload_mb * 1024 * 1024:
                    raise AppError(413, "file_too_large", "A file exceeds the upload size limit.")
                name = safe_filename(original_name)
                document_id = hashlib.sha256(name.encode() + b"\0" + data).hexdigest()[:24]
                if document_id in staged:
                    continue
                if len(staged) >= self.settings.max_files_per_session:
                    raise AppError(413, "too_many_files", "The session document limit was reached.")
                if any(doc.info.file_name == name for doc in staged.values()):
                    raise AppError(
                        409,
                        "filename_conflict",
                        "That filename already exists. Rename it or clear the knowledge base.",
                    )
                pages = load_document(name, data, document_id, self.settings.max_extracted_chars)
                info = DocumentInfo(
                    document_id=document_id,
                    file_name=name,
                    file_type=PurePosixPath(name).suffix.lower()[1:],
                    size=len(data),
                    status="uploaded",
                )
                staged[document_id] = StoredDocument(info, pages)
            state.documents = staged
            logger.info("Document batch received: documents=%d", len(staged))
            return DocumentList(documents=[doc.info for doc in staged.values()])

    def list_documents(self, session_id: str) -> DocumentList:
        """Return only this session's public document status."""
        with self.registry.session(session_id) as state:
            return DocumentList(documents=[doc.info for doc in state.documents.values()])

    def process(self, session_id: str) -> ProcessResponse:
        """Embed pending documents atomically; repeated processing does no extra work."""
        started = perf_counter()
        with self.registry.session(session_id) as state:
            if not state.documents:
                raise AppError(409, "missing_documents", "Upload documents before processing.")
            pending = [doc for doc in state.documents.values() if doc.info.status == "uploaded"]
            chunks = []
            counts = {}
            logger.info("Document processing started: pending=%d", len(pending))
            for document in pending:
                split = split_documents(document.pages, self.settings)
                counts[document.info.document_id] = len(split)
                chunks.extend(split)
            if chunks:
                store = state.vector_store or VectorStore(self.embedding_factory())
                store.add_documents(chunks)
                state.vector_store = store
                for document in pending:
                    document.info = document.info.model_copy(
                        update={
                            "status": "processed",
                            "chunks": counts[document.info.document_id],
                        }
                    )
            return ProcessResponse(
                documents=[doc.info for doc in state.documents.values()],
                processed_documents=len(pending),
                added_chunks=len(chunks),
                total_chunks=len(state.vector_store.documents) if state.vector_store else 0,
                processing_time_ms=round((perf_counter() - started) * 1000, 2),
            )

    def clear(self, session_id: str) -> None:
        """Delete the knowledge base and RAG memory so stale document facts cannot leak."""
        with self.registry.session(session_id) as state:
            state.documents.clear()
            if state.vector_store:
                state.vector_store.clear()
            state.vector_store = None
            state.history["rag"].clear()
