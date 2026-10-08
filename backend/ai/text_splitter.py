"""Split extracted text while preserving provenance on every LangChain Document."""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.core.config import Settings


def split_documents(documents: list[Document], settings: Settings) -> list[Document]:
    """Create overlapping chunks with stable IDs derived from document ID and position."""
    # Overlap preserves meaning that spans the boundary between adjacent chunks.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        add_start_index=True,
    )
    chunks = splitter.split_documents(documents)
    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = f"{chunk.metadata['document_id']}_{index}"
    return chunks
