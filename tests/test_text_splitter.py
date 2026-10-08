"""Verify chunk size, overlap, and preserved source identities."""

import pytest
from pydantic import ValidationError

from backend.ai.document_loader import load_document
from backend.ai.text_splitter import split_documents
from backend.core.config import Settings


def test_chunking(settings):
    """Produce bounded overlapping chunks without losing original metadata."""
    docs = load_document("policy.txt", ("Annual leave policy. " * 50).encode(), "abc", 10000)
    chunks = split_documents(docs, settings)
    assert len(chunks) > 1
    assert all(len(chunk.page_content) <= settings.chunk_size for chunk in chunks)
    assert all(chunk.metadata["source"] == "policy.txt" for chunk in chunks)
    assert len({chunk.metadata["chunk_id"] for chunk in chunks}) == len(chunks)
    assert chunks[1].metadata["start_index"] < (
        chunks[0].metadata["start_index"] + len(chunks[0].page_content)
    )


def test_invalid_overlap():
    """Catch bad configuration at startup rather than during processing."""
    with pytest.raises(ValidationError):
        Settings(_env_file=None, chunk_size=100, chunk_overlap=100)
