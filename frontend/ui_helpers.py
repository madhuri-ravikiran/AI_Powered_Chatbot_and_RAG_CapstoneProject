"""Render citation cards and create a portable Markdown conversation export."""

import streamlit as st


def show_sources(sources: list[dict]) -> None:
    """Display genuine backend metadata; omit page labels when pagination is unknown."""
    if not sources:
        return
    with st.expander(f"Sources · {len(sources)}"):
        for source in sources:
            label = source["file_name"]
            if source.get("page") is not None:
                label += f" · page {source['page']}"
            st.text(label)
            st.caption(f"Chunk {source['chunk_id']} · cosine similarity {source['similarity']:.3f}")
            # Plain text keeps an uploaded excerpt from rendering links or remote images.
            st.text(source["excerpt"])


def export_conversation(messages: list[dict]) -> str:
    """Export displayed messages with their document references as Markdown."""
    parts = ["# AI-Powered Chatbot", "IntelliChat – AI-Powered Knowledge Assistant", ""]
    for message in messages:
        parts.extend([f"## {message['role'].title()} ({message['mode']})", message["content"], ""])
        for source in message.get("sources", []):
            page = f", page {source['page']}" if source.get("page") is not None else ""
            parts.append(f"Source: {source['file_name']}{page}; chunk {source['chunk_id']}")
    return "\n".join(parts)
