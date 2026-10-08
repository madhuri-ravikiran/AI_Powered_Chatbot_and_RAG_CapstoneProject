"""IntelliChat's Streamlit UI while delegating all AI operations through HTTP.

The sidebar controls document lifecycle and conversation settings. Session state
stores UI messages and an opaque backend session ID, never API credentials.
"""

import os
import sys
from pathlib import Path
from uuid import uuid4

import streamlit as st
from dotenv import load_dotenv
from streamlit.runtime.scriptrunner import get_script_run_ctx

# Session state and st.stop require Streamlit's script runner, not a plain Python process.
if get_script_run_ctx(suppress_warning=True) is None:
    raise SystemExit(
        "IntelliChat must be launched through Streamlit. Run this command in your terminal:\n\n"
        f'"{sys.executable}" -m streamlit run "{Path(__file__).resolve()}"'
    )

# Streamlit executes this file as a script; ensure package imports work from the repo root.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from frontend.api_client import ApiClient, ApiError  # noqa: E402
from frontend.ui_helpers import export_conversation, show_sources  # noqa: E402

# ==============================================================
# PAGE CONFIGURATION
# ==============================================================
load_dotenv(ROOT / ".env")
st.set_page_config(
    page_title="IntelliChat – AI-Powered Knowledge Assistant",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(
    """<style>
.block-container {max-width: 1120px; padding-top: 2.5rem;}
h1 {letter-spacing: -0.045em;}
[data-testid="stSidebar"] {border-right: 1px solid #D5E2DB;}
[data-testid="stChatMessage"] {border: 1px solid #E0E9E3; border-radius: 14px;}
</style>""",
    unsafe_allow_html=True,
)

# ==============================================================
# SESSION STATE INITIALIZATION
# ==============================================================
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid4())
    st.session_state.messages = []
    st.session_state.documents = []
    st.session_state.uploader_version = 0
if "api" not in st.session_state:
    st.session_state.api = ApiClient(os.getenv("BACKEND_URL", "http://127.0.0.1:8000"))
api = st.session_state.api

st.caption("INTELLICHAT / KNOWLEDGE, IN CONVERSATION")
st.title("IntelliChat – AI-Powered Knowledge Assistant")
st.markdown("**AI-Powered Chatbot using LLM, LangChain, FastAPI & RAG**")
st.caption(
    "Ask questions, have intelligent conversations, or chat with your documents "
    "using Retrieval-Augmented Generation."
)

try:
    # One lightweight health call per interaction; configuration is cached in session state.
    api.health_check()
    if "config" not in st.session_state:
        st.session_state.config = api.get_config()
except ApiError as error:
    st.error(str(error))
    if st.button("Retry connection"):
        st.rerun()
    st.stop()
config = st.session_state.config


def refresh_documents() -> None:
    """Refresh server-owned status only after a mutation or explicit user request."""
    st.session_state.documents = api.get_documents(st.session_state.session_id)["documents"]


# ==============================================================
# SIDEBAR AND DOCUMENT UPLOAD
# ==============================================================
with st.sidebar:
    st.title("✦ IntelliChat")
    st.caption("AI-Powered Knowledge Assistant")
    st.success("Backend connected")
    st.subheader("Knowledge Base")
    st.caption(
        f"PDF, TXT, DOCX · {config['max_upload_mb']} MB per file. "
        "Content is sent to your configured embedding and chat providers."
    )
    uploaded = st.file_uploader(
        "Upload Documents",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
        key=f"uploads_{st.session_state.uploader_version}",
    )
    if st.button("Upload selected files", disabled=not uploaded, width="stretch"):
        try:
            with st.spinner("Validating and extracting documents…"):
                result = api.upload_documents(
                    st.session_state.session_id,
                    [(file.name, file.getvalue()) for file in uploaded],
                )
            st.session_state.documents = result["documents"]
            st.success("Uploaded. Process documents to make them searchable.")
        except ApiError as error:
            st.error(str(error))

    pending = any(doc["status"] == "uploaded" for doc in st.session_state.documents)
    if st.button("Process Documents", disabled=not pending, type="primary", width="stretch"):
        try:
            with st.spinner("Creating chunks and building the knowledge index…"):
                result = api.process_documents(st.session_state.session_id)
            st.session_state.documents = result["documents"]
            st.success(f"Ready · {result['total_chunks']} searchable chunks")
        except ApiError as error:
            st.error(str(error))
    with st.expander(f"View uploaded documents ({len(st.session_state.documents)})"):
        if not st.session_state.documents:
            st.caption("Your knowledge base is empty.")
        for doc in st.session_state.documents:
            st.text(doc["file_name"])
            st.caption(
                f"{doc['status'].title()} · {doc['size'] / 1024:.1f} KB · {doc['chunks']} chunks"
            )
        refresh_requested = st.button("Refresh document status")
    clear_requested = st.button("Clear Knowledge Base", disabled=not st.session_state.documents)

    st.divider()
    st.subheader("Chat Mode")
    mode_label = st.radio(
        "Choose a mode", ["General AI Chat", "Document / RAG Chat"], label_visibility="collapsed"
    )
    mode = "general" if mode_label == "General AI Chat" else "rag"
    st.subheader("RAG Settings")
    top_k = st.slider("Top-K retrieved chunks", 1, 10, config["top_k"])
    temperature = st.slider(
        "Temperature", 0.0, 1.0, 0.2, 0.1, disabled=not config["temperature_supported"]
    )
    st.caption("Lower temperature favors consistency. Top-K controls retrieved context.")

    st.divider()
    st.subheader("Conversation")
    st.caption("New Chat starts with an empty knowledge base. Clear Conversation keeps documents.")
    if st.button("New Chat", width="stretch"):
        try:
            api.clear_documents(st.session_state.session_id)
            api.clear_chat(st.session_state.session_id)
            st.session_state.session_id = str(uuid4())
            st.session_state.messages = []
            st.session_state.documents = []
            st.session_state.uploader_version += 1
            st.rerun()
        except ApiError as error:
            st.error(str(error))
    if st.button("Clear Conversation", width="stretch"):
        try:
            api.clear_chat(st.session_state.session_id)
            st.session_state.messages = []
            st.rerun()
        except ApiError as error:
            st.error(str(error))
    st.download_button(
        "Export Conversation",
        export_conversation(st.session_state.messages),
        file_name="intellichat-conversation.md",
        mime="text/markdown",
        disabled=not st.session_state.messages,
        width="stretch",
    )
    st.caption(f"{config['llm_provider']} / {config['llm_model']}")
    st.caption("Memory expires after inactivity or backend restart.")

    # Render settings before rerunning so Streamlit does not discard their widget state.
    if refresh_requested:
        try:
            refresh_documents()
            st.rerun()
        except ApiError as error:
            st.error(str(error))
    if clear_requested:
        try:
            api.clear_documents(st.session_state.session_id)
            st.session_state.documents = []
            st.session_state.messages = [
                message for message in st.session_state.messages if message["mode"] != "rag"
            ]
            st.session_state.uploader_version += 1
            st.rerun()
        except ApiError as error:
            st.error(str(error))


# ==============================================================
# CHAT INTERFACE
# ==============================================================
if not config["chat_configured"]:
    st.warning("Configure LLM_API_KEY in .env and restart FastAPI to enable AI answers.")
if mode == "rag" and not config["embeddings_configured"]:
    st.warning("Configure the embedding provider credentials before processing documents.")
ready = any(doc["status"] == "processed" for doc in st.session_state.documents)
if mode == "rag" and not ready:
    st.info("Upload and process documents in the sidebar to begin document chat.")

if not st.session_state.messages:
    with st.container(border=True):
        st.subheader("👋 Welcome to IntelliChat")
        st.markdown("### Your AI-Powered Knowledge Assistant")
        st.write(
            "You can ask general questions or upload documents "
            "and ask questions based on their contents."
        )
        left, right = st.columns(2)
        with left:
            st.markdown("**Explore an idea**")
            st.caption("Explain retrieval-augmented generation in simple terms.")
        with right:
            st.markdown("**Find an answer in your files**")
            st.caption("How many annual leave days do employees receive?")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.caption("Document / RAG Chat" if message["mode"] == "rag" else "General AI Chat")
        st.markdown(message["content"])
        show_sources(message.get("sources", []))
        if "processing_time_ms" in message:
            st.caption(f"{message['processing_time_ms'] / 1000:.2f} seconds")

question = st.chat_input(
    "Ask IntelliChat a question…",
    disabled=not config["chat_configured"] or (mode == "rag" and not ready),
    max_chars=4000,
)
if question:
    with st.chat_message("user"):
        st.markdown(question)
    try:
        with st.spinner("Searching your documents…" if mode == "rag" else "Thinking…"):
            response = api.send_chat_message(
                st.session_state.session_id, question, mode, top_k, temperature
            )
        # Commit UI history only after the backend succeeds, keeping both views aligned.
        st.session_state.messages.extend(
            [
                {"role": "user", "content": question, "mode": mode},
                {
                    "role": "assistant",
                    "content": response["answer"],
                    "mode": mode,
                    "sources": response["sources"],
                    "processing_time_ms": response["processing_time_ms"],
                },
            ]
        )
        st.rerun()
    except ApiError as error:
        st.error(str(error))
        st.caption("Your message was not saved. Please submit it again when the issue is resolved.")
