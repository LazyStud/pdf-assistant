"""Streamlit frontend for the RAG PDF Assistant.

Handles file upload, PDF ingestion via RAGPipeline, chat history display,
and renders source citations alongside each assistant response.
"""

import tempfile
import os

import streamlit as st

from src.rag_pipeline import RAGPipeline


st.set_page_config(
    page_title="PDF Assistant",
    page_icon="📄",
    layout="centered",
    initial_sidebar_state="expanded",
)
st.title("📄 RAG PDF Assistant")
st.markdown(
    """
    Upload a PDF document and ask questions about its content.
    The assistant retrieves relevant passages and answers based solely on the PDF.
    """
)
st.caption("Upload a PDF. Ask anything. Sources included. Powered by Groq and ChromaDB.")

# --- Session state -----------------------------------------------------------
# pipeline: holds the active RAGPipeline instance; None until a PDF is processed
# messages: chat history as a list of dicts with keys "role", "content", "sources"
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Sidebar: PDF upload and processing --------------------------------------
with st.sidebar:
    st.header("Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    if uploaded_file is not None and st.button("Process PDF", type="primary"):
        with st.spinner("Chunking and embedding PDF..."):
            # Write the upload to a temp file so fitz can open it by path
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file_path: str = tmp_file.name

            pipeline = RAGPipeline()
            chunk_count: int = pipeline.ingest(tmp_file_path)
            st.session_state.pipeline = pipeline
            st.session_state.messages = []  # clear chat history for the new document
            os.unlink(tmp_file_path)        # remove the temp file immediately after ingestion

        st.success(f"PDF processed successfully! Total chunks created: {chunk_count}")

    # Allow the user to reset without restarting the app
    if st.session_state.pipeline is not None:
        st.divider()
        if st.button("Clear PDF Data", type="secondary"):
            st.session_state.pipeline = None
            st.session_state.messages = []
            st.success("PDF data cleared. You can upload a new PDF now.")
            st.rerun()

# --- Main area: chat interface -----------------------------------------------
if not st.session_state.pipeline:
    st.info("👈 Upload a PDF in the sidebar to get started.")
else:
    # Replay the full conversation history on each rerun
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "sources" in msg:
                with st.expander("Sources"):
                    for i, source in enumerate(msg["sources"], 1):
                        st.caption(f"[Page {source['page']}] {source['text'][:200]}...")

    if query := st.chat_input("Ask a question about the document"):
        # Persist and display the user message immediately
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        # Query the pipeline and stream the answer into the assistant bubble
        with st.chat_message("assistant"):
            with st.spinner("Searching and generating answer..."):
                response: dict = st.session_state.pipeline.query(query)
            st.markdown(response["answer"])
            with st.expander("Sources"):
                for i, source in enumerate(response["sources"], 1):
                    st.caption(f"[Page {source['page']}] - {source['text'][:200]}...")

        # Persist the assistant message including source metadata for history replay
        st.session_state.messages.append({
            "role": "assistant",
            "content": response["answer"],
            "sources": response["sources"],
        })
