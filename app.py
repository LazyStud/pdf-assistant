"""Streamlit frontend for the RAG PDF Assistant.

Page layout
-----------
  Sidebar  — PDF upload, "Process PDF" button, chunk count feedback, "Clear" reset.
  Main     — Chat history replay, optional image attachment, chat input box.

Session state keys
------------------
  pipeline (RAGPipeline | None): Active pipeline instance; None until a PDF is ingested.
      A new RAGPipeline is created for each uploaded document so the vector store is
      always scoped to the current file.
  messages (list[dict]): Full chat history for the current document.  Each entry has:
      - "role"    (str):        "user" or "assistant"
      - "content" (str):        Message text (markdown)
      - "sources" (list[dict]): Retrieved chunk metadata — present on assistant messages only
      - "image"   (bytes):      Raw image bytes — present on user messages that had an attachment

Query paths
-----------
  Text-only  → RAGPipeline.query()            — embeds question, retrieves chunks, Groq LLM
  With image → RAGPipeline.query_with_image() — base64 image + retrieved chunks, Groq vision LLM
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
# Initialise keys on first run; Streamlit preserves these across reruns.
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None  # set to a RAGPipeline instance after successful ingest
if "messages" not in st.session_state:
    st.session_state.messages = []    # list of message dicts; cleared when a new PDF is loaded

# --- Sidebar: PDF upload and processing --------------------------------------
with st.sidebar:
    st.header("Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    if uploaded_file is not None and st.button("Process PDF", type="primary"):
        with st.spinner("Chunking and embedding PDF..."):
            # fitz (PyMuPDF) requires a file path, not a file-like object, so we write
            # the Streamlit upload to a temp file and delete it immediately after ingestion.
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file_path: str = tmp_file.name

            pipeline = RAGPipeline()
            chunk_count: int = pipeline.ingest(tmp_file_path)
            st.session_state.pipeline = pipeline
            st.session_state.messages = []  # clear history — new document, fresh conversation
            os.unlink(tmp_file_path)        # clean up temp file; all data is now in ChromaDB

        st.success(f"PDF processed successfully! Total chunks created: {chunk_count}")

    # "Clear" lets the user swap documents without restarting the Streamlit server.
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
    # Replay the full conversation on every Streamlit rerun (required by Streamlit's
    # execution model — the script re-runs top-to-bottom on each interaction).
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if "image" in msg:
                # Re-render the attached image above the message text on history replay
                st.image(msg["image"], width=280, caption="Attached image")
            st.markdown(msg["content"])
            if "sources" in msg:
                with st.expander("Sources"):
                    for i, source in enumerate(msg["sources"], 1):
                        st.caption(f"[Page {source['page']}] {source['text'][:200]}...")

    # Optional image attachment — rendered above the chat input for visibility.
    # key="image_uploader" ensures Streamlit treats this as a separate widget from
    # the PDF uploader above and clears it correctly on st.rerun().
    uploaded_image = st.file_uploader(
        "Attach an image (optional)",
        type=["png", "jpg", "jpeg", "webp"],
        key="image_uploader",
    )

    if uploaded_image:
        st.image(uploaded_image, width=280, caption="Image attached — ask your question below")

    if query := st.chat_input(
        "Ask about the image..." if uploaded_image else "Ask anything about the document..."
    ):
        image_bytes = uploaded_image.getvalue() if uploaded_image else None

        # Show the user message immediately — Streamlit chat messages appear as they are written,
        # giving instant feedback before the (potentially slow) LLM call begins.
        with st.chat_message("user"):
            if image_bytes:
                st.image(image_bytes, width=280, caption="Attached image")
            st.markdown(query)

        # Persist the user message before calling the LLM so it survives if generation fails.
        user_msg: dict = {"role": "user", "content": query}
        if image_bytes:
            user_msg["image"] = image_bytes  # store raw bytes so the image re-renders on rerun
        st.session_state.messages.append(user_msg)

        with st.chat_message("assistant"):
            if image_bytes:
                # Vision path: base64-encode the image and retrieve PDF context in one call.
                # use_pdf_context=True is always correct here — the whole app is PDF-centric.
                with st.spinner("Searching and generating answer..."):
                    response: dict = st.session_state.pipeline.query_with_image(
                        query,
                        image_bytes=image_bytes,
                        use_pdf_context=True,
                    )
            else:
                # Text-only path: embed the query, retrieve top-k chunks, generate answer.
                with st.spinner("Searching and generating answer..."):
                    response = st.session_state.pipeline.query(query)

            st.markdown(response["answer"])
            if response["sources"]:
                with st.expander("Sources"):
                    for i, source in enumerate(response["sources"], 1):
                        st.caption(f"[Page {source['page']}] - {source['text'][:200]}...")

        # Persist the assistant message with sources so citations re-render on history replay.
        st.session_state.messages.append({
            "role": "assistant",
            "content": response["answer"],
            "sources": response["sources"],
        })

        # Rerun resets the image uploader widget so the next turn starts without a stale image.
        st.rerun()
