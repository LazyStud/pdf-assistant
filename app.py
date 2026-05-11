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
            
            # If this message had an image attached, show it above the answer
            if "image" in msg:
                st.image(msg["image"], width=280, caption="Attached image")

            st.markdown(msg["content"])
            if "sources" in msg:
                with st.expander("Sources"):
                    for i, source in enumerate(msg["sources"], 1):
                        st.caption(f"[Page {source['page']}] {source['text'][:200]}...")

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
        # Grab the image bytes if an image was uploaded, so we can pass it to the pipeline
        image_bytes = uploaded_image.getvalue() if uploaded_image else None

        # Display user message immediately, including the attached image if there is one
        with st.chat_message("user"):
            if image_bytes:
                st.image(image_bytes, width=280, caption="Attached image")
            st.markdown(query)

        # Save user message before generating — question stays in history
        # even if generation fails midway
        user_msg = {"role": "user", "content": query}
        if image_bytes:
            user_msg["image"] = image_bytes  # store bytes so image re-renders on rerun
        st.session_state.messages.append(user_msg)

        # Query the pipeline and stream the answer into the assistant bubble
        with st.chat_message("assistant"):
            if image_bytes:
                # Vision path — model reads the image AND searches the PDF
                # use_pdf_context=True always, PDF is the whole point of this app 
                with st.spinner("Searching and generating answer..."):
                    response: dict = st.session_state.pipeline.query_with_image(
                        query, 
                        image_bytes=image_bytes,
                        use_pdf_context=True
                    )
            else:
                with st.spinner("Searching and generating answer..."):
                    response: dict = st.session_state.pipeline.query(query)
                
            st.markdown(response["answer"])
            if response["sources"]:
                with st.expander("Sources"):
                    for i, source in enumerate(response["sources"], 1):
                        st.caption(f"[Page {source['page']}] - {source['text'][:200]}...")

        # Persist the assistant message including source metadata for history replay
        st.session_state.messages.append({
            "role": "assistant",
            "content": response["answer"],
            "sources": response["sources"],
        })

        # Rerun clears the image uploader so the next message starts fresh
        st.rerun()
