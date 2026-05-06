import streamlit as st
import tempfile, os
from src.rag_pipeline import RAGPipeline


# Page configuration
st.set_page_config(
    page_title="PDF Assistant",
    page_icon="📄",
    layout="centered",
    initial_sidebar_state="expanded",
)
st.title("📄 RAG PDF Assistant" )
st.markdown(
    """
    Upload a PDF document and ask questions about its content. The assistant will retrieve relevant information and provide answers based on the PDF.
    """
)
st.caption("Upload a PDF. Ask anything. Sources included. Powered by Groq and ChromaDB.")

# Initialize RAG pipeline and session state
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# File upload
with st.sidebar:
    st.header("Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
    # click to process PDF
    if uploaded_file is not None and st.button("Process PDF", type="primary"):
        # write the uploaded file to a temporary location and process it with RAG pipeline
        with st.spinner("Chunking and embedding PDF..."):
            # Save uploaded file to a temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file_path = tmp_file.name
            
            # Initialize RAG pipeline and process the PDF
            pipeline = RAGPipeline()
            chunks = pipeline.ingest(tmp_file_path)
            st.session_state.pipeline = pipeline
            st.session_state.messages = []
            os.unlink(tmp_file_path)  # Clean up temporary file
            
        # Show success message with number of chunks created
        st.success(f"PDF processed successfully! Total chunks created: {chunks}")

    if st.session_state.pipeline is not None:
        st.divider()
        if st.button("Clear PDF Data", type="secondary"):
            st.session_state.messages = []
            st.success("PDF data cleared. You can upload a new PDF now.")
            st.rerun()

# if no pdf is uploaded, show a message to upload a pdf, otherwise show the chat interface
if not st.session_state.pipeline:
    st.info("👈 Upload a PDF in the sidebar to get started.")
else:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "sources" in msg:
                with st.expander("Sources"):
                    for i, source in enumerate(msg["sources"], 1):
                        st.caption(f"[Page {source['page']}] {source['text'][:200]}...")
    
    if query := st.chat_input("Ask a question about the document"):
        # save query to history and display in chat
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)
        
        # pipeline query and response
        with st.chat_message("assistant"):
            with st.spinner("Searching and generating answer..."):
                response = st.session_state.pipeline.query(query)
            st.markdown(response["answer"])
            with st.expander("Sources"):
                for i, source in enumerate(response["sources"], 1):
                    st.caption(f"[Page {source['page']}] - {source['text'][:200]}...")
        
        # save response to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": response["answer"], 
            "sources": response["sources"]
        })