# RAG PDF Assistant

A Retrieval-Augmented Generation (RAG) application that lets you chat with any PDF document using natural language. Upload a PDF through the Streamlit UI, and ask questions — the assistant retrieves the most relevant passages and generates answers grounded in the document.

## How it works

1. **Ingest** — loads a PDF with PyMuPDF, splits each page into 500-character overlapping chunks, and embeds them with a SentenceTransformer model
2. **Store** — vectors and metadata (page number, text preview) are indexed in an in-memory ChromaDB collection
3. **Query** — the user's question is embedded, the top-k closest chunks are retrieved, and Groq's LLM generates a cited answer using only that context

## Project structure

```text
pdf-assistant/
├── app.py               # Streamlit frontend (upload, chat UI, source citations)
├── src/
│   └── rag_pipeline.py  # RAGPipeline class (ingest, embed, query)
├── requirements.txt
└── .env                 # GROQ_API_KEY (not committed)
```

## Setup

### Prerequisites

- Python 3.10+
- A [Groq API key](https://console.groq.com/)

### Installation

```bash
# Clone the repo
git clone https://github.com/LazyStud/pdf-assistant.git
cd pdf-assistant

# Create and activate a virtual environment
python -m venv pdfassist
pdfassist\Scripts\activate        # Windows
# source pdfassist/bin/activate   # macOS / Linux

# Install dependencies
pip install -r requirements.txt
```

### Environment variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

## Running the app

```bash
streamlit run app.py
```

Open the URL printed in the terminal (usually `http://localhost:8501`), upload a PDF in the sidebar, and start chatting.

## Roadmap

- [ ] Image chat — upload an image and ask questions about it using a Groq vision model

## Dependencies

| Package | Purpose |
|---|---|
| `streamlit` | Web UI |
| `groq` | LLM inference (llama-3.1-8b-instant) |
| `pymupdf` | PDF text extraction |
| `sentence-transformers` | Text → vector embeddings (all-MiniLM-L6-v2) |
| `chromadb` | In-memory vector store and similarity search |
| `langchain-text-splitters` | Recursive character-based text chunking |
| `python-dotenv` | `.env` file loading |
