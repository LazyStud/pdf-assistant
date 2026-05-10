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

## Model options

### LLM (Groq — all free)

Change the `model` string in `RAGPipeline.query()` ([src/rag_pipeline.py:144](src/rag_pipeline.py#L144)):

| Model | Speed | Quality | Notes |
|---|---|---|---|
| `llama-3.1-8b-instant` | Fastest | Good | **Default** |
| `llama-3.3-70b-versatile` | Medium | Best on Groq | |
| `mixtral-8x7b-32768` | Fast | Good | 32k context — good for long PDFs |
| `gemma2-9b-it` | Fast | Good | Google's model, different reasoning style |

To switch provider entirely, replace the Groq client in `RAGPipeline.__init__()`:

| Provider | Notes |
|---|---|
| **Ollama** | Fully local, offline: `ollama.chat(model="llama3.2", ...)` |
| **Gemini Flash** | Google, free tier: `google.generativeai` with `gemini-1.5-flash` |
| **OpenAI** | Paid, cheapest option: `client.chat.completions` with `gpt-4o-mini` |

### Embedding model (local, no API needed)

Change the `SentenceTransformer` string in `RAGPipeline.__init__()` ([src/rag_pipeline.py:30](src/rag_pipeline.py#L30)). The same model string must be used in both `ingest()` and `query()`.

| Model | Size | Speed | Quality | Notes |
|---|---|---|---|---|
| `all-MiniLM-L6-v2` | 90 MB | Fastest | Good | **Default** |
| `BAAI/bge-small-en-v1.5` | 130 MB | Fast | Better | Recommended upgrade |
| `all-mpnet-base-v2` | 420 MB | Medium | Good | |
| `BAAI/bge-large-en-v1.5` | 1.3 GB | Slow | Best | |
| `paraphrase-multilingual-MiniLM-L12-v2` | 470 MB | Medium | Good | Non-English PDFs |

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
