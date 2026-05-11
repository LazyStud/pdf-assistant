# RAG PDF Assistant

A local Retrieval-Augmented Generation (RAG) application that lets you chat with any PDF document using natural language.  Upload a PDF through the Streamlit UI, ask questions in plain English, and get answers grounded in the document — complete with page-level source citations.  You can also attach an image alongside your question to get vision-powered answers cross-referenced against the PDF.

---

## Features

| Feature | Details |
|---|---|
| **PDF Q&A** | Ask anything about an uploaded PDF; the assistant answers from the document only |
| **Source citations** | Every answer links back to the exact page(s) it was drawn from |
| **Image Q&A** | Attach a screenshot, chart, or photo and ask questions about it |
| **Hybrid image + PDF** | Image questions automatically pull relevant PDF context for richer answers |
| **Fully local embeddings** | SentenceTransformer runs on CPU — no external embedding API, no cost |
| **Free LLM inference** | Powered by Groq's free tier (no credit card needed) |
| **In-memory vector store** | ChromaDB stores vectors in RAM — no database setup, no disk writes |
| **Per-session privacy** | All data is cleared when the browser tab closes or "Clear PDF Data" is clicked |

---

## How it works

### Text Q&A flow

```
PDF file
   │
   ▼
PyMuPDF (fitz)          — extract raw text, page by page
   │
   ▼
RecursiveCharacterTextSplitter  — split into 500-char chunks (50-char overlap)
   │
   ▼
SentenceTransformer     — encode all chunks into 384-d float vectors (local CPU)
   │
   ▼
ChromaDB (in-memory)    — store vectors + metadata (page number, text preview)
   │
   │   ◄── user question
   ▼
SentenceTransformer     — encode the question into a vector
   │
   ▼
ChromaDB (cosine search) — retrieve top-4 most similar chunks
   │
   ▼
Groq LLM (llama-3.1-8b-instant) — generate a cited answer from retrieved context
   │
   ▼
Streamlit UI            — display answer + collapsible source citations
```

### Image Q&A flow

```
Image upload (JPEG / PNG / WEBP)
   │
   ▼
base64 encode           — in-memory, never written to disk
   │
   ├── (if PDF is loaded) ChromaDB cosine search → top-3 PDF chunks appended as context
   │
   ▼
Groq Vision LLM (llama-4-scout-17b-16e-instruct) — reads image + optional PDF context
   │
   ▼
Streamlit UI            — display answer + PDF source citations (if any)
```

---

## Project structure

```text
pdf-assistant/
├── app.py               # Streamlit frontend — upload, chat UI, image attachment, citations
├── src/
│   └── rag_pipeline.py  # RAGPipeline class — ingest(), query(), query_with_image(), _retrieve()
├── requirements.txt     # Python dependencies
└── .env                 # GROQ_API_KEY (not committed to git)
```

### `src/rag_pipeline.py` — `RAGPipeline` class

| Method | Description |
|---|---|
| `__init__()` | Initialises Groq client, SentenceTransformer embedder, in-memory ChromaDB collection, and text splitter |
| `ingest(pdf_path)` | Extracts text page-by-page, chunks it, batch-encodes embeddings, and indexes everything in ChromaDB |
| `_retrieve(query, k)` | Embeds the query and returns the top-k nearest chunks as `(sources, context_string)` |
| `query(query, k=4)` | Full text Q&A: retrieve → prompt → Groq LLM → `{answer, sources}` |
| `query_with_image(query, image_bytes, k=3, use_pdf_context=True)` | Vision Q&A: base64-encode image → optionally retrieve PDF context → Groq vision LLM → `{answer, sources}` |

### `app.py` — Streamlit frontend

| Section | Description |
|---|---|
| Sidebar | PDF file uploader, "Process PDF" button, chunk count feedback, "Clear PDF Data" reset |
| Session state | `pipeline` (active `RAGPipeline` or `None`), `messages` (full chat history with image bytes and sources) |
| Chat history | Replays all messages on every Streamlit rerun, including attached images and source expanders |
| Image uploader | Optional per-message image attachment; cleared automatically after each submission |
| Chat input | Routes to `query()` (text-only) or `query_with_image()` (with image) based on attachment presence |

---

## Setup

### Prerequisites

- Python 3.10 or newer
- A free [Groq API key](https://console.groq.com/) (no credit card required)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/LazyStud/pdf-assistant.git
cd pdf-assistant

# 2. Create and activate a virtual environment
python -m venv pdfassist
pdfassist\Scripts\activate        # Windows
# source pdfassist/bin/activate   # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt
```

### Environment variables

Create a `.env` file in the project root (it is gitignored):

```env
GROQ_API_KEY=your_groq_api_key_here
```

Get your key at [console.groq.com](https://console.groq.com/) → API Keys → Create API Key.

---

## Running the app

```bash
streamlit run app.py
```

Open the URL printed in the terminal (usually `http://localhost:8501`).

**Usage:**
1. Upload a PDF in the left sidebar and click **Process PDF**
2. Wait for the "PDF processed" confirmation — it shows the number of chunks indexed
3. Type a question in the chat box and press Enter
4. Optionally attach an image using the image uploader above the chat box before asking

---

## Configuration

### Text LLM (Groq — all free)

Change the `model` string in `RAGPipeline.query()` ([src/rag_pipeline.py](src/rag_pipeline.py)):

| Model | Speed | Quality | Notes |
|---|---|---|---|
| `llama-3.1-8b-instant` | Fastest | Good | **Default** |
| `llama-3.3-70b-versatile` | Medium | Best on Groq | Better for complex reasoning |
| `mixtral-8x7b-32768` | Fast | Good | 32k context — better for very long PDFs |
| `gemma2-9b-it` | Fast | Good | Google's model, different reasoning style |

To switch provider entirely, replace `self.groq` in `RAGPipeline.__init__()`:

| Provider | Notes |
|---|---|
| **Ollama** | Fully local, offline: `ollama-python` client, `model="llama3.2"` |
| **Gemini Flash** | Google, free tier: `google-generativeai`, `model="gemini-1.5-flash"` |
| **OpenAI** | Paid (cheapest option): `openai` client, `model="gpt-4o-mini"` |

### Vision LLM (Groq — all free)

Change the `model` string in `RAGPipeline.query_with_image()`:

| Model | Speed | Notes |
|---|---|---|
| `meta-llama/llama-4-scout-17b-16e-instruct` | Fast | **Default** — ~750 tokens/sec |
| `meta-llama/llama-4-maverick-17b-128e-instruct` | Slower | Stronger reasoning, larger context |
| `llama-3.2-11b-vision-preview` | Fastest | Lighter model, lower quality |

### Embedding model (local — no API key needed)

Change the `SentenceTransformer` string in `RAGPipeline.__init__()`:

| Model | Size | Speed | Quality | Notes |
|---|---|---|---|---|
| `all-MiniLM-L6-v2` | 90 MB | Fastest | Good | **Default** |
| `BAAI/bge-small-en-v1.5` | 130 MB | Fast | Better | Recommended upgrade |
| `all-mpnet-base-v2` | 420 MB | Medium | Good | |
| `BAAI/bge-large-en-v1.5` | 1.3 GB | Slow | Best | |
| `paraphrase-multilingual-MiniLM-L12-v2` | 470 MB | Medium | Good | Non-English PDFs |

### Chunking parameters

Edit `RecursiveCharacterTextSplitter` in `RAGPipeline.__init__()`:

| Parameter | Default | Effect |
|---|---|---|
| `chunk_size` | 500 chars | Larger → fewer chunks, more context per chunk; smaller → more precise retrieval |
| `chunk_overlap` | 50 chars | Larger → less chance of cutting a sentence mid-thought; costs more storage |

### Retrieval depth

Pass `k` to `query()` or `query_with_image()`:

- `k=4` (default for text) — good balance of context vs. prompt size
- `k=3` (default for vision) — keeps the prompt compact for the vision model
- Increase `k` for broader coverage on long, complex PDFs

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| `streamlit` | ≥1.57.0 | Web UI framework |
| `groq` | ≥1.2.0 | Groq API client for LLM and vision inference |
| `pymupdf` | ≥1.27.2 | PDF text extraction (imported as `fitz`) |
| `sentence-transformers` | ≥5.4.1 | Local CPU text embeddings |
| `chromadb` | ≥1.5.9 | In-memory vector store with cosine similarity search |
| `langchain-text-splitters` | ≥1.1.2 | Recursive character-based text chunking |
| `python-dotenv` | ≥1.2.2 | `.env` file loading |
| `torchvision` | ≥0.26.0 | Required by `sentence-transformers` for image transforms |

---

## Limitations

- **In-memory only** — all vector data is lost when the app restarts or the PDF is cleared.  For persistent storage, swap `chromadb.Client()` for `chromadb.PersistentClient(path="./chroma_db")`.
- **Text PDFs only** — scanned PDFs without a text layer are skipped silently.  Add OCR (e.g. `pytesseract`) to handle image-based PDFs.
- **Single document** — one PDF per session.  Clicking "Process PDF" on a second file replaces the first.
- **English-optimised** — the default embedding model (`all-MiniLM-L6-v2`) works best with English text.  Switch to `paraphrase-multilingual-MiniLM-L12-v2` for other languages.
