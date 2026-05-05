# RAG PDF Reader

A Retrieval-Augmented Generation (RAG) pipeline that lets you query PDF documents using natural language. It extracts text from PDFs, indexes the content, and uses Groq's LLM to answer questions based on the document.

## How it works

1. **Ingest** — loads a PDF and extracts its text using PyMuPDF
2. **Embed** — chunks the text and stores it in a vector database (in progress)
3. **Query** — embeds the user's question, retrieves relevant chunks, and passes them to Groq's LLM for an answer

## Setup

### Prerequisites

- Python 3.9+
- A [Groq API key](https://console.groq.com/)

### Installation

```bash
# Clone the repo
git clone https://github.com/LazyStud/pdf-assistant.git
cd pdf-assistant

# Create and activate a virtual environment
python -m venv pdfassist
pdfassist\Scripts\activate   # Windows
# source pdfassist/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_groq_api_key_here
```

## Usage

```python
from rag_pipeline import RAGpipeline

pipeline = RAGpipeline()

# Ingest a PDF
pipeline.ingest("path/to/your/document.pdf")

# Ask a question
answer = pipeline.query("What is the main topic of this document?")
print(answer)
```

## Dependencies

| Package | Purpose |
|---|---|
| `groq` | LLM inference via Groq API |
| `pymupdf` | PDF text extraction |
| `python-dotenv` | Environment variable management |

## Project Status

Under active development. PDF text extraction, chunking, embedding, and vector storage are all in progress.
