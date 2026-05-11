"""RAG pipeline for PDF question-answering using ChromaDB, SentenceTransformers, and Groq."""

import os
from typing import Any
import base64
from groq import Groq
import fitz  # PyMuPDF
from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv() # Load environment variables from .env file, including GROQ_API_KEY for authentication


class RAGPipeline:
    def __init__(self) -> None:
        """Initialize the RAG pipeline: LLM client, embedding model, vector store, and text splitter."""
        # Groq LLM client — requires GROQ_API_KEY in .env
        self.groq = Groq(api_key=os.getenv("GROQ_API_KEY"))

        # EMBEDDING MODEL — converts text to vectors, runs locally on CPU, no API needed
        # Swap the model string below to change quality/speed tradeoff:
        #   "all-MiniLM-L6-v2"                        → 90MB,  fastest,  good quality    (current)
        #   "BAAI/bge-small-en-v1.5"                   → 130MB, fast,     better quality  (recommended upgrade)
        #   "all-mpnet-base-v2"                        → 420MB, medium,   good quality
        #   "BAAI/bge-large-en-v1.5"                   → 1.3GB, slow,     best quality
        #   "paraphrase-multilingual-MiniLM-L12-v2"    → 470MB, medium,   use for non-English PDFs
        # NOTE: whatever model you set here MUST match the one used in query() below
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

        # In-memory ChromaDB client — no external server required
        self.chroma = chromadb.Client()
        self.collection = self.chroma.get_or_create_collection(
            name="pdf_chunks",
            metadata={"hnsw:space": "cosine"}  # cosine similarity is best for text embeddings
        )

        # 500-char chunks with 50-char overlap to preserve sentence context across boundaries
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", " ", ". "]
        )

    def ingest(self, pdf_path: str) -> int:
        """Extract text from a PDF, split into chunks, embed, and store in ChromaDB.

        Args:
            pdf_path: Absolute path to the PDF file to process.

        Returns:
            Total number of text chunks indexed into the vector store.
        """
        doc = fitz.open(pdf_path)

        # These three lists must stay in sync — same index = same chunk
        chunks: list[str] = []
        metadatas: list[dict[str, Any]] = []  # page number + text, returned as UI citations
        ids: list[str] = []                   # unique ChromaDB identifier per chunk

        # fitz stubs don't declare __iter__, so iterate by index
        for i in range(len(doc)):
            page_num = i + 1
            page: fitz.Page = doc[i]
            text: str = page.get_text()  # type: ignore[assignment]

            # Skip pages with no extractable text (scanned images, blank pages, etc.)
            if not text.strip():
                continue

            page_chunks: list[str] = self.splitter.split_text(text)

            for j, chunk in enumerate(page_chunks):
                chunks.append(chunk)
                metadatas.append({
                    "page": page_num,   # shown as "Page N" in the UI source citations
                    "text": chunk       # shown as preview text under each citation
                })
                # ID format: "p3_c1" = page 3, chunk index 1
                ids.append(f"p{page_num}_c{j}")

        # Batch-encode all chunks at once — significantly faster than encoding one by one
        # Output shape: (num_chunks, 384); .tolist() converts numpy array to plain Python list
        embeddings = self.embedder.encode(chunks, show_progress_bar=True).tolist()

        # Store chunks, their vectors, and metadata; ChromaDB builds an HNSW index internally
        self.collection.add(
            documents=chunks,
            embeddings=embeddings,  # type: ignore[arg-type]  # runtime-correct; stubs are too strict
            metadatas=metadatas,    # type: ignore[arg-type]
            ids=ids
        )

        return len(chunks)

    def _retrieve(self, query: str, k: int) -> tuple[list[dict[str, Any]], str]:
        """Embed a query and return the top-k chunks as (sources, context_string)."""
        embedding = self.embedder.encode([query]).tolist()
        results = self.collection.query(
            query_embeddings=embedding,  # type: ignore[arg-type]
            n_results=k
        )
        raw_metadatas = results["metadatas"] or [[]]
        raw_documents = results["documents"] or [[]]
        sources: list[dict[str, Any]] = [dict(m) for m in raw_metadatas[0]]  # type: ignore[misc]
        context = "\n\n--\n\n".join(raw_documents[0])
        return sources, context

    def query(self, query: str, k: int = 4) -> dict[str, Any]:
        """Embed a user query, retrieve the most relevant chunks, and generate an LLM answer.

        Args:
            query: Natural-language question to answer from the indexed document.

        Returns:
            A dict with:
                - "answer" (str): LLM-generated response with inline page citations.
                - "sources" (list[dict]): Metadata of the retrieved chunks used as context.
        """
        sources, context = self._retrieve(query, k)

        prompt = f"""You are a helpful assistant answering questions about a document.
        Use ONLY the context below to answer. If the answer isn't in the context,
        say "I couldn't find that in the document."

        Context:
        {context}

        Question: {query}

        Answer concisely and cite page numbers like [Page X]:"""

        # LLM MODEL — reads retrieved chunks and writes the final answer via Groq API (free)
        # Swap the model string below — all free on console.groq.com/docs/models:
        #   "llama-3.1-8b-instant"       → fastest,  good quality              (current)
        #   "llama-3.3-70b-versatile"    → medium,   best quality on Groq
        #   "mixtral-8x7b-32768"         → fast,     32k context window, good for long PDFs
        #   "gemma2-9b-it"               → fast,     Google's model, different reasoning style
        # To switch provider entirely, see alternatives below:
        #   Ollama (fully local, offline): ollama.chat(model="llama3.2", ...)
        #   Gemini Flash (Google, free):   google.generativeai model="gemini-1.5-flash"
        #   OpenAI (paid, cheapest):       client.chat.completions model="gpt-4o-mini"
        response = self.groq.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=512
        )

        return {
            "answer": (response.choices[0].message.content or "").strip(),
            "sources": sources
        }

    def query_with_image(self, query: str, image_bytes: bytes,k: int = 3, use_pdf_context: bool = True) -> dict:
        """Read an image and answer a question about its content.

        Args:
            question:        Natural language question about the image.
            image_bytes:     Raw bytes of the image file (JPEG, PNG, or WEBP).
            use_pdf_context: If True and a PDF is loaded, retrieves relevant
                            PDF chunks to supplement the image context.

        Returns:
            Dict with keys:
                answer  (str)  — Answer generated from the image content.
                sources (list) — PDF citations if use_pdf_context is True,
                                otherwise an empty list.
        """

        # Encode the image to base64
        # We are using Vision API, which accepts images as base64-encoded strings. The encoding is done in-memory without saving to disk.
        image_b64 = base64.standard_b64encode(image_bytes).decode("utf-8")

        # Detect the image format for the prompt (e.g., "jpeg", "png", "webp")
        # This is a simple heuristic based on common file signatures. For production, consider using a
        # robust library like python-magic to detect MIME types.
        if image_bytes.startswith(b"\xff\xd8\xff"):
            image_format = "jpeg"
        elif image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
            image_format = "png"
        elif image_bytes.startswith(b"RIFF") and b"WEBP" in image_bytes[8:16]:
            image_format = "webp"
        else:
            image_format = "jpeg"  # Default to jpeg if unknown

        # optionally retrieve PDF context if requested and available
        sources: list = []
        context_block: str = ""
        if use_pdf_context and self.collection.count() > 0:
            sources, pdf_context = self._retrieve(query, k)
            context_block = "\n\nAdditional context from the uploaded PDF:\n" + pdf_context

        # Construct the prompt for the vision-capable LLM
        prompt = f"""You are a helpful assistant. The user has shared an image with a question.

First, carefully read ALL content visible in the image:
- Read every word of text, headings, labels, and captions
- If there is a chart: read the title, axis labels, legend, and key data values
- If there is a table: read each row and column value
- If there is code: read it line by line
- If there is handwriting: transcribe it as accurately as possible
- If there is a diagram: describe the components and their relationships

Then use that content to answer the question accurately and completely.{context_block}

Question: {query}

Answer based on what you see in the image:"""

        # VISION MODEL — reads the image and generates an answer via Groq API
        # Swap the model string below — all available on console.groq.com/docs/models:
        #   "meta-llama/llama-4-scout-17b-16e-instruct"  → fast,   vision + text, 750 t/sec  (current)
        #   "meta-llama/llama-4-maverick-17b-128e-instruct" → slower, stronger reasoning, larger context
        #   "llama-3.2-11b-vision-preview"               → lighter, faster, lower quality
        response = self.groq.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/{image_format};base64,{image_b64}"
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ],
            temperature=0.1, # slightly higher than text-only — image description needs flexibility
            max_tokens=1024, # images can contain more information, so allow a longer answer
        )

        return {
            "answer": (response.choices[0].message.content or "").strip(),
            "sources": sources if use_pdf_context else []
        }

