"""RAG pipeline for PDF question-answering using ChromaDB, SentenceTransformers, and Groq.

Architecture
------------
                ┌──────────────────────────────────────────────────┐
                │                  RAGPipeline                     │
                │                                                  │
PDF file  ──► │  ingest()  →  chunk → embed → ChromaDB           │
                │                                                  │
text query ──► │  query()   →  embed query → retrieve → Groq LLM  │ ──► answer + sources
                │                                                  │
image+query ──►│  query_with_image() → base64 + retrieve → Groq  │ ──► answer + sources
                │                       vision LLM                 │
                └──────────────────────────────────────────────────┘

Components
----------
- SentenceTransformer ("all-MiniLM-L6-v2"): local CPU-only embedding model, no API needed.
- ChromaDB (in-memory, cosine similarity): HNSW vector index for fast nearest-neighbour search.
- RecursiveCharacterTextSplitter (500 chars / 50 overlap): splits PDF text into overlapping
chunks that preserve sentence context across boundaries.
- Groq LLM ("llama-3.1-8b-instant"): free-tier inference for text-only Q&A.
- Groq Vision LLM ("meta-llama/llama-4-scout-17b-16e-instruct"): multimodal inference that
accepts base64-encoded images and optional PDF context.
"""

import os
from typing import Any
import base64
from groq import Groq
import fitz  # PyMuPDF
from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()  # Reads GROQ_API_KEY (and any other vars) from .env into os.environ


class RAGPipeline:
    """End-to-end pipeline: PDF ingestion, vector storage, text Q&A, and vision Q&A.

    One instance per uploaded document — call ingest() once, then query() or
    query_with_image() as many times as needed.  All state (vectors, metadata)
    lives in-memory inside self.collection and is discarded when the object is
    garbage-collected.
    """

    def __init__(self) -> None:
        """Set up the Groq client, embedding model, ChromaDB collection, and text splitter.

        Raises:
            ValueError: If GROQ_API_KEY is missing from the environment / .env file.
        """
        # Groq client — requires GROQ_API_KEY set in .env (see README)
        self.groq = Groq(api_key=os.getenv("GROQ_API_KEY"))

        # LOCAL EMBEDDING MODEL — converts text chunks and queries to 384-d float vectors.
        # Runs entirely on CPU; no API key or internet connection required after first download.
        #
        # Swap the model string here (and nowhere else — _retrieve reuses self.embedder) to
        # change the quality/speed/size tradeoff:
        #
        #   Model                                    Size    Speed    Quality
        #   "all-MiniLM-L6-v2"                      90 MB   fastest  good       ← default
        #   "BAAI/bge-small-en-v1.5"                130 MB  fast     better     (recommended upgrade)
        #   "all-mpnet-base-v2"                      420 MB  medium   good
        #   "BAAI/bge-large-en-v1.5"                1.3 GB  slow     best
        #   "paraphrase-multilingual-MiniLM-L12-v2" 470 MB  medium   good       (non-English PDFs)
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

        # IN-MEMORY CHROMADB — no external server, no disk writes, no Docker needed.
        # Data is lost when the Python process exits; that's intentional for per-session privacy.
        self.chroma = chromadb.Client()
        self.collection = self.chroma.get_or_create_collection(
            name="pdf_chunks",
            metadata={"hnsw:space": "cosine"},  # cosine similarity works best for sentence embeddings
        )

        # TEXT SPLITTER — breaks raw page text into overlapping chunks so that:
        #   • each chunk fits comfortably inside the LLM's context window
        #   • the 50-char overlap prevents a sentence from being cut in half at a boundary
        # Separators are tried in order: prefer paragraph breaks, then line breaks, then spaces.
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", " ", ". "],
        )

    def ingest(self, pdf_path: str) -> int:
        """Extract text from a PDF, chunk it, embed the chunks, and index them in ChromaDB.

        Pages with no extractable text (scanned images, blank pages) are silently skipped.
        Each chunk gets a unique ID in the form "p<page>_c<chunk_index>" (e.g. "p3_c1"),
        and its metadata stores the originating page number plus the raw text so the UI
        can display source citations.

        Args:
            pdf_path: Absolute or relative path to the PDF file to process.

        Returns:
            Total number of text chunks successfully indexed into the vector store.
        """
        doc = fitz.open(pdf_path)

        # All three lists are built in lockstep — index N in each list describes the same chunk.
        chunks: list[str] = []
        metadatas: list[dict[str, Any]] = []  # page + text preview, surfaced as UI citations
        ids: list[str] = []                   # ChromaDB requires a unique string ID per document

        # fitz page objects don't declare __iter__ in their type stubs, so index manually
        for i in range(len(doc)):
            page_num = i + 1
            page: fitz.Page = doc[i]
            text: str = page.get_text()  # type: ignore[assignment]

            if not text.strip():
                continue  # skip blank or image-only pages

            page_chunks: list[str] = self.splitter.split_text(text)

            for j, chunk in enumerate(page_chunks):
                chunks.append(chunk)
                metadatas.append({
                    "page": page_num,  # shown as "Page N" in the sidebar source expander
                    "text": chunk,     # shown as the 200-char preview beneath the citation
                })
                ids.append(f"p{page_num}_c{j}")

        # Encode all chunks in one call — batch encoding is ~10x faster than encoding one at a time.
        # Output shape: (num_chunks, 384). .tolist() converts numpy ndarray → plain Python list
        # because ChromaDB's type stubs expect List[List[float]].
        embeddings = self.embedder.encode(chunks, show_progress_bar=True).tolist()

        # Persist chunks + vectors + metadata into the in-memory HNSW index.
        self.collection.add(
            documents=chunks,
            embeddings=embeddings,  # type: ignore[arg-type]  # runtime-correct; stubs are too strict
            metadatas=metadatas,    # type: ignore[arg-type]
            ids=ids,
        )

        return len(chunks)

    def _retrieve(self, query: str, k: int) -> tuple[list[dict[str, Any]], str]:
        """Embed *query* and return the top-k most similar chunks from the vector store.

        Args:
            query: Natural-language string to search for.
            k:     Number of nearest-neighbour chunks to retrieve.

        Returns:
            A 2-tuple of:
                sources (list[dict]): Chunk metadata dicts — each has "page" (int) and
                    "text" (str) keys — used by the UI to render source citations.
                context (str): The chunk texts joined by "\\n\\n--\\n\\n", ready to be
                    pasted directly into an LLM prompt as the RAG context block.
        """
        embedding = self.embedder.encode([query]).tolist()
        results = self.collection.query(
            query_embeddings=embedding,  # type: ignore[arg-type]
            n_results=k,
        )
        raw_metadatas = results["metadatas"] or [[]]
        raw_documents = results["documents"] or [[]]
        sources: list[dict[str, Any]] = [dict(m) for m in raw_metadatas[0]]  # type: ignore[misc]
        context = "\n\n--\n\n".join(raw_documents[0])
        return sources, context

    def query(self, query: str, k: int = 4) -> dict[str, Any]:
        """Answer a natural-language question using retrieved PDF context.

        Retrieves the top-k chunks most semantically similar to *query*, constructs a
        grounded prompt that instructs the LLM to answer only from that context, and
        returns the LLM response together with the source chunk metadata for citations.

        Args:
            query: Natural-language question to answer from the indexed document.
            k:     Number of context chunks to retrieve (default 4).

        Returns:
            dict with keys:
                "answer"  (str):        LLM-generated response with inline [Page N] citations.
                "sources" (list[dict]): Metadata of each retrieved chunk (keys: "page", "text").
        """
        sources, context = self._retrieve(query, k)

        prompt = f"""You are a helpful assistant answering questions about a document.
        Use ONLY the context below to answer. If the answer isn't in the context,
        say "I couldn't find that in the document."

        Context:
        {context}

        Question: {query}

        Answer concisely and cite page numbers like [Page X]:"""

        # TEXT LLM — generates the final answer from the retrieved context via Groq (free tier).
        # Swap the model string to change speed/quality:
        #
        #   Model                      Speed    Quality   Notes
        #   "llama-3.1-8b-instant"     fastest  good      ← default
        #   "llama-3.3-70b-versatile"  medium   best on Groq
        #   "mixtral-8x7b-32768"       fast     good      32k context — better for long PDFs
        #   "gemma2-9b-it"             fast     good      Google's model, different reasoning style
        #
        # To switch provider entirely, replace self.groq in __init__:
        #   Ollama (local, offline):  use ollama-python client, model="llama3.2"
        #   Gemini Flash (free):      use google-generativeai, model="gemini-1.5-flash"
        #   OpenAI (paid, cheapest):  use openai client, model="gpt-4o-mini"
        response = self.groq.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,   # low temperature = more factual, less creative hallucination
            max_tokens=512,
        )

        return {
            "answer": (response.choices[0].message.content or "").strip(),
            "sources": sources,
        }

    def query_with_image(
        self,
        query: str,
        image_bytes: bytes,
        k: int = 3,
        use_pdf_context: bool = True,
    ) -> dict:
        """Answer a question about an image, optionally grounded by PDF context.

        The image is base64-encoded in-memory (never written to disk) and sent to
        a vision-capable LLM alongside a structured prompt.  When *use_pdf_context*
        is True and a PDF has already been ingested, the top-k most relevant PDF
        chunks are retrieved and appended to the prompt so the model can cross-
        reference the image against the document.

        Image format is detected from the file's magic bytes.  Unknown formats fall
        back to "jpeg", which is accepted by all Groq vision models.

        Args:
            query:           Natural-language question about the image.
            image_bytes:     Raw bytes of the image file (JPEG, PNG, or WEBP).
            k:               Number of PDF context chunks to retrieve (default 3).
            use_pdf_context: If True and a PDF is ingested, retrieves relevant PDF
                            chunks to supplement the image context (default True).

        Returns:
            dict with keys:
                "answer"  (str):        Answer generated from the image (and PDF context).
                "sources" (list[dict]): PDF citations if use_pdf_context is True and a
                                        PDF has been ingested; empty list otherwise.
        """
        # Vision API requires the image as a base64 data-URI; encode entirely in memory.
        image_b64 = base64.standard_b64encode(image_bytes).decode("utf-8")

        # Detect MIME type from magic bytes so the data-URI has the correct content-type header.
        # python-magic would be more robust for production, but these three signatures cover
        # all formats currently accepted by Groq vision models.
        if image_bytes.startswith(b"\xff\xd8\xff"):
            image_format = "jpeg"
        elif image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
            image_format = "png"
        elif image_bytes.startswith(b"RIFF") and b"WEBP" in image_bytes[8:16]:
            image_format = "webp"
        else:
            image_format = "jpeg"  # safe fallback; Groq rejects unknown types gracefully

        # Optionally pull relevant PDF passages to give the vision model document context.
        sources: list = []
        context_block: str = ""
        if use_pdf_context and self.collection.count() > 0:
            sources, pdf_context = self._retrieve(query, k)
            context_block = "\n\nAdditional context from the uploaded PDF:\n" + pdf_context

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

        # VISION LLM — accepts image data-URIs via the "image_url" content block.
        # Swap the model string to change capability/speed:
        #
        #   Model                                            Speed   Notes
        #   "meta-llama/llama-4-scout-17b-16e-instruct"     fast    vision + text, ~750 tok/s  ← default
        #   "meta-llama/llama-4-maverick-17b-128e-instruct" slower  stronger reasoning, larger context
        #   "llama-3.2-11b-vision-preview"                  faster  lighter weight, lower quality
        response = self.groq.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/{image_format};base64,{image_b64}"},
                        },
                        {
                            "type": "text",
                            "text": prompt,
                        },
                    ],
                }
            ],
            temperature=0.1,   # low temp keeps answers factual; vision still needs slight flexibility
            max_tokens=1024,   # images can pack more info than plain text — allow a longer reply
        )

        return {
            "answer": (response.choices[0].message.content or "").strip(),
            "sources": sources if use_pdf_context else [],
        }

