import os
from groq import Groq
import fitz  # PyMuPDF
from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from torch import embedding

load_dotenv()  # Load environment variables from .env file

class RAGPipeline:
    def __init__(self) -> None:
        # LLM - Groq - make sure to set GROQ_API_KEY in your .env file
        self.groq = Groq(api_key = os.getenv("GROQ_API_KEY"))

        # Transformer model convert text to list of numbers (embeddings)
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")  # You can choose other models from sentence-transformers

        # In memory vector database - no server needed - Chroma
        self.chroma = chromadb.Client()
        self.collection = self.chroma.get_or_create_collection(name="pdf_chunks")

        # 500 chars per chunk with 50 chars overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", " ", ". "]
        )

    def ingest(self, pdf_path: str) -> int:
        """Ingest pdf -> break into chunks -> embed -> store in vector database"""
        # Open the PDF file with PyMuPDF
        doc = fitz.open(pdf_path)

        # These three lists must stay in sync — same index = same chunk
        chunks = []     # raw text of each chunk
        metadatas = []  # page number + text, shown as citations in the UI
        ids = []        # unique ID for each chunk (required by ChromaDB)

        # ── Loop over every page ─────────────────────────────────────────────
        # enumerate(doc, start=1) gives (1, page1), (2, page2), ...
        for page_num, page in enumerate(doc, start=1):

            # get_text() extracts all readable text from the page
            text = page.get_text()

            # Skip pages with no text (scanned images, blank pages, etc.)
            if not text.strip():
                continue

            # Split this page's text into overlapping chunks
            page_chunks = self.splitter.split_text(text)

            # Add each chunk to the lists with its metadata
            for j, chunk in enumerate(page_chunks):
                chunks.append(chunk)

                # metadata is returned to the UI as the citation source
                metadatas.append({
                    "page": page_num,   # shown as "Page 3" in the UI
                    "text": chunk       # shown as preview text under the citation
                })

                # unique ID format: p3_c1 = page 3, chunk 1
                ids.append(f"p{page_num}_c{j}")

        # ── Embed all chunks in one batch ────────────────────────────────────
        # encode() runs the transformer model on every chunk
        # Batch processing is much faster than encoding one by one
        # Result shape: (num_chunks, 384) — each chunk → 384 numbers
        # .tolist() converts numpy array to plain Python list for ChromaDB
        embeddings = self.embedder.encode(
            chunks,
            show_progress_bar=True  # prints a progress bar in the terminal
        ).tolist()

        # ── Store in ChromaDB ────────────────────────────────────────────────
        # All three lists are stored together — linked by their shared index
        # ChromaDB builds an HNSW index for fast approximate nearest-neighbour search
        self.collection.add(
            documents=chunks,       # raw text (returned in query results)
            embeddings=embeddings,  # vectors (used for similarity search)
            metadatas=metadatas,    # page + text (returned as citation info)
            ids=ids                 # must be unique across the collection
        )

        # Return chunk count so the UI can show "Indexed 147 chunks"
        return len(chunks)
        

    def query(self, query: str) -> dict:
        """embeded query -> searcch database -> return chunks -> get answer from LLM"""

        # Enbed the query
        query_embedding = self.embedder.encode([query]).tolist()

        # Search in ChromaDB for top 4 relevant chunks
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=4
        )

        sources = results["metadatas"][0]  # Get metadata of retrieved chunks for source info

        # Build context from retrieved chunks
        context = "\n\n--\n\n".join(results["documents"][0])

        # Prompt for LLM - you can customize this as needed
        prompt = f"""You are a helpful assistant answering questions about a document.
        Use ONLY the context below to answer. If the answer isn't in the context,
        say "I couldn't find that in the document."

        Context:
        {context}

        Question: {query}

        Answer concisely and cite page numbers like [Page X]:"""

        # Call Groq LLM with the prompt
        response = self.groq.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=512
        )

        return {
            "answer": response.choices[0].message.content.strip(),
            "sources": sources
        }