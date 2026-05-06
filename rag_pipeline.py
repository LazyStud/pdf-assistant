import os
from groq import Groq
import fitz  # PyMuPDF
from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from torch import embedding

load_dotenv()  # Load environment variables from .env file

class RAGpipeline:
    def __init__(self):
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
        # Extract text from PDF using PyMuPDF per page to avoid memory issues with large PDFs
        doc = fitz.open(pdf_path)
        pages = []
        for i,page in enumerate(doc):
            text = page.get_text()
            if text.strip():  # Only process non-empty pages
                pages.append(text)
        
        # Chunk every page separately to manage memory better, then embed and store in ChromaDB
        chunks, metadata, ids = [], [], []
        for page_data in pages:
            # Split text into chunks
            page_chunks = self.splitter.split_text(page_data["text"])
            for j, chunk in enumerate(page_chunks):
                chunks.append(chunk)
                metadata.append({"page": page_data["page"], "text": chunk})
                ids.append(f"p{page_data['page']}_c{j}")

        # Embed all chunks in one batch to optimize performance
        embeddings = self.embedder.encode(chunks, show_progress_bar=True).tolist()

        # Store in ChromaDB
        self.collection.add(
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadata,
            ids=ids
        )

        return len(chunks)
        

    def query(self, query):
        """embeded query -> searcch database -> return chunks -> get answer from LLM"""
        response = self.groq.query(query)
        return response