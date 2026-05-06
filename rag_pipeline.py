import os
from groq import Groq
import fitz  # PyMuPDF
from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer

load_dotenv()  # Load environment variables from .env file

class RAGpipeline:
    def __init__(self):
        # LLM - Groq - make sure to set GROQ_API_KEY in your .env file
        self.groq = Groq(api_key = os.getenv("GROQ_API_KEY"))

        # Transformer model convert text to list of numbers (embeddings) - OpenAI's text-embedding-3-small
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")  # You can choose other models from sentence-transformers

        # In memory vector database - no server needed - Chroma
        self.chroma = chromadb.Client()
        self.collection = self.chroma.get_or_create_collection(name="pdf_chunks")

    def ingest(self, pdf_path: str) -> int:
        """Ingest pdf -> break into chunks -> embed -> store in vector database"""
        # Extract text from PDF using PyMuPDF
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text

    def query(self, query):
        """embeded query -> searcch database -> return chunks -> get answer from LLM"""
        response = self.groq.query(query)
        return response