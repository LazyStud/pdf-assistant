import os
from groq import Groq
import fitz  # PyMuPDF
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

class RAGpipeline:
    def __init__(self):
        self.groq = Groq(api_key = os.getenv("GROQ_API_KEY"))

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