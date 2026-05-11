# Graph Report - .  (2026-05-12)

## Corpus Check
- Corpus is ~1,248 words - fits in a single context window. You may not need a graph.

## Summary
- 26 nodes · 38 edges · 7 communities (4 shown, 3 thin omitted)
- Extraction: 89% EXTRACTED · 11% INFERRED · 0% AMBIGUOUS · INFERRED: 4 edges (avg confidence: 0.88)
- Token cost: 19,798 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_RAGPipeline Core Class|RAGPipeline Core Class]]
- [[_COMMUNITY_Text Query & Embedding|Text Query & Embedding]]
- [[_COMMUNITY_Module & Imports|Module & Imports]]
- [[_COMMUNITY_Vision Query & Image Handling|Vision Query & Image Handling]]
- [[_COMMUNITY_Groq LLM Client & Init|Groq LLM Client & Init]]
- [[_COMMUNITY_Text Chunking Strategy|Text Chunking Strategy]]
- [[_COMMUNITY_Vector Store & Similarity|Vector Store & Similarity]]

## God Nodes (most connected - your core abstractions)
1. `RAGPipeline.query_with_image` - 8 edges
2. `RAGPipeline.query` - 7 edges
3. `RAGPipeline.ingest` - 6 edges
4. `RAGPipeline` - 5 edges
5. `RAGPipeline.__init__` - 5 edges
6. `ChromaDB Collection (pdf_chunks, cosine)` - 5 edges
7. `RAGPipeline` - 4 edges
8. `SentenceTransformer Embedder (all-MiniLM-L6-v2)` - 4 edges
9. `Groq LLM Client` - 3 edges
10. `RecursiveCharacterTextSplitter (500/50)` - 3 edges

## Surprising Connections (you probably didn't know these)
- `RAGPipeline.ingest` --shares_data_with--> `RAGPipeline.query_with_image`  [INFERRED]
  src/rag_pipeline.py → src/rag_pipeline.py  _Bridges community 1 → community 3_
- `RAGPipeline` --implements--> `RAGPipeline.ingest`  [EXTRACTED]
  src/rag_pipeline.py → src/rag_pipeline.py  _Bridges community 4 → community 1_
- `RAGPipeline` --implements--> `RAGPipeline.query_with_image`  [EXTRACTED]
  src/rag_pipeline.py → src/rag_pipeline.py  _Bridges community 4 → community 3_
- `RAGPipeline.__init__` --calls--> `ChromaDB Collection (pdf_chunks, cosine)`  [EXTRACTED]
  src/rag_pipeline.py → src/rag_pipeline.py  _Bridges community 4 → community 6_
- `RAGPipeline.__init__` --calls--> `RecursiveCharacterTextSplitter (500/50)`  [EXTRACTED]
  src/rag_pipeline.py → src/rag_pipeline.py  _Bridges community 4 → community 5_

## Hyperedges (group relationships)
- **Retrieval-Augmented Generation Flow** — rag_pipeline_ingest, rag_pipeline_embedder, rag_pipeline_chroma_collection, rag_pipeline_splitter, rag_pipeline_query, rag_pipeline_text_model [INFERRED 0.95]
- **Multimodal (Image + PDF Context) Query Flow** — rag_pipeline_query_with_image, rag_pipeline_embedder, rag_pipeline_chroma_collection, rag_pipeline_vision_model, rag_pipeline_groq_client [INFERRED 0.90]

## Communities (7 total, 3 thin omitted)

### Community 0 - "RAGPipeline Core Class"
Cohesion: 0.25
Nodes (5): RAGPipeline, Read an image and answer a question about its content.          Args:, Initialize the RAG pipeline: LLM client, embedding model, vector store, and text, Extract text from a PDF, split into chunks, embed, and store in ChromaDB., Embed a user query, retrieve the most relevant chunks, and generate an LLM answe

### Community 1 - "Text Query & Embedding"
Cohesion: 0.67
Nodes (4): SentenceTransformer Embedder (all-MiniLM-L6-v2), RAGPipeline.ingest, RAGPipeline.query, Text LLM (llama-3.1-8b-instant)

### Community 3 - "Vision Query & Image Handling"
Cohesion: 0.67
Nodes (3): Image Format Detection Heuristic, RAGPipeline.query_with_image, Vision LLM (llama-4-scout-17b)

### Community 4 - "Groq LLM Client & Init"
Cohesion: 0.67
Nodes (3): Groq LLM Client, RAGPipeline.__init__, RAGPipeline

## Knowledge Gaps
- **9 isolated node(s):** `RAG pipeline for PDF question-answering using ChromaDB, SentenceTransformers, an`, `Initialize the RAG pipeline: LLM client, embedding model, vector store, and text`, `Extract text from a PDF, split into chunks, embed, and store in ChromaDB.`, `Embed a user query, retrieve the most relevant chunks, and generate an LLM answe`, `Read an image and answer a question about its content.          Args:` (+4 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **3 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `RAGPipeline` connect `RAGPipeline Core Class` to `Module & Imports`?**
  _High betweenness centrality (0.147) - this node is a cross-community bridge._
- **Why does `RAGPipeline.query_with_image` connect `Vision Query & Image Handling` to `Text Query & Embedding`, `Groq LLM Client & Init`, `Vector Store & Similarity`?**
  _High betweenness centrality (0.082) - this node is a cross-community bridge._
- **Why does `RAGPipeline.ingest` connect `Text Query & Embedding` to `Vision Query & Image Handling`, `Groq LLM Client & Init`, `Text Chunking Strategy`, `Vector Store & Similarity`?**
  _High betweenness centrality (0.051) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `RAGPipeline.query_with_image` (e.g. with `RAGPipeline.query` and `RAGPipeline.ingest`) actually correct?**
  _`RAGPipeline.query_with_image` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `RAGPipeline.query` (e.g. with `RAGPipeline.query_with_image` and `RAGPipeline.ingest`) actually correct?**
  _`RAGPipeline.query` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `RAGPipeline.ingest` (e.g. with `RAGPipeline.query` and `RAGPipeline.query_with_image`) actually correct?**
  _`RAGPipeline.ingest` has 2 INFERRED edges - model-reasoned connections that need verification._
- **What connects `RAG pipeline for PDF question-answering using ChromaDB, SentenceTransformers, an`, `Initialize the RAG pipeline: LLM client, embedding model, vector store, and text`, `Extract text from a PDF, split into chunks, embed, and store in ChromaDB.` to the rest of the system?**
  _9 weakly-connected nodes found - possible documentation gaps or missing edges._