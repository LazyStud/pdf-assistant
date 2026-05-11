# Graph Report - pdf-assistant  (2026-05-12)

## Corpus Check
- 1 files · ~1,204 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 26 nodes · 38 edges · 6 communities (4 shown, 2 thin omitted)
- Extraction: 89% EXTRACTED · 11% INFERRED · 0% AMBIGUOUS · INFERRED: 4 edges (avg confidence: 0.88)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `4c7874b5`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- [[_COMMUNITY_RAGPipeline Core Class|RAGPipeline Core Class]]
- [[_COMMUNITY_Text Query & Embedding|Text Query & Embedding]]
- [[_COMMUNITY_Module & Imports|Module & Imports]]
- [[_COMMUNITY_Vision Query & Image Handling|Vision Query & Image Handling]]
- [[_COMMUNITY_Groq LLM Client & Init|Groq LLM Client & Init]]
- [[_COMMUNITY_Text Chunking Strategy|Text Chunking Strategy]]

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
  src/rag_pipeline.py → src/rag_pipeline.py  _Bridges community 2 → community 1_
- `RAGPipeline` --implements--> `RAGPipeline.__init__`  [EXTRACTED]
  src/rag_pipeline.py → src/rag_pipeline.py  _Bridges community 2 → community 3_
- `RAGPipeline.__init__` --calls--> `RecursiveCharacterTextSplitter (500/50)`  [EXTRACTED]
  src/rag_pipeline.py → src/rag_pipeline.py  _Bridges community 3 → community 5_
- `RAGPipeline.ingest` --calls--> `RecursiveCharacterTextSplitter (500/50)`  [EXTRACTED]
  src/rag_pipeline.py → src/rag_pipeline.py  _Bridges community 2 → community 5_
- `RAGPipeline.query_with_image` --calls--> `ChromaDB Collection (pdf_chunks, cosine)`  [EXTRACTED]
  src/rag_pipeline.py → src/rag_pipeline.py  _Bridges community 1 → community 3_

## Hyperedges (group relationships)
- **Retrieval-Augmented Generation Flow** — rag_pipeline_ingest, rag_pipeline_embedder, rag_pipeline_chroma_collection, rag_pipeline_splitter, rag_pipeline_query, rag_pipeline_text_model [INFERRED 0.95]
- **Multimodal (Image + PDF Context) Query Flow** — rag_pipeline_query_with_image, rag_pipeline_embedder, rag_pipeline_chroma_collection, rag_pipeline_vision_model, rag_pipeline_groq_client [INFERRED 0.90]

## Communities (6 total, 2 thin omitted)

### Community 0 - "RAGPipeline Core Class"
Cohesion: 0.25
Nodes (5): RAGPipeline, Read an image and answer a question about its content.          Args:, Initialize the RAG pipeline: LLM client, embedding model, vector store, and text, Extract text from a PDF, split into chunks, embed, and store in ChromaDB., Embed a user query, retrieve the most relevant chunks, and generate an LLM answe

### Community 1 - "Text Query & Embedding"
Cohesion: 0.5
Nodes (4): Image Format Detection Heuristic, RAGPipeline.query_with_image, Text LLM (llama-3.1-8b-instant), Vision LLM (llama-4-scout-17b)

### Community 2 - "Module & Imports"
Cohesion: 0.83
Nodes (4): SentenceTransformer Embedder (all-MiniLM-L6-v2), RAGPipeline.ingest, RAGPipeline.query, RAGPipeline

### Community 3 - "Vision Query & Image Handling"
Cohesion: 0.5
Nodes (4): ChromaDB Collection (pdf_chunks, cosine), Cosine Similarity for Text Embeddings, Groq LLM Client, RAGPipeline.__init__

## Knowledge Gaps
- **9 isolated node(s):** `RAG pipeline for PDF question-answering using ChromaDB, SentenceTransformers, an`, `Initialize the RAG pipeline: LLM client, embedding model, vector store, and text`, `Extract text from a PDF, split into chunks, embed, and store in ChromaDB.`, `Embed a user query, retrieve the most relevant chunks, and generate an LLM answe`, `Read an image and answer a question about its content.          Args:` (+4 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **2 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `RAGPipeline` connect `RAGPipeline Core Class` to `Groq LLM Client & Init`?**
  _High betweenness centrality (0.147) - this node is a cross-community bridge._
- **Why does `RAGPipeline.query_with_image` connect `Text Query & Embedding` to `Module & Imports`, `Vision Query & Image Handling`?**
  _High betweenness centrality (0.082) - this node is a cross-community bridge._
- **Why does `RAGPipeline.ingest` connect `Module & Imports` to `Text Query & Embedding`, `Vision Query & Image Handling`, `Text Chunking Strategy`?**
  _High betweenness centrality (0.051) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `RAGPipeline.query_with_image` (e.g. with `RAGPipeline.ingest` and `RAGPipeline.query`) actually correct?**
  _`RAGPipeline.query_with_image` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `RAGPipeline.query` (e.g. with `RAGPipeline.ingest` and `RAGPipeline.query_with_image`) actually correct?**
  _`RAGPipeline.query` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `RAGPipeline.ingest` (e.g. with `RAGPipeline.query` and `RAGPipeline.query_with_image`) actually correct?**
  _`RAGPipeline.ingest` has 2 INFERRED edges - model-reasoned connections that need verification._
- **What connects `RAG pipeline for PDF question-answering using ChromaDB, SentenceTransformers, an`, `Initialize the RAG pipeline: LLM client, embedding model, vector store, and text`, `Extract text from a PDF, split into chunks, embed, and store in ChromaDB.` to the rest of the system?**
  _9 weakly-connected nodes found - possible documentation gaps or missing edges._