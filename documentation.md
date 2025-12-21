# Vectra Python Documentation

## 1. Getting Started
- Introduction
  - Vectra is a provider-agnostic RAG SDK for Python that orchestrates: load files, chunk, embed, store, retrieve, rerank, and generate answers with async streaming support.
- Key Features
  - Multi-Provider (OpenAI, Gemini, Anthropic, OpenRouter, HuggingFace)
  - HyDE and Multi-Query retrieval strategies
  - Hybrid Search with RRF fusion (vector + keyword)
  - Agentic Chunking using an LLM to find semantic breaks
  - Streaming responses and metadata enrichment
- Architecture
  ```mermaid
  graph LR
      A[Files] --> B(Chunking)
      B --> C{Embedding API}
      C --> D[(Vector Store)]
      E[User Query] --> F(Retrieval)
      D --> F
      F --> G(Reranking)
      G --> H[LLM Generation]
      H --> I[Stream Output]
  ```
- Installation
  - Prerequisites: Python 3.8+, optional Postgres (for Prisma + pgvector)
  - Commands:
    - `pip install vectra-py`
    - `uv pip install vectra-py`
    - CLI available as `vectra`; alternative: `python -m vectra.cli`
- Quickstart
  - Minimal setup with ChromaDB to avoid Postgres in first run:
  ```python
  import asyncio, os
  from chromadb import Client
  from vectra import VectraClient, VectraConfig, ProviderType, RetrievalStrategy
  
  async def main():
      chroma = Client()
      config = VectraConfig(
          embedding={ 'provider': ProviderType.OPENAI, 'api_key': os.getenv('OPENAI_API_KEY'), 'model_name': 'text-embedding-3-small' },
          llm={ 'provider': ProviderType.GEMINI, 'api_key': os.getenv('GOOGLE_API_KEY'), 'model_name': 'gemini-1.5-pro-latest' },
          database={ 'type': 'chroma', 'client_instance': chroma, 'table_name': 'rag_collection' },
          retrieval={ 'strategy': RetrievalStrategy.NAIVE }
      )
      client = VectraClient(config)
      await client.ingest_documents('./docs/hello.txt')
      res = await client.query_rag('Hello')
      print(res['answer'])
  
  asyncio.run(main())
  ```
  - Environment Variables
    - `OPENAI_API_KEY`, `GOOGLE_API_KEY`, `ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY`, `HUGGINGFACE_API_KEY`
  - First Query
    - `await client.query_rag("Hello")` returns `{ 'answer': str | dict, 'sources': list }`.

## 2. Fundamentals
- Configuration
  - Centralized Pydantic `VectraConfig` object validating providers, database, and pipeline options.
  - Copy-Paste template:
  ```python
  import os
  from vectra import VectraConfig, ProviderType, ChunkingStrategy, RetrievalStrategy

  redis_client = None
  postgres_client = None
  
  config = VectraConfig(
    # Embedding
    embedding={
      'provider': ProviderType.OPENAI,
      'api_key': os.getenv('OPENAI_API_KEY'),
      'model_name': 'text-embedding-3-small',
      # 'dimensions': 1536
    },
    # LLM (generation)
    llm={
      'provider': ProviderType.GEMINI,
      'api_key': os.getenv('GOOGLE_API_KEY'),
      'model_name': 'gemini-1.5-pro-latest',
      # 'temperature': 0.3,
      # 'max_tokens': 1024,
      # 'default_headers': {} # OpenRouter only
    },
    # Memory (toggleable, defaults off)
    memory={
      'enabled': False,
      'type': 'in-memory',  # or 'redis' | 'postgres'
      'max_messages': 20,
      # Redis options
      'redis': {
        'client_instance': redis_client,
        'key_prefix': 'vectra:chat:'
      },
      # Postgres options
      'postgres': {
        'client_instance': postgres_client,
        'table_name': 'ChatMessage',
        'column_map': { 'sessionId': 'sessionId', 'role': 'role', 'content': 'content', 'createdAt': 'createdAt' }
      }
    },
    # Ingestion (rate limit is toggleable, defaults off)
    ingestion={ 'rate_limit_enabled': False, 'concurrency_limit': 5 },
    # Database
    database={
      'type': 'chroma', # 'prisma' | 'qdrant' | 'milvus'
      'client_instance': None, # your DB client
      'table_name': 'Document',
      'column_map': { 'content': 'content', 'vector': 'embedding', 'metadata': 'metadata' } # Prisma only
    },
    # Chunking
    chunking={
      'strategy': ChunkingStrategy.RECURSIVE, # or ChunkingStrategy.AGENTIC
      'chunk_size': 1000,
      'chunk_overlap': 200,
      # 'separators': ['\n\n', '\n', ' ', '']
      # 'agentic_llm': { 'provider': ProviderType.OPENAI, 'api_key': os.getenv('OPENAI_API_KEY'), 'model_name': 'gpt-4o-mini' } # AGENTIC
    },
    # Retrieval
    retrieval={
      'strategy': RetrievalStrategy.HYBRID, # NAIVE | HYDE | MULTI_QUERY | HYBRID | MMR
      # 'llm_config': { 'provider': ProviderType.OPENAI, 'api_key': os.getenv('OPENAI_API_KEY'), 'model_name': 'gpt-4o-mini' }, # HYDE/MULTI_QUERY
      # 'hybrid_alpha': 0.5
      # 'mmr_lambda': 0.5,
      # 'mmr_fetch_k': 20
    },
    # Reranking
    reranking={
      'enabled': False,
      # 'top_n': 5,
      # 'window_size': 20,
      # 'llm_config': { 'provider': ProviderType.ANTHROPIC, 'api_key': os.getenv('ANTHROPIC_API_KEY'), 'model_name': 'claude-3-haiku' }
    },
    # Metadata
    metadata={ 'enrichment': False }, # summary, keywords, hypothetical_questions
    # Query Planning
    query_planning={ 'token_budget': 2048, 'prefer_summaries_below': 1024, 'include_citations': True },
    # Grounding
    grounding={ 'enabled': False, 'strict': False, 'max_snippets': 4 },
    # Generation
    generation={ 'output_format': 'text', 'structured_output': 'none' }, # 'json' and 'citations' supported
    # Prompts
    prompts={ 'query': 'Use only the following context.\nContext:\n{{context}}\n\nQ: {{question}}' },
    # Callbacks
    callbacks=[]
  )
  ```
- Ingestion
  - File Loading: PDF, DOCX, TXT, MD, XLSX
  - Directory Walking: `await client.ingest_documents('./folder')` recursively processes supported files
  - Index Management (Postgres/Prisma): `await client.vector_store.ensure_indexes()` after ingestion
- Querying
  - Standard:
  ```python
  result = await client.query_rag("Question")
  print(result['answer'])
  ```
  - Streaming:
  ```python
  stream = await client.query_rag("Draft a welcome email...", stream=True)
  async for chunk in stream:
      print(chunk.get('delta', ''), end='')
  ```
  - Stateful Chat (Memory):
  ```python
  result = await client.query_rag("Does this apply to contractors?", session_id="user-123")
  ```
  - Filtering:
  ```python
  result = await client.query_rag("Vacation policy", filter={ 'docTitle': 'Employee Handbook' })
  ```

## 3. Database & Vector Stores
- Supported Backends
  - Prisma (Postgres + pgvector): rich SQL, hybrid search and indexes
  - ChromaDB: simple local collections, easy first-run
  - Qdrant: high-performance vector search
  - Milvus: scalable vector database
- Prisma (Postgres + pgvector)
  - Prerequisite: enable `vector` extension
    - `CREATE EXTENSION IF NOT EXISTS vector;`
  - Schema
  ```prisma
  model Document {
    id        String                 @id @default(uuid())
    content   String
    metadata  Json
    embedding Unsupported("vector")? // pgvector type
    createdAt DateTime               @default(now())
  }
  ```
  - Column Mapping: `column_map` maps SDK fields to DB columns, e.g. `{ 'content': 'content', 'vector': 'embedding', 'metadata': 'metadata' }`
  - Index Management: ivfflat for vector cosine ops and GIN for FTS
    - `await client.vector_store.ensure_indexes()`
- ChromaDB / Qdrant / Milvus
  - Chroma: `from chromadb import Client; chroma = Client()`
  - Qdrant: `from qdrant_client import QdrantClient`
  - Milvus: `from pymilvus import MilvusClient`
  - Pass `client_instance` and `table_name` to `database` config.

## 4. Providers (LLM & Embeddings)
- Provider Setup
  - OpenAI:
    - Embeddings: `text-embedding-3-small`, `text-embedding-3-large`
    - Generation: `gpt-4o`, `gpt-4o-mini`
  - Gemini:
    - Generation: `gemini-1.5-pro-latest`
  - Anthropic:
    - Generation only (`claude-3-haiku`, `claude-3-opus`) — use a different embedding provider
  - Ollama:
    - Local development; set `provider = ProviderType.OLLAMA`
    - Defaults to `http://localhost:11434` (override with `base_url`)
  - OpenRouter:
    - Unified gateway; set `llm.provider = ProviderType.OPENROUTER` and `model_name` to e.g. `openai/gpt-4o`
  - HuggingFace:
    - Use Inference API for embeddings and generation with open-source models
- Customizing Models
  - `temperature`, `max_tokens`, and embedding `dimensions` (must match pgvector column for Prisma)

## 5. Advanced Concepts
- Chunking Strategies
  - Recursive: control `chunk_size`, `chunk_overlap`, and optional `separators`
  - Agentic: configure `chunking.agentic_llm`; uses an LLM to place semantic boundaries
- Retrieval Strategies
  - Naive: cosine similarity on vectors
  - HyDE: generate a hypothetical answer and search on its embedding
  - Hybrid Search (RRF): combine vector search and keyword FTS using reciprocal rank fusion
  - Multi-Query: produce query variations via LLM to improve recall
- Reranking
  - Enable with `reranking.enabled`; tune `top_n` and `window_size`
- Metadata Enrichment
  - Set `metadata.enrichment = True` to generate summaries, keywords, and hypothetical questions during ingestion
- Conversation Memory
  - Enable stateful chat by setting `memory` config and passing `session_id` to `query_rag`.
  - Automatically appends history to prompts and saves interactions.
- Production Evaluation
  - Use `client.evaluate(test_set)` to measure Faithfulness (answer derived from context) and Relevance (answer addresses question).
  - Returns per-test scores (0-1) for each question.
  ```python
  # Example Test Set structure
  test_set = [
    {
      'question': 'What is the remote work policy?',
      'expectedGroundTruth': 'Employees can work remotely up to 3 days a week.'
    }
  ]
  report = await client.evaluate(test_set)
  average_faithfulness = (sum(r.get('faithfulness', 0) for r in report) / len(report)) if report else 0
  average_relevance = (sum(r.get('relevance', 0) for r in report) / len(report)) if report else 0
  print({ 'average_faithfulness': average_faithfulness, 'average_relevance': average_relevance, 'report': report })
  ```

## 6. Production Guide
- Query Planning & Grounding
  - Token Budgets: `query_planning.token_budget`
  - Grounding: `grounding.enabled` and `grounding.strict` to restrict answers to grounded snippets
  - Citations: include titles/sections/pages via `query_planning.include_citations`; parse when using `generation.structured_output = 'citations'`
- Observability & Debugging
  - Logging: use `StructuredLoggingCallbackHandler` for structured events
  - Tracing: hook into pipeline events like `on_retrieval_start`, `on_generation_end`
- CLI Tools
  - Global `vectra` binary for ingestion and queries without writing code
  - `vectra ingest ./docs --config=./config.json`
  - `vectra query "What are the payment terms?" --config=./config.json --stream`

## 7. API Reference
- VectraClient
  - Constructor: `VectraClient(config: VectraConfig)`
  - Methods:
    - `async ingest_documents(path: str, ingestion_mode: str = "append") -> None`
    - `async query_rag(query: str, filter: dict | None = None, stream: bool = False, session_id: str | None = None) -> dict | AsyncGenerator[dict, None]`
    - `async list_documents(filter: dict | None = None, limit: int = 100) -> list[dict]`
    - `async delete_documents(filter: dict) -> int`
    - `async update_documents(filter: dict, update_data: dict) -> int`
- VectorStore Interface
  - Extend and implement:
    - `add_documents(documents)`
    - `similarity_search(vector, limit = 5, filter = None)`
    - Optional: `hybrid_search(text, vector, limit = 5, filter = None)`
    - `list_documents(filter = None, limit = 100)`
    - `delete_documents(filter) -> int`
    - `update_documents(filter, update_data) -> int`
- Type Definitions (shape)
  - `VectraConfig`: `{ embedding, llm, database, chunking?, retrieval?, reranking?, metadata?, query_planning?, grounding?, generation?, prompts?, callbacks? }`
  - `QueryResponse`: `{ 'answer': str | dict, 'sources': list }` or streaming `AsyncGenerator[{ 'delta', 'finish_reason', 'usage' }]`

## 8. Recipes / FAQ
- How do I use a local LLM?
  - Use **Ollama** (`ProviderType.OLLAMA`) for the easiest local setup.
  - Alternatively, use HuggingFace Inference API or a custom provider.
- How do I extract JSON from the answer?
  - Set `generation={'output_format': 'json'}` and parse `answer`; fallback to text on parse errors.
- Why is my retrieval slow?
  - Ensure Prisma indexes are created (`ensure_indexes()`); confirm embedding `dimensions` match pgvector column; consider Hybrid Search and metadata filters.
