# Vectra (Python)

A production-ready, provider-agnostic Python SDK for End-to-End RAG (Retrieval-Augmented Generation) pipelines.

![GitHub Release](https://img.shields.io/github/v/release/iamabhishek-n/vectra-py)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/vectra-rag-py?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/vectra-rag-py)

## Features

*   **Multi-Provider Support**: First-class support for **OpenAI**, **Gemini**, **Anthropic**, **OpenRouter**, and **Hugging Face**.
*   **Async First**: Built on `asyncio` for non-blocking I/O (files & network).
*   **Modular Vector Store**:
    *   **Prisma**: Use your existing PostgreSQL database with `pgvector` via `prisma-client-py`.
    *   **Native PostgreSQL**: Direct connection to PostgreSQL using `asyncpg` (no ORM required).
    *   **ChromaDB**: Native support for the open-source vector database.
    *   **Qdrant & Milvus**: Additional backends for portability.
    *   **Extensible**: Easily add others by subclassing `VectorStore`.
*   **Advanced Chunking**:
    *   **Recursive**: Smart splitting with token-aware sentence/paragraph fallback and adaptive overlap.
    *   **Agentic**: Uses an LLM to split text into semantically complete propositions with JSON validation and dedupe.
*   **Advanced Retrieval Strategies**:
    *   **Naive**: Standard cosine similarity search.
    *   **HyDE (Hypothetical Document Embeddings)**: Generates a fake answer to the query and searches for that.
    *   **Multi-Query**: Generates multiple variations of the query to catch different phrasings (RRF fusion).
    *   **Hybrid Search**: Combines semantic and lexical results using **Reciprocal Rank Fusion (RRF)**.
    *   **MMR**: Diversifies results to reduce redundancy.
*   **Streaming**: Full support for token-by-token streaming responses (Async Generators).
*   **Reranking**: LLM-based reranking to re-order retrieved documents for maximum relevance.
*   **File Support**: Native parsing for PDF, DOCX, XLSX, TXT, and Markdown.
*   **Index Helpers**: ivfflat for pgvector, GIN FTS index.
*   **Embedding Cache**: SHA256 content-based cache to skip re-embedding.
*   **Batch Embeddings**: Gemini and OpenAI adapters support array inputs and dimension control.
*   **Metadata Enrichment**: Per-chunk summary, keywords, hypothetical questions; page and section mapping for PDFs/Markdown. Retrieval boosts matching keywords and uses summaries in prompts.
*   **Conversation Memory**: Built-in chat history management for context-aware multi-turn conversations.
*   **Production Evaluation**: Integrated evaluation module to measure RAG quality (Faithfulness, Relevance).
*   **Local LLMs**: First-class support for **Ollama** for local/offline development.
*   **Web Configuration UI**: Visual generator to create and validate your configuration file (`vectra webconfig`).

---

## Installation

```bash
## Library
pip install vectra-py

## Library (uv)
uv pip install vectra-py

## CLI
# After install, the `vectra` command is available
vectra --help

# Alternative: invoke via module if scripts are disabled
python -m vectra.cli ingest ./docs --config=./config.json
```

**Requirements:**
`pypdf`, `mammoth`, `openpyxl`, `openai`, `google-generativeai`, `anthropic`, `pydantic`, `prisma`, `chromadb`, `asyncpg`

---

## Usage Guide

### 1. Configuration

The SDK uses Pydantic models for strict validation.

```python
from prisma import Prisma
from vectra import VectraClient, VectraConfig, ProviderType, RetrievalStrategy

# Initialize your DB Client
prisma = Prisma()
await prisma.connect()

config = VectraConfig(
    # 1. Embedding Provider
    embedding={
        "provider": ProviderType.OPENAI,
        "api_key": "sk-...",
        "model_name": "text-embedding-3-small"
    },

    # 2. LLM Provider
    llm={
        "provider": ProviderType.GEMINI,
        "api_key": "AIza...",
        "model_name": "gemini-1.5-pro-latest"
    },

    # 3. Database (Modular)
    database={
        "type": "prisma", # or 'chroma'
        "client_instance": prisma,
        "table_name": "Document",
        "column_map": {"content": "content", "vector": "embedding", "metadata": "metadata"}
    },

    # 4. Retrieval (Optional)
    retrieval={
        "strategy": RetrievalStrategy.HYBRID # Uses RRF
    }
)
```

### Configuration Reference

- Embedding
  - `provider`: `ProviderType.OPENAI` | `ProviderType.GEMINI`
  - `api_key`: provider API key string
  - `model_name`: embedding model identifier
  - `dimensions`: number (optional); match DB `pgvector(n)` dimension
- LLM
  - `provider`: `ProviderType.OPENAI` | `ProviderType.GEMINI` | `ProviderType.ANTHROPIC` | `ProviderType.OLLAMA`
  - `api_key`: provider API key string (optional for Ollama)
  - `model_name`: generation model identifier
  - `base_url`: optional custom URL (e.g., for Ollama)
  - `temperature`: number (optional)
  - `max_tokens`: number (optional)
- Memory
  - `enabled`: bool; toggle memory on/off (default: False)
  - `type`: `'in-memory' | 'redis' | 'postgres'`
  - `max_messages`: int; number of recent messages to retain (default: 20)
  - `redis`: `{ client_instance, key_prefix }` where `key_prefix` defaults to `'vectra:chat:'`
  - `postgres`: `{ client_instance, table_name, column_map }` where `table_name` defaults to `'ChatMessage'` and `column_map` maps `{ sessionId, role, content, createdAt }`
- Ingestion
  - `rate_limit_enabled`: bool; toggle rate limiting on/off (default: False)
  - `concurrency_limit`: int; max concurrent embedding requests when enabled (default: 5)
- Database
  - `type`: `prisma` | `chroma` | `qdrant` | `milvus`
  - `client_instance`: instantiated client for the chosen backend
  - `table_name`: table/collection name
  - `column_map`: maps SDK fields to DB columns
    - `content`: text column name
    - `vector`: embedding vector column name (Postgres pgvector)
    - `metadata`: JSON column name
- Chunking
  - `strategy`: `ChunkingStrategy.RECURSIVE` | `ChunkingStrategy.AGENTIC`
  - `chunk_size`: number
  - `chunk_overlap`: number
  - `separators`: list[str] (optional)
- Retrieval
  - `strategy`: `RetrievalStrategy.NAIVE` | `HYDE` | `MULTI_QUERY` | `HYBRID`
  - `llm_config`: optional for query rewriting (HyDE/Multi-Query)
- Reranking
  - `enabled`: bool
  - `top_n`: int (optional)
  - `window_size`: int
  - `llm_config`: optional reranker LLM config
- Metadata
  - `enrichment`: bool; generate `summary`, `keywords`, `hypothetical_questions`
- Callbacks
  - `callbacks`: list of handlers; `LoggingCallbackHandler` or `StructuredLoggingCallbackHandler`
- Observability
  - `enabled`: bool; enable SQLite-based observability (default: False)
  - `sqlite_path`: str; path to SQLite database file (default: 'vectra-observability.db')
  - `project_id`: str; project identifier for multi-project support (default: 'default')
  - `track_metrics`: bool; track latency and other metrics
  - `track_traces`: bool; track detailed workflow traces
  - `session_tracking`: bool; track chat sessions


### 2. Initialization & Ingestion

```python
client = VectraClient(config)

# Ingest a file (supports .pdf, .docx, .txt, .md, .xlsx)
# This will: Load -> Chunk -> Embed -> Store
await client.ingest_documents("./documents/contract.pdf")

# Enable metadata enrichment
# config.metadata = { 'enrichment': True }
```

### Document Management

```python
docs = await client.list_documents(filter={ "docTitle": "Contract" }, limit=50)
deleted = await client.delete_documents({ "absolutePath": "/abs/path/to/contract.pdf" })
updated = await client.update_documents({ "docTitle": "Contract" }, { "metadata": { "status": "archived" } })
```

### 3. Querying (Standard)

```python
result = await client.query_rag("What are the payment terms?")

print("Answer:", result['answer'])
print("Sources:", result['sources']) # Metadata of retrieved chunks
```

### 4. Querying (Streaming)

Ideal for Chat UIs. Returns an Async Generator of unified chunks.

```python
stream = await client.query_rag("Draft a response...", stream=True)

async for chunk in stream:
    print(chunk.get('delta', ''), end="", flush=True)
```

### 5. Conversation Memory

Enable multi-turn conversations by configuring memory and passing a `session_id`.

```python
# In config (enable memory: default is off)
config = VectraConfig(
    # ...
    memory={ "enabled": True, "type": "in-memory", "max_messages": 10 }
)

# Redis-backed memory
redis = ...  # your async Redis client instance
config_redis = VectraConfig(
    # ...
    memory={
        "enabled": True,
        "type": "redis",
        "redis": { "client_instance": redis, "key_prefix": "vectra:chat:" },
        "max_messages": 20
    }
)

# Postgres-backed memory
prisma = ...  # your Prisma client instance
config_postgres = VectraConfig(
    # ...
    memory={
        "enabled": True,
        "type": "postgres",
        "postgres": {
            "client_instance": prisma,
            "table_name": "ChatMessage",
            "column_map": { "sessionId": "sessionId", "role": "role", "content": "content", "createdAt": "createdAt" }
        },
        "max_messages": 20
    }
)

# In your app:
session_id = "user-123-session-abc"
result = await client.query_rag("What is the refund policy?", session_id=session_id)
follow_up = await client.query_rag("Does it apply to sale items?", session_id=session_id)
```

### 6. Production Evaluation

Measure the quality of your RAG pipeline using the built-in evaluation module.

```python
test_set = [
    { 
        "question": "What is the capital of France?", 
        "expected_ground_truth": "Paris is the capital of France." 
    }
]

results = await client.evaluate(test_set)

print(f"Faithfulness: {results['average_faithfulness']}")
print(f"Relevance: {results['average_relevance']}")
```

---

## Supported Providers & Backends

| Feature | OpenAI | Gemini | Anthropic | Ollama | OpenRouter | HuggingFace |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Embeddings** | ✅ | ✅ | ❌ | ✅ | ❌ | ✅ |
| **Generation** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Streaming** | ✅ | ✅ | ✅ | ✅ | ✅ | ⚠️ |

### Ollama (Local)
- Use Ollama for local, offline development.
- Set `provider = ProviderType.OLLAMA`.
- Default `base_url` is `http://localhost:11434`.
```python
config = VectraConfig(
    embedding={ 'provider': ProviderType.OLLAMA, 'model_name': 'nomic-embed-text' },
    llm={ 'provider': ProviderType.OLLAMA, 'model_name': 'llama3' }
)
```

### OpenRouter (Generation)
- Use OpenRouter as a unified generation provider.
- Set `llm.provider = ProviderType.OPENROUTER`, `llm.model_name` to a supported model (e.g., `openai/gpt-4o`).
- Provide `OPENROUTER_API_KEY`; optional attribution via `OPENROUTER_REFERER`, `OPENROUTER_TITLE`.
```python
config = VectraConfig(
  llm={
    'provider': ProviderType.OPENROUTER,
    'api_key': os.getenv('OPENROUTER_API_KEY'),
    'model_name': 'openai/gpt-4o',
    'default_headers': {
      'HTTP-Referer': 'https://your.app',
      'X-Title': 'Your App'
    }
  }
)
```

### Database Schemas

**Prisma (PostgreSQL)**
```prisma
model Document {
  id        String                 @id @default(uuid())
  content   String
  metadata  Json
  embedding Unsupported("vector")? // pgvector type
  createdAt DateTime               @default(now())
}
```

---

## API Reference

### `VectraClient(config: VectraConfig)`
Creates a new client instance.

### `async client.ingest_documents(path: str, ingestion_mode: str = "append")`
Reads a file **or recursively iterates a directory**, chunks content, embeds, and saves to the configured DB.
- If `path` is a file: Ingests that single file.
- If `path` is a directory: Recursively finds all supported files and ingests them.

### `async client.query_rag(query: str, filter: dict = None, stream: bool = False, session_id: str = None)`
Performs the RAG pipeline.

**Returns**:
*   If `stream=False` (default): `Dict { 'answer': str | dict, 'sources': list }`
*   If `stream=True`: `AsyncGenerator[Dict { 'delta': str, 'finish_reason': str | None, 'usage': Any | None }, None]`

### Advanced Configuration

- Query Planning
  - `query_planning.token_budget`: int; total token budget for context
  - `query_planning.prefer_summaries_below`: int; prefer metadata summaries under this budget
  - `query_planning.include_citations`: bool; include titles/sections/pages in context
- Grounding
  - `grounding.enabled`: bool; enable extractive snippet grounding
  - `grounding.strict`: bool; use only grounded snippets when true
  - `grounding.max_snippets`: int; max snippets to include
- Generation
  - `generation.structured_output`: `'none' | 'citations'`; enable inline citations
  - `generation.output_format`: `'text' | 'json'`; return JSON when set to `json`
- Prompts
  - `prompts.query`: string template using `{{context}}` and `{{question}}`
  - `prompts.reranking`: optional template for reranker prompt
- Tracing
  - `tracing.enable`: bool; enable provider/DB/pipeline span hooks

### CLI

Quickly ingest and query to validate configurations.

```bash
vectra ingest ./docs --config=./config.json
vectra query "What are the payment terms?" --config=./config.json --stream
```

### Ingestion Rate Limiting
- Toggle ingestion rate limiting via `config.ingestion`.
```python
config = VectraConfig(
    # ...
    ingestion={ "rate_limit_enabled": True, "concurrency_limit": 5 }
)
```

---

## Extending

### Custom Vector Store
Inherit from `VectorStore` class (`vectra.interfaces`) and implement abstract methods.

```python
from vectra.interfaces import VectorStore

class MyCustomDB(VectorStore):
    async def add_documents(self, documents):
        ...
    async def similarity_search(self, vector, limit=5, filter=None):
        ...
```

---

## Developer Guide

### Setup
- Python 3.8+ recommended.
- Install editable: `pip install -e .`.
- CLI is exposed as `vectra` via `pyproject.toml`.

### Environment
- `OPENAI_API_KEY`, `GOOGLE_API_KEY`, `ANTHROPIC_API_KEY` for providers.
- Database client instance configured under `config.database.client_instance`.

### Architecture
- Pipeline: Load → Chunk → Embed → Store → Retrieve → Rerank → Plan → Ground → Generate → Stream.
- Core client: `VectraClient` (library export).
- Configuration: `VectraConfig` (Pydantic model).
- Vector store interface: `VectorStore` (extend to add custom stores).
- Callbacks: pass a handler object implementing the documented event methods.

### Retrieval Strategies
- Supports NAIVE, HYDE, MULTI_QUERY, HYBRID, MMR.

### Query Planning & Grounding
- Context assembly respects `query_planning` (token budget, summary preference, citations).
- Snippet extraction controlled by `grounding` (strict mode, max snippets).

### Streaming Interface
- Unified streaming shape `{ 'delta', 'finish_reason', 'usage' }` across OpenAI, Gemini, Anthropic.

### Adding a Provider
- Implement `embed_documents`, `embed_query`, `generate`, `generate_stream`.
- Ensure streaming yields `{ 'delta', 'finish_reason', 'usage' }`.
- Wire via `llm.provider` in config.

### Adding a Vector Store
- Extend `VectorStore`; implement `add_documents`, `similarity_search`, `list_documents`, `delete_documents`, `update_documents`, optionally `hybrid_search`.
- Select via `database.type` in config.

### Callbacks & Observability
- Available events: `on_ingest_start`, `on_ingest_end`, `on_ingest_summary`, `on_chunking_start`, `on_embedding_start`, `on_retrieval_start`, `on_retrieval_end`, `on_reranking_start`, `on_reranking_end`, `on_generation_start`, `on_generation_end`, `on_error`.
- Implement a simple callback object with these methods; pass via `config.callbacks`.

### CLI
- Binary `vectra` is installed with the package.
- Ingest: `vectra ingest ./docs --config=./config.json`.
- Query: `vectra query "<text>" --config=./config.json --stream`.

### Coding Conventions
- Async-first I/O; use `asyncio`.
- Pydantic for configuration validation.

---

## Feature Guide

### Embeddings
- Providers: `OPENAI`, `GEMINI`.
- Configure `dimensions` to match DB `pgvector(n)` when applicable.
```python
config = VectraConfig(
  embedding={
    'provider': ProviderType.OPENAI,
    'api_key': os.getenv('OPENAI_API_KEY'),
    'model_name': 'text-embedding-3-small',
    'dimensions': 1536,
  }
)
```

### Generation
- Providers: `OPENAI`, `GEMINI`, `ANTHROPIC`.
- Options: `temperature`, `max_tokens`.
- Structured output: set `generation={'output_format': 'json', 'structured_output': 'citations'}`.
```python
config = VectraConfig(
  llm={ 'provider': ProviderType.GEMINI, 'api_key': os.getenv('GOOGLE_API_KEY'), 'model_name': 'gemini-1.5-pro-latest', 'temperature': 0.3 },
  generation={ 'output_format': 'json', 'structured_output': 'citations' }
)
client = VectraClient(config)
res = await client.query_rag('Summarize our policy with citations.')
print(res['answer'])
```

- OpenRouter usage:
```python
config = VectraConfig(
  llm={
    'provider': ProviderType.OPENROUTER,
    'api_key': os.getenv('OPENROUTER_API_KEY'),
    'model_name': 'openai/gpt-4o',
    'default_headers': { 'HTTP-Referer': 'https://your.app', 'X-Title': 'Your App' }
  }
)
```

### Chunking
- Strategies: `RECURSIVE`, `AGENTIC`.
- Agentic requires `chunking.agentic_llm`.
```python
config = VectraConfig(
  chunking={
    'strategy': ChunkingStrategy.AGENTIC,
    'agentic_llm': {
      'provider': ProviderType.OPENAI,
      'api_key': os.getenv('OPENAI_API_KEY'),
      'model_name': 'gpt-4o-mini'
    },
    'chunk_size': 1200,
    'chunk_overlap': 200
  }
)
```

### Retrieval
- Strategies: `NAIVE`, `HYDE`, `MULTI_QUERY`, `HYBRID`.
- HYDE/MULTI_QUERY require `retrieval.llm_config`.
```python
config = VectraConfig(
  retrieval={
    'strategy': RetrievalStrategy.MULTI_QUERY,
    'llm_config': { 'provider': ProviderType.OPENAI, 'api_key': os.getenv('OPENAI_API_KEY'), 'model_name': 'gpt-4o-mini' },
    'hybrid_alpha': 0.5
  }
)
```

### Reranking
- Enable LLM-based reranking to reorder results.
```python
config = VectraConfig(
  reranking={
    'enabled': True,
    'top_n': 5,
    'window_size': 20,
    'llm_config': { 'provider': ProviderType.ANTHROPIC, 'api_key': os.getenv('ANTHROPIC_API_KEY'), 'model_name': 'claude-3-haiku' }
  }
)
```

### Metadata Enrichment
- Add summaries, keywords, hypothetical questions during ingestion.
```python
config = VectraConfig(metadata={ 'enrichment': True })
await client.ingest_documents('./docs/contract.pdf')
```

### Query Planning
- Control context assembly with token budget and summary preference.
```python
config = VectraConfig(query_planning={ 'token_budget': 2048, 'prefer_summaries_below': 1024, 'include_citations': True })
```

### Answer Grounding
- Inject extractive snippets; use `strict` to only allow grounded quotes.
```python
config = VectraConfig(grounding={ 'enabled': True, 'strict': False, 'max_snippets': 4 })
```

### Prompts
- Provide a custom query template using `{{context}}` and `{{question}}`.
```python
config = VectraConfig(prompts={ 'query': 'Use only the following context to answer.\nContext:\n{{context}}\n\nQ: {{question}}' })
```

### Streaming
- Unified async generator with chunks `{ 'delta', 'finish_reason', 'usage' }`.
```python
stream = await client.query_rag('Draft a welcome email', stream=True)
async for chunk in stream:
    print(chunk.get('delta', ''), end='')
```

### Filters
- Limit retrieval to metadata fields.
```python
res = await client.query_rag('Vacation policy', filter={ 'docTitle': 'Employee Handbook' })
```

### Callbacks
- Hook into pipeline stages for logging/metrics via `config.callbacks`.
```python
class MyLogger:
    def on_retrieval_start(self, query):
        print('Retrieval start:', query)
config = VectraConfig(callbacks=[MyLogger()])
```

### Observability

Built-in SQLite-based observability to track metrics, traces, and sessions.

```python
config = VectraConfig(
    observability={
        'enabled': True,
        'sqlite_path': 'vectra-observability.db',
        'project_id': 'my-project',
        'track_metrics': True,
        'track_traces': True,
        'session_tracking': True
    }
)
```

This tracks:
- **Metrics**: Latency (ingest, query).
- **Traces**: Detailed spans for retrieval, generation, and ingestion workflows.
- **Sessions**: Chat session history and last query tracking.

### Vector Stores
- Prisma (Postgres + pgvector), Chroma, Qdrant, Milvus.
- Configure `database.type`, `table_name`, `column_map`, `client_instance`.
```python
config = VectraConfig(
  database={
    'type': 'prisma',
    'client_instance': prisma,
    'table_name': 'Document',
    'column_map': { 'content': 'content', 'vector': 'embedding', 'metadata': 'metadata' }
  }
)
```
### HuggingFace (Embeddings & Generation)
- Use HuggingFace Inference API for embeddings and generation.
- Set `provider = ProviderType.HUGGINGFACE`, `model_name` to a supported model (e.g., `sentence-transformers/all-MiniLM-L6-v2` for embeddings, `tiiuae/falcon-7b-instruct` for generation).
- Provide `HUGGINGFACE_API_KEY`.
```python
config = VectraConfig(
  embedding={ 'provider': ProviderType.HUGGINGFACE, 'api_key': os.getenv('HUGGINGFACE_API_KEY'), 'model_name': 'sentence-transformers/all-MiniLM-L6-v2' },
  llm={ 'provider': ProviderType.HUGGINGFACE, 'api_key': os.getenv('HUGGINGFACE_API_KEY'), 'model_name': 'tiiuae/falcon-7b-instruct' }
)
```
