# Vectra (Python)

Vectra is a production-grade, provider-agnostic Python SDK for building retrieval-augmented generation systems. It's async-first from the ground up and handles the full pipeline from loading documents to streaming an answer back, built so you can swap out any piece (embedding provider, vector store, LLM, retrieval strategy) without rewriting application code.

![PyPI - Downloads](https://img.shields.io/pypi/dm/vectra-rag-py)
![GitHub Release](https://img.shields.io/github/v/release/iamabhishek-n/vectra-py)
[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=iamabhishek-n_vectra-py&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=iamabhishek-n_vectra-py)

If you find this project useful, consider supporting it:<br>
[![Star this project on GitHub](https://img.shields.io/github/stars/iamabhishek-n/vectra-py?style=social)](https://github.com/iamabhishek-n/vectra-py/stargazers)
[![Sponsor me on GitHub](https://img.shields.io/badge/Sponsor%20me%20on-GitHub-%23FFD43B?logo=github)](https://github.com/sponsors/iamabhishek-n)
[![Buy me a Coffee](https://img.shields.io/badge/Buy%20me%20a%20Coffee-%23FFDD00?logo=buy-me-a-coffee&logoColor=black)](https://www.buymeacoffee.com/iamabhishekn)

## Table of Contents

* [1. Overview](#1-overview)
* [2. Design Goals](#2-design-goals)
* [3. Feature Matrix](#3-feature-matrix)
* [4. Installation](#4-installation)
* [5. Quick Start](#5-quick-start)
* [6. Core Concepts](#6-core-concepts)
* [7. Configuration Reference](#7-configuration-reference)
* [8. Context and Memory Layer](#8-context-and-memory-layer)
* [9. Ingestion Pipeline](#9-ingestion-pipeline)
* [10. Querying and Streaming](#10-querying-and-streaming)
* [11. Conversation Memory](#11-conversation-memory)
* [12. Evaluation](#12-evaluation)
* [13. CLI](#13-cli)
* [14. Observability and Callbacks](#14-observability-and-callbacks)
* [15. Telemetry](#15-telemetry)
* [16. Guardrails](#16-guardrails)
* [17. Database Schema](#17-database-schema)
* [18. Extending Vectra](#18-extending-vectra)
* [19. Architecture](#19-architecture)
* [20. Development](#20-development)
* [21. Production Notes](#21-production-notes)

---

## 1. Overview

The pipeline looks like this:

```
Load -> Chunk -> Embed -> Store -> Retrieve -> Rerank -> Plan -> Ground -> Generate -> Stream
```

<p align="center">
  <img src="https://vectra.thenxtgenagents.com/vectraArch.png" alt="Vectra SDK Architecture" width="900">
</p>

<p align="center">
  <em>Vectra SDK, end to end RAG architecture</em>
</p>

Every stage is explicit and every stage is async. There's no hidden default embedding model, no silent fallback vector store. If something isn't configured, Vectra tells you rather than guessing.

### What's in the box

* A provider-agnostic embedding and generation layer (OpenAI, Gemini, Anthropic, Ollama, OpenRouter, HuggingFace)
* Seven vector store backends, swappable via one config key
* Retrieval strategies beyond naive cosine similarity: HyDE, multi-query expansion, hybrid RRF, MMR
* A context and memory layer for assembling budget-aware prompts and carrying facts across sessions
* A CLI with the same capabilities as the SDK, plus a local web UI for config and observability

---

## 2. Design Goals

**Explicit over implicit.** Chunking, retrieval, grounding and memory behavior are all things you configure on purpose. Vectra won't quietly pick a strategy for you.

**Production-first.** Rate limiting, embedding caching, index helpers, observability and evaluation aren't add-ons bolted on later. They're part of the core design.

**No vendor lock-in.** Moving from OpenAI to Gemini, or from Postgres to Qdrant, is a config change. Your ingestion and query code doesn't move.

**Interfaces you can extend.** Providers, vector stores and middleware are all built against small abstract base classes. Writing your own backend is a matter of implementing a handful of methods, not fighting the framework.

---

## 3. Feature Matrix

**Providers**

* Embeddings: OpenAI, Gemini, Ollama, HuggingFace
* Generation: OpenAI, Gemini, Anthropic, Ollama, OpenRouter, HuggingFace
* Streaming: async generators, normalized output across providers

**Vector stores**

* PostgreSQL via Prisma and pgvector
* PostgreSQL via native `asyncpg`
* ChromaDB
* Qdrant
* Milvus
* Pinecone
* Weaviate (native hybrid search and filter-based listing, not the client-side fallback the others use)

**Retrieval strategies**

* Naive cosine similarity
* HyDE (hypothetical document embeddings)
* Multi-query expansion
* Hybrid semantic and lexical search, fused with reciprocal rank fusion
* MMR diversification

---

## 4. Installation

```bash
pip install vectra-rag-py
# or
uv pip install vectra-rag-py
```

Install the client for whichever backend you're using. Vectra doesn't bundle these, since most projects only need one or two.

```bash
pip install asyncpg           # native Postgres
pip install prisma-client-py  # Prisma + pgvector, https://prisma.brendonovich.dev
pip install chromadb          # ChromaDB, https://docs.trychroma.com
pip install qdrant-client     # Qdrant, https://qdrant.tech/documentation
pip install pymilvus          # Milvus, https://milvus.io/docs
pip install pinecone          # Pinecone, https://docs.pinecone.io/
pip install weaviate-client   # Weaviate, https://weaviate.io/developers/weaviate
```

CLI:

```bash
vectra --help
# or, if the entry point isn't on your PATH
python -m vectra.cli --help
```

Core dependencies: `pydantic`, `openai`, `google-generativeai`, `anthropic`, `pypdf`, `mammoth`, `openpyxl`.

---

## 5. Quick Start

```python
import os
import asyncpg
from vectra import VectraClient, VectraConfig, ProviderType

pool = await asyncpg.create_pool(os.getenv('DATABASE_URL'))

config = VectraConfig(
    embedding={
        'provider': ProviderType.OPENAI,
        'api_key': os.getenv('OPENAI_API_KEY'),
        'model_name': 'text-embedding-3-small'
    },
    llm={
        'provider': ProviderType.GEMINI,
        'api_key': os.getenv('GOOGLE_API_KEY'),
        'model_name': 'gemini-2.5-flash'
    },
    database={
        'type': 'postgres',
        'client_instance': pool,
        'table_name': 'document',
        'column_map': {'content': 'content', 'metadata': 'metadata', 'vector': 'vector'}
    }
)

client = VectraClient(config)
await client.ingest_documents('./docs')
result = await client.query_rag('What is the vacation policy?')
print(result['answer'])
```

That's the whole setup for a working RAG pipeline. Everything past this point is about tuning it.

---

## 6. Core Concepts

**Providers** implement embeddings, generation, or both. Vectra normalizes the response shape and the streaming interface so switching providers doesn't touch your call sites.

**Vector stores** persist embeddings and metadata. They're swappable through config, and every backend implements the same interface (add, search, hybrid search, list, delete, update, file-exists check).

**Chunking** has two strategies: recursive, token-aware splitting for most content, and agentic LLM-driven splitting for documents where semantic boundaries matter more than character counts (contracts, policies, anything dense).

**Retrieval** is where you trade recall for precision. Hybrid is the sane default for production workloads; the others exist for cases where you know your query distribution well enough to hand-tune.

**Reranking** is an optional second pass that reorders candidate chunks with an LLM before they're used.

**Metadata enrichment** generates summaries, keywords and hypothetical questions per chunk during ingestion, which improves retrieval quality at the cost of a slower ingest.

**Query planning and grounding** control how retrieved context gets assembled into a prompt and how strictly the model has to stick to what it was given.

**Conversation memory** persists chat history across turns. Section 8 covers a second, complementary kind of memory: durable facts extracted from conversations, not just the raw transcript.

---

## 7. Configuration Reference

All configuration is validated with Pydantic at runtime, so a typo in a config key fails loudly at startup instead of silently doing nothing.

### Embedding

```python
embedding = {
    'provider': ProviderType.OPENAI,
    'api_key': os.getenv('OPENAI_API_KEY'),
    'model_name': 'text-embedding-3-small',
    'dimensions': 1536
}
```

Set `dimensions` explicitly when using pgvector. The column is created with a fixed dimension, and a mismatch fails at query time rather than at startup.

### LLM

```python
llm = {
    'provider': ProviderType.GEMINI,
    'api_key': os.getenv('GOOGLE_API_KEY'),
    'model_name': 'gemini-2.5-flash',
    'temperature': 0.3,
    'max_tokens': 1024
}
```

This model is used for answer generation, HyDE, multi-query expansion, agentic chunking and reranking, unless you override any of those with their own `llm_config`.

### Database

```python
# Native Postgres (asyncpg)
database = {
    'type': 'postgres',
    'client_instance': pg_pool,
    'table_name': 'document',
    'column_map': {'content': 'content', 'metadata': 'metadata', 'vector': 'vector'}
}
```

```python
# Prisma
database = {
    'type': 'prisma',
    'client_instance': prisma,
    'table_name': 'Document',
    'column_map': {'content': 'content', 'metadata': 'metadata', 'vector': 'embedding'}
}
```

```python
# ChromaDB
database = {
    'type': 'chroma',
    'client_instance': chroma_client,
    'table_name': 'rag_collection'
}
```

```python
# Qdrant
database = {
    'type': 'qdrant',
    'client_instance': qdrant_client,
    'table_name': 'rag_collection'
}
```

```python
# Milvus
database = {
    'type': 'milvus',
    'client_instance': milvus_client,
    'table_name': 'rag_collection',
    'metric_type': 'COSINE'  # or 'IP', 'L2', matching how the collection was created
}
```

```python
# Pinecone
database = {
    'type': 'pinecone',
    'client_instance': pinecone_index,  # an Index handle from the Pinecone client
    'table_name': 'my-namespace'        # optional, maps to a Pinecone namespace
}
```

Pinecone has no listing or scroll endpoint. On this backend, `list_documents` and `update_documents` raise `NotImplementedError` rather than pretending to support them, and `delete_documents` returns `-1` since Pinecone can't report a count for a filtered delete.

```python
# Weaviate
database = {
    'type': 'weaviate',
    'client_instance': weaviate_client,  # a v3 collections-API client
    'table_name': 'Document'             # the collection name
}
```

Weaviate supports hybrid search and filtered listing natively, so `list_documents`, `update_documents` and `delete_documents` all work with real results here, unlike on Pinecone.

### Chunking

```python
chunking = {
    'strategy': ChunkingStrategy.RECURSIVE,
    'chunk_size': 1000,
    'chunk_overlap': 200
}
```

```python
# Agentic
chunking = {
    'strategy': ChunkingStrategy.AGENTIC,
    'agentic_llm': {
        'provider': ProviderType.OPENAI,
        'api_key': os.getenv('OPENAI_API_KEY'),
        'model_name': 'gpt-4o-mini'
    }
}
```

### Retrieval

```python
retrieval = {'strategy': RetrievalStrategy.HYBRID}
```

### Reranking

```python
reranking = {
    'enabled': True,
    'window_size': 20,
    'top_n': 5
}
```

### Conversation memory

```python
memory = {'enabled': True, 'type': 'in-memory', 'max_messages': 20}
```

```python
# Redis
memory = {
    'enabled': True,
    'type': 'redis',
    'max_messages': 20,
    'redis': {
        'client_instance': redis_client,
        'key_prefix': 'vectra:chat:'
    }
}
```

```python
# Postgres
memory = {
    'enabled': True,
    'type': 'postgres',
    'max_messages': 20,
    'postgres': {
        'client_instance': pg_pool,
        'table_name': 'ChatMessage',
        'column_map': {
            'sessionId': 'sessionId',
            'role': 'role',
            'content': 'content',
            'createdAt': 'createdAt'
        }
    }
}
```

### Observability

```python
observability = {
    'enabled': True,
    'sqlite_path': 'vectra-observability.db'
}
```

---

## 8. Context and Memory Layer

Conversation memory (section 11) stores the raw back-and-forth. The context layer is a different thing: it's the primitive that assembles whatever a model needs to see, from whatever sources you have, packed into a token budget, with nothing dropped silently.

The simplest entry point is `client.context.ask`, which runs guardrails and middleware the same way `query_rag` does, retrieves from your configured vector store, and packs the result:

```python
packed = await client.context.ask('what did we agree on for pricing?', session_id='user-42')

print(packed['text'])            # the assembled context, ready to hand to an LLM
print(packed['tokens_used'], packed['tokens_budget'])
if packed['warnings']:
    print(packed['warnings'])
```

`packed['dropped']` and `packed['warnings']` are never silent. If a source ran out of budget or a store timed out, it shows up there instead of just vanishing.

The default budget is 2048 tokens. Override it, and the order sources get packed in, through `context_layer` on your `VectraConfig`:

```python
config = VectraConfig(
    # ...
    context_layer={
        'budget': {'max_tokens': 4000},
        'priority': ['memory', 'docs', 'tools']  # packed in this order until the budget runs out
    }
)
```

### Durable facts

Alongside raw conversation history, Vectra can maintain a separate store of facts extracted from conversations, each with a validity window rather than a hard delete. When a new fact contradicts an old one, the old one is marked invalid at that point in time instead of being erased, so you can still answer "what did we believe last month."

Turn this on by adding a `facts` block under `memory`, pointing at a Postgres-compatible client (the fact store uses pgvector under the hood):

```python
memory = {
    'enabled': True,
    'facts': {
        'enabled': True,
        'client_instance': facts_pool,
        'table_name': 'VectraFact'
    }
}
```

Once enabled, `client.fact_store` is available directly on the client:

```python
await client.fact_store.ensure_indexes()  # run once, sets up the table and indexes

await client.fact_store.write('user-42', {
    'user_message': 'Our deploy target is Tokyo from now on.',
    'assistant_message': 'Got it, defaulting to the Tokyo region.'
})

# context.ask automatically pulls relevant facts into the packed context
# once a fact store is configured and a session_id is passed in.
packed = await client.context.ask('where should this deploy?', session_id='user-42')
```

Writing facts isn't automatic. `query_rag` doesn't call `fact_store.write` for you, so if you want facts to persist you call it yourself after a turn completes, with whatever extraction trigger makes sense for your app.

---

## 9. Ingestion Pipeline

```python
await client.ingest_documents('./documents')
```

Works on a single file or a directory, walked recursively, with an embedding cache keyed by content hash and optional rate limiting on the embedding calls. Supported formats: PDF, DOCX, XLSX, TXT, Markdown.

---

## 10. Querying and Streaming

```python
result = await client.query_rag('Refund policy?')
```

```python
stream = await client.query_rag('Draft an email', stream=True)
async for chunk in stream:
    print(chunk.get('delta', ''), end='')
```

---

## 11. Conversation Memory

Pass a `session_id` to `query_rag` to carry history across turns. This is the raw transcript, separate from the fact store described in section 8.

---

## 12. Evaluation

```python
await client.evaluate([
    {'question': 'Capital of France?', 'expected_ground_truth': 'Paris'}
])
```

Reports faithfulness and relevance scores against your ground truth set.

---

## 13. CLI

```bash
vectra ingest ./docs --config=./config.json
vectra query "What are the payment terms?" --config=./config.json --stream
```

**WebConfig** is a local UI for building and validating a `vectra.config.json` without hand-writing it.

```bash
vectra webconfig
```

**Dashboard** is a local, SQLite-backed UI showing ingestion latency, query latency, retrieval and generation traces, and chat sessions.

```bash
vectra dashboard
```

---

## 14. Observability and Callbacks

Enabling `observability` records metrics, traces and sessions automatically. Callbacks give you hooks into ingestion, retrieval, reranking and generation, if you want to wire your own logging or metrics on top.

---

## 15. Telemetry

Vectra collects anonymous usage data to help prioritize features and catch broken releases. It's off by default.

What's tracked: a random UUID stored locally in `~/.vectra/telemetry.json` (no PII, no emails, no IPs), plus coarse event data like which providers and vector stores get configured, ingestion batch sizes and durations, which retrieval strategy gets used, and error types by stage (no stack traces, no query content).

Turn it on explicitly:

```python
client = VectraClient(
    VectraConfig(
        # ...
        telemetry={'enabled': True}
    )
)
```

`VECTRA_TELEMETRY_DISABLED=1` or `DO_NOT_TRACK=1` in the environment overrides the config either way, a reliable way to guarantee nothing gets sent regardless of what a config file says.

---

## 16. Guardrails

`query_rag` enforces a few defaults before anything gets embedded or sent to an LLM:

* `max_query_length` (default 2000 characters): longer queries are rejected outright.
* `block_pii` (default off): rejects queries that look like they contain an email, phone number, SSN-shaped number or a long digit run.
* `content_filter` (default off): rejects queries matching a small built-in list of harmful phrases, extendable with `blocked_terms`.

Ingestion enforces `ingestion.max_file_size_bytes` (default 50MB) before reading a file.

If you're upgrading from an older version: the 2000-character query limit and the 50MB file limit are enforced now even if you never set a `guardrails` or `ingestion` block. They existed in the schema before this release but weren't actually checked. To raise them:

```python
config = VectraConfig(
    # ... other config ...
    guardrails={'max_query_length': 10000},
    ingestion={'max_file_size_bytes': 200 * 1024 * 1024},
)

client = VectraClient(config)
```

The exact detection logic lives in `vectra/guardrails.py` if you need to know precisely what triggers a block.

---

## 17. Database Schema

For Prisma users, something like this:

```prisma
model Document {
  id        String   @id @default(uuid())
  content   String
  metadata  Json
  embedding Unsupported("vector")?
  createdAt DateTime @default(now())
}
```

---

## 18. Extending Vectra

Every vector store implements the same abstract base class. To add your own:

```python
class MyStore(VectorStore):
    async def add_documents(self, documents): ...
    async def similarity_search(self, vector, limit=5, filter=None): ...
    async def hybrid_search(self, text, vector, limit=5, filter=None): ...
    async def list_documents(self, filter=None, limit=100, cursor=None): ...
    async def delete_documents(self, filter): ...
    async def update_documents(self, filter, update_data): ...
    async def file_exists(self, sha256, size, last_modified): ...
```

If your store has no native hybrid search, follow the pattern in `vectra/backends/qdrant_store.py`: pull a wider candidate pool with `similarity_search`, score it against the query lexically, and fuse the two rankings with reciprocal rank fusion.

---

## 19. Architecture

`VectraClient` is the orchestrator. Config is parsed and validated once at construction. Providers and vector stores are chosen behind abstract base classes, so nothing downstream needs to know which one is active. Streaming uses one async generator shape regardless of provider.

---

## 20. Development

* Python 3.8 or newer
* Async-first throughout (`asyncio`)
* Pydantic-based configuration, validated at construction time

---

## 21. Production Notes

Match your embedding `dimensions` to whatever your vector column was created with, especially on pgvector where a mismatch is a runtime error, not a warning. Prefer hybrid retrieval unless you have a specific reason not to. Turn on observability in staging before you need it in an incident. Re-run evaluation before changing chunk size or embedding model, since both quietly shift retrieval quality in ways that are easy to miss without a baseline.
