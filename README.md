# Vectra (Python)

**Vectra** is a **production-grade, provider-agnostic Python SDK** for building **end-to-end Retrieval-Augmented Generation (RAG)** systems. It is designed for teams that need **correctness, extensibility, async performance, and observability** across embeddings, vector databases, retrieval strategies, and LLM providers.

![PyPI - Downloads](https://img.shields.io/pypi/dm/vectra-rag-py)
![GitHub Release](https://img.shields.io/github/v/release/iamabhishek-n/vectra-py)
[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=iamabhishek-n_vectra-py&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=iamabhishek-n_vectra-py)

If you find this project useful, consider supporting it:<br>
[![Star this project on GitHub](https://img.shields.io/github/stars/iamabhishek-n/vectra-py?style=social)](https://github.com/iamabhishek-n/vectra-py/stargazers)
[![Sponsor me on GitHub](https://img.shields.io/badge/Sponsor%20me%20on-GitHub-%23FFD43B?logo=github)](https://github.com/sponsors/iamabhishek-n)
[![Buy me a Coffee](https://img.shields.io/badge/Buy%20me%20a%20Coffee-%23FFDD00?logo=buy-me-a-coffee&logoColor=black)](https://www.buymeacoffee.com/iamabhishekn)


## Table of Contents

* [1. Overview](#1-overview)
* [2. Design Goals & Philosophy](#2-design-goals--philosophy)
* [3. Feature Matrix](#3-feature-matrix)
* [4. Installation](#4-installation)
* [5. Quick Start](#5-quick-start)
* [6. Core Concepts](#6-core-concepts)

  * [Providers](#providers)
  * [Vector Stores](#vector-stores)
  * [Chunking](#chunking)
  * [Retrieval](#retrieval)
  * [Reranking](#reranking)
  * [Metadata Enrichment](#metadata-enrichment)
  * [Query Planning & Grounding](#query-planning--grounding)
  * [Conversation Memory](#conversation-memory)
* [7. Configuration Reference (Usage-Driven)](#7-configuration-reference-usage-driven)
* [8. Ingestion Pipeline](#8-ingestion-pipeline)
* [9. Querying & Streaming](#9-querying--streaming)
* [10. Conversation Memory](#10-conversation-memory)
* [11. Evaluation & Quality Measurement](#11-evaluation--quality-measurement)
* [12. CLI](#12-cli)

  * [Ingest & Query](#ingest--query)
  * [WebConfig (Config Generator UI)](#webconfig-config-generator-ui)
  * [Observability Dashboard](#observability-dashboard)
* [13. Observability & Callbacks](#13-observability--callbacks)
* [14. Database Schemas & Indexing](#14-database-schemas--indexing)
* [15. Extending Vectra](#15-extending-vectra)
* [16. Architecture Overview](#16-architecture-overview)
* [17. Development & Contribution Guide](#17-development--contribution-guide)
* [18. Production Best Practices](#18-production-best-practices)

---

## 1. Overview

Vectra implements a **fully modular RAG pipeline**:

```
Load → Chunk → Embed → Store → Retrieve → Rerank → Plan → Ground → Generate → Stream
```
<p align="center">
  <img src="https://vectra.thenxtgenagents.com/vectraArch.png" alt="Vectra SDK Architecture" width="900">
</p>

<p align="center">
  <em>Vectra SDK – End-to-End RAG Architecture</em>
</p>

All stages are **explicitly configured**, **async-first**, and **observable**.

### Key Characteristics

* Async-first API (`asyncio`)
* Provider-agnostic embeddings & LLMs
* Multiple vector backends (Postgres, Chroma, Qdrant, Milvus)
* Advanced retrieval (HyDE, Multi-Query, Hybrid RRF, MMR)
* Unified streaming interface
* Built-in evaluation and observability
* CLI + SDK parity

---

## 2. Design Goals & Philosophy

### Explicitness over Magic

Vectra avoids hidden defaults. Chunking, retrieval, grounding, memory, and generation behavior are always explicit and validated.

### Production-First

Index helpers, rate limiting, embedding cache, observability, and evaluation are first-class features.

### Provider Neutrality

Switching providers (OpenAI ↔ Gemini ↔ Anthropic ↔ Ollama) requires **no application code changes**.

### Extensibility

All major subsystems are interface-driven and designed to be extended safely.

---

## 3. Feature Matrix

### Providers

* **Embeddings**: OpenAI, Gemini, Ollama, HuggingFace
* **Generation**: OpenAI, Gemini, Anthropic, Ollama, OpenRouter, HuggingFace
* **Streaming**: Async generators with normalized output

### Vector Stores

* PostgreSQL (Prisma + pgvector)
* ChromaDB
* Qdrant
* Milvus

### Retrieval Strategies

* Naive cosine similarity
* HyDE (Hypothetical Document Embeddings)
* Multi-Query expansion (RRF)
* Hybrid semantic + lexical (RRF)
* MMR diversification

---

## 4. Installation

### Library

```bash
pip install vectra-py
# or
uv pip install vectra-py
```

### CLI

```bash
vectra --help
# alternative
python -m vectra.cli --help
```

### Requirements

Vectra depends on:
`pydantic`, `asyncio`, `prisma-client-py`, `chromadb`, `openai`, `google-generativeai`, `anthropic`, `pypdf`, `mammoth`, `openpyxl`

---

## 5. Quick Start

```python
from prisma import Prisma
from vectra import VectraClient, VectraConfig, ProviderType

prisma = Prisma()
await prisma.connect()

config = VectraConfig(
    embedding={
        'provider': ProviderType.OPENAI,
        'api_key': os.getenv('OPENAI_API_KEY'),
        'model_name': 'text-embedding-3-small'
    },
    llm={
        'provider': ProviderType.GEMINI,
        'api_key': os.getenv('GOOGLE_API_KEY'),
        'model_name': 'gemini-1.5-pro-latest'
    },
    database={
        'type': 'prisma',
        'client_instance': prisma,
        'table_name': 'Document'
    }
)

client = VectraClient(config)
await client.ingest_documents('./docs')
result = await client.query_rag('What is the vacation policy?')
print(result['answer'])
```

---

## 6. Core Concepts

### Providers

Providers implement embeddings, generation, or both. Vectra normalizes responses and streaming across providers.

### Vector Stores

Vector stores persist embeddings and metadata. Backends are swappable via configuration.

### Chunking

* **Recursive**: Token-aware, separator-aware splitting
* **Agentic**: LLM-driven semantic propositions

### Retrieval

Configurable strategies to balance recall, precision, and latency.

### Reranking

Optional LLM-based reordering of candidate chunks.

### Metadata Enrichment

Optional per-chunk summaries, keywords, and hypothetical questions generated during ingestion.

### Query Planning & Grounding

Controls context assembly and factual grounding constraints.

### Conversation Memory

Persist multi-turn chat history across sessions.

---

## 7. Configuration Reference (Usage-Driven)

> All configuration is validated using **Pydantic** at runtime.

### Embedding

```python
embedding={
  'provider': ProviderType.OPENAI,
  'api_key': os.getenv('OPENAI_API_KEY'),
  'model_name': 'text-embedding-3-small',
  'dimensions': 1536
}
```

Use `dimensions` when using pgvector to avoid runtime mismatches.

---

### LLM

```python
llm={
  'provider': ProviderType.GEMINI,
  'api_key': os.getenv('GOOGLE_API_KEY'),
  'model_name': 'gemini-1.5-pro-latest',
  'temperature': 0.3,
  'max_tokens': 1024
}
```

Used for generation, reranking, HyDE, Multi-Query, and agentic chunking.

---

### Database

```python
database={
  'type': 'prisma',
  'client_instance': prisma,
  'table_name': 'Document'
}
```

Supports Prisma, Chroma, Qdrant, Milvus.

---

### Chunking

```python
chunking={
  'strategy': ChunkingStrategy.RECURSIVE,
  'chunk_size': 1000,
  'chunk_overlap': 200
}
```

Agentic:

```python
chunking={
  'strategy': ChunkingStrategy.AGENTIC,
  'agentic_llm': {
    'provider': ProviderType.OPENAI,
    'api_key': os.getenv('OPENAI_API_KEY'),
    'model_name': 'gpt-4o-mini'
  }
}
```

---

### Retrieval

```python
retrieval={ 'strategy': RetrievalStrategy.HYBRID }
```

Hybrid is recommended for production workloads.

---

### Reranking

```python
reranking={
  'enabled': True,
  'window_size': 20,
  'top_n': 5
}
```

---

### Memory

```python
memory={ 'enabled': True, 'type': 'in-memory', 'max_messages': 20 }
```

Redis and Postgres are supported.

---

### Observability

```python
observability={
  'enabled': True,
  'sqlite_path': 'vectra-observability.db'
}
```

---

## 8. Ingestion Pipeline

```python
await client.ingest_documents('./documents')
```

* Files or directories supported
* Recursive traversal
* Embedding cache via SHA256
* Optional rate limiting

Supported formats: PDF, DOCX, XLSX, TXT, Markdown

---

## 9. Querying & Streaming

Standard:

```python
res = await client.query_rag('Refund policy?')
```

Streaming:

```python
stream = await client.query_rag('Draft email', stream=True)
async for chunk in stream:
    print(chunk.get('delta', ''), end='')
```

---

## 10. Conversation Memory

Pass a `session_id` to preserve multi-turn context.

---

## 11. Evaluation & Quality Measurement

```python
await client.evaluate([
  { 'question': 'Capital of France?', 'expected_ground_truth': 'Paris' }
])
```

Metrics: Faithfulness, Relevance

---

## 12. CLI

### Ingest & Query

```bash
vectra ingest ./docs --config=./config.json
vectra query "What are the payment terms?" --config=./config.json --stream
```

---

### WebConfig (Config Generator UI)

```bash
vectra webconfig
```

Launches a local web UI to interactively generate and validate `vectra.config.json`.

---

### Observability Dashboard

```bash
vectra dashboard
```

Launches a local dashboard for metrics, traces, and session analysis.

---

## 13. Observability & Callbacks

Tracks metrics, traces, and chat sessions when enabled.

Callbacks allow hooking into ingestion, retrieval, reranking, and generation stages.

---

## 14. Database Schemas & Indexing

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

## 15. Extending Vectra

Implement custom vector stores by extending `VectorStore`.

---

## 16. Architecture Overview

Vectra follows a modular, provider-agnostic RAG architecture with clear separation of ingestion, retrieval, and generation pipelines.

---

## 17. Development & Contribution Guide

* Python 3.8+
* Async-first (`asyncio`)
* Pydantic-based configuration

---

## 18. Production Best Practices

* Match embedding dimensions to pgvector
* Prefer Hybrid retrieval
* Enable observability in staging
* Evaluate before changing chunk sizes

---

**Vectra (Python) scales cleanly from local prototypes to production-grade RAG platforms.**
