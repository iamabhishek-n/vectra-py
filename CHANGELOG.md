# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Entries below were reconstructed from `git log` and the version history in
`pyproject.toml`. Where a version's exact scope wasn't clear from commit
messages alone, the entry is intentionally brief and generic rather than
guessed.

## [Unreleased]

Work landed on top of the `1.0.0` release, not yet published to PyPI under a
new version number.

### Added
- `pytest` test runner and substantially expanded test coverage: reciprocal
  rank fusion and MMR selection, telemetry/config behavior, and integration
  tests for the Postgres, Prisma, Chroma, and Qdrant vector stores.
- `max_query_length` and `block_pii` guardrails, plus a `content_filter`
  guardrail, all wired into `query_rag`.
- Configurable max file size limit for ingestion.
- CI: run tests and `pip-audit` on every push and pull request.
- `SECURITY.md`, documenting guardrail enforcement and telemetry behavior.
- Real hybrid search implementations for the Chroma, Qdrant, and Milvus
  vector stores (previously stubbed).
- Real Cohere and Jina reranker integrations via `aiohttp`.
- Embedding-space MMR diversity selection, falling back to lexical Jaccard
  similarity when no embeddings are available.
- Real BPE tokenization via `tiktoken`, replacing a character-count
  heuristic, with an offline-safe fallback when `tiktoken`'s encoder data
  can't be downloaded.
- `CONTRIBUTING.md`, issue templates (bug report, feature request), a pull
  request template, and documented manual branch-protection setup
  (`docs/BRANCH_PROTECTION_SETUP.md`).
- `ruff` linting, wired into CI, against a narrow high-signal rule set
  (pyflakes + pycodestyle errors).
- A context and memory layer: `client.context.ask`, a budget-aware packing
  primitive (`build_context`) that fuses retrieved docs, durable facts, tool
  output, and history into one prompt, reporting `dropped` and `warnings`
  explicitly instead of silently truncating.
- A bi-temporal fact store (`FactStore`, `memory['facts']` config): facts
  extracted from conversations via LLM, with contradiction detection that
  marks a superseded fact invalid at that point in time instead of deleting
  it. Enable with `memory['facts']['enabled']`, then
  `client.fact_store.write(...)` and `client.fact_store.read(...)`.
- Multi-database fusion: a `docs` context source can take a `stores` list
  instead of one vector store, fanning out concurrently with a per-store
  timeout and fusing results with reciprocal rank fusion. A slow or failed
  store produces a warning, not a failed call.
- `context_layer['budget']` and `context_layer['priority']` config,
  controlling `context.ask`'s token budget and source packing order.
- Two new vector store backends: Pinecone (client-side hybrid search RRF
  fallback; `list_documents`/`update_documents` raise `NotImplementedError`,
  no native listing endpoint) and Weaviate (native hybrid search and native
  filter-based listing, full CRUD support, no fallback needed).

### Fixed
- `context_layer` config had no field declared on `VectraConfig`, so
  Pydantic's default `extra='ignore'` silently dropped it. `context.ask`'s
  budget and priority were always the hardcoded 2048-token default no matter
  what a caller configured. Now declared and respected.
- Telemetry now defaults to off and is opt-in only.
- `Milvus` store: `delete_documents` now returns the real delete count;
  `update_documents` implemented for both Milvus and Postgres, merging
  metadata instead of replacing it and skipping documents with a missing
  vector; SQL identifier sanitization added to `PostgresVectorStore` (was
  completely absent); filter values escaped and unsafe keys rejected in
  Milvus's filter-expression builder; unsupported filter value types now
  raise instead of being silently mishandled.
- ReDoS-vulnerable quantifiers in the email PII detection regex fully
  bounded.
- Vector store embedding dimension now respects configuration instead of
  hardcoding `1536`.
- `pyproject.toml` `license` field aligned with the MIT `LICENSE` file
  (previously contradictory).
- Milvus scores normalized by metric type; reranker/hybrid/multi-query/MMR
  ordering preserved through `query_rag`; L2 score normalization guarded
  against negative distances; an unauthorized Milvus distance inversion was
  reverted.
- MMR strategy skips an unnecessary embedding call when `fetch_k <= k`.
- README `pip install` command corrected to `vectra-rag-py`; README
  telemetry section corrected to reflect the opt-in default.
- Untracked `.egg-info` build artifact; fixed a `.gitignore` typo.

### Changed
- README repositioned around two co-equal pillars, RAG and the context/memory
  layer, instead of presenting the context layer as a subsection of a
  RAG-first pitch. Vector store list, feature matrix, and config reference
  updated for Pinecone and Weaviate.

## [1.0.0] - 2026-04-01
### Added
- Guardrails and "Vectra Middleware": configurable guardrail enforcement and
  substantial fixes/refactors across the core pipeline and all backend
  implementations (Anthropic, Chroma, Milvus, Ollama, Postgres, Prisma,
  Qdrant, HuggingFace).

### Changed
- Version bumped from `0.9.11` to `1.0.0`, with accompanying dependency and
  metadata updates in `pyproject.toml`, in a follow-up commit made three
  minutes after the above ("Production Release").

## [0.9.11] - 2026-01-06
### Fixed
- SonarCube-flagged issues addressed ("Sonar Fix" / "Updated SonaCube
  fixes"); no further detail recoverable from the commit messages.

### Changed
- Version-only bump; no functional changes recorded in this commit itself.

## [0.9.10] - 2026-01-06
### Changed
- Unspecified documentation update (commit message: "- Documentation
  update"); README grew substantially and a couple of standalone test/db
  files from the observability work were removed. No further detail
  recoverable from the commit message.
- Version-only bump; no functional changes recorded in this commit itself.

## [0.9.8] - 2026-01-05
### Added
- Version and downloads badges to README.

### Changed
- README restructured for clarity and completeness.
- `.gitignore` updated to exclude `build`/`dist` artifacts;
  `CODE_OF_CONDUCT.md` added.

### Removed
- Committed `build/`/`dist/` build artifacts, previously tracked by mistake.

## [0.9.7] - 2026-01-03
### Added
- Native PostgreSQL vector store support.

### Changed
- Error handling improvements across the codebase.

Note: this version jumped directly from `0.9.0`; versions `0.9.1`-`0.9.6`
are not represented anywhere in this repository's `pyproject.toml` history
and are not documented here.

## [0.9.0] - 2025-12-28
### Added
- SQLite-based observability with a dashboard UI (trace visualization,
  session tracking, a `dashboard` CLI command).
- Dark mode support and an updated color scheme for the dashboard UI.
- GitHub Actions workflow to publish the package to PyPI.

### Changed
- Package renamed from `vectra-py` to `vectra-rag-py`; version reset from
  `1.0.0` to `0.9.0` to reflect pre-release status ahead of the actual
  `1.0.0` production release above.

---

The initial commit that implemented the SDK (`0e05b57`, 2025-12-21) shipped
under the name `vectra-py` at version `1.0.0`. That version number was a
placeholder rather than a considered release and was superseded a week
later by the rename and reset described under `0.9.0` above; it is
intentionally not given its own entry here to avoid confusion with the
current, actual `1.0.0` release.
