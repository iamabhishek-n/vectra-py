from typing import List, Dict, Any, Optional, AsyncGenerator, Tuple
import re
import json
import time
import hashlib
import asyncio
import os
import uuid
import time
from collections import OrderedDict
from .config import VectraConfig, ProviderType, ChunkingStrategy, RetrievalStrategy
from .observability import SQLiteLogger
from .telemetry import telemetry
from .processor import DocumentProcessor
from .backends.openai import OpenAIBackend
from .backends.gemini import GeminiBackend
from .backends.anthropic import AnthropicBackend
from .backends.openrouter import OpenRouterBackend
from .backends.prisma_store import PrismaVectorStore
from .backends.postgres_store import PostgresVectorStore
from .backends.chroma_store import ChromaVectorStore
from .backends.qdrant_store import QdrantVectorStore
from .backends.milvus_store import MilvusVectorStore
from .backends.huggingface import HuggingFaceBackend
from .reranker import get_reranker
from .memory import InMemoryHistory, RedisHistory, PostgresHistory
from .backends.ollama import OllamaBackend

class LRUCache:
    def __init__(self, maxsize=10000):
        self._cache = OrderedDict()
        self._maxsize = maxsize
    
    def __getitem__(self, key):
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        raise KeyError(key)

    def __setitem__(self, key, value):
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = value
        if len(self._cache) > self._maxsize:
            self._cache.popitem(last=False)
            
    def __contains__(self, key):
        return key in self._cache
        
    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

class VectraClient:
    def __init__(self, config: VectraConfig):
        self.config = config
        self.callbacks = config.callbacks or []
        self.middlewares = config.middlewares or []
        self._embedding_cache = LRUCache(config.max_cache_size)
        
        # Initialize observability
        self.logger = SQLiteLogger(config.observability)

        # Initialize telemetry
        telemetry.init(config)
        telemetry.track('sdk_initialized', {
            'vector_store': config.database.type,
            'embedding_provider': config.embedding.provider,
            'llm_provider': config.llm.provider,
            'observability_enabled': config.observability.enabled,
            'memory_enabled': config.memory.get('enabled', False) if config.memory else False,
            'session_type': config.session_type
        })

        self.embedder = self._create_embedder()
        self.llm = self._create_llm(config.llm)
        
        if config.database.type == 'prisma':
            self.vector_store = PrismaVectorStore(config.database)
        elif config.database.type == 'postgres':
            self.vector_store = PostgresVectorStore(config.database)
        elif config.database.type == 'chroma':
            self.vector_store = ChromaVectorStore(config.database)
        elif config.database.type == 'qdrant':
            self.vector_store = QdrantVectorStore(config.database)
        elif config.database.type == 'milvus':
            self.vector_store = MilvusVectorStore(config.database)
        else:
            raise ValueError(f"Unsupported database type: {config.database.type}")
            
        agentic_llm = None
        if config.chunking.strategy == ChunkingStrategy.AGENTIC and config.chunking.agentic_llm:
            agentic_llm = self._create_llm(config.chunking.agentic_llm)
            
        self.processor = DocumentProcessor(config.chunking, agentic_llm)

        mm = int((getattr(self.config, 'memory', {}) or {}).get('max_messages', 20))
        mem = (getattr(self.config, 'memory', {}) or {})
        if mem.get('enabled'):
            if mem.get('type') == 'in-memory':
                self.history = InMemoryHistory(mm)
            elif mem.get('type') == 'redis':
                rc = mem.get('redis', {}) or {}
                self.history = RedisHistory(rc.get('client_instance'), rc.get('key_prefix', 'vectra:chat:'), mm)
            elif mem.get('type') == 'postgres':
                pc = mem.get('postgres', {}) or {}
                self.history = PostgresHistory(pc.get('client_instance'), pc.get('table_name', 'ChatMessage'), pc.get('column_map', { 'sessionId': 'sessionId', 'role': 'role', 'content': 'content', 'createdAt': 'createdAt' }), mm)
            else:
                self.history = None
        else:
            self.history = None
        
        if config.retrieval and config.retrieval.llm_config:
            self.retrieval_llm = self._create_llm(config.retrieval.llm_config)
            
        self.reranker = None
        if config.reranking and config.reranking.enabled:
            rerank_llm = self._create_llm(config.reranking.llm_config) if config.reranking.llm_config else self.llm
            self.reranker = get_reranker(config.reranking, rerank_llm)

    def _create_embedder(self):
        conf = self.config.embedding
        if conf.provider == ProviderType.OPENAI: return OpenAIBackend(conf)
        if conf.provider == ProviderType.GEMINI: return GeminiBackend(conf)
        if conf.provider == ProviderType.OLLAMA: return OllamaBackend(conf)
        raise ValueError(f"Embedding provider {conf.provider} not implemented")

    def _create_llm(self, conf):
        if conf.provider == ProviderType.OPENAI: return OpenAIBackend(conf)
        if conf.provider == ProviderType.GEMINI: return GeminiBackend(conf)
        if conf.provider == ProviderType.ANTHROPIC: return AnthropicBackend(conf)
        if conf.provider == ProviderType.OPENROUTER: return OpenRouterBackend(conf)
        if conf.provider == ProviderType.HUGGINGFACE: return HuggingFaceBackend(conf)
        if conf.provider == ProviderType.OLLAMA: return OllamaBackend(conf)
        raise ValueError(f"LLM provider {conf.provider} not implemented")

    def _trigger(self, event: str, *args):
        for cb in self.callbacks:
            if hasattr(cb, event) and callable(getattr(cb, event)):
                getattr(cb, event)(*args)

    async def _run_middlewares(self, method_name: str, *args):
        if not self.middlewares:
            return args[0] if len(args) == 1 else args
        
        current_args = args
        for mw in self.middlewares:
            if hasattr(mw, method_name):
                func = getattr(mw, method_name)
                # Handle both sync and async middleware methods
                if asyncio.iscoroutinefunction(func):
                    res = await func(*current_args)
                else:
                    res = func(*current_args)
                
                # Update args for next middleware in chain
                if isinstance(res, tuple):
                    current_args = res
                else:
                    current_args = (res,)
        
        return current_args[0] if len(current_args) == 1 else current_args

    def _is_temporary_file(self, path: str) -> bool:
        name = os.path.basename(path)
        if name.startswith('~$'): return True
        if name.endswith('.tmp') or name.endswith('.temp'): return True
        if name.endswith('.crdownload') or name.endswith('.part'): return True
        if name.startswith('.'): return True
        return False
    
    async def ingest_batch(self, file_paths: List[str], ingestion_mode: str = "append"):
        all_files = []
        for p in file_paths:
            if os.path.isdir(p):
                for root, _, files in os.walk(p):
                    for f in files:
                        full = os.path.join(root, f)
                        if not self._is_temporary_file(full):
                            all_files.append(full)
            elif os.path.isfile(p):
                if not self._is_temporary_file(p):
                    all_files.append(p)

        if not all_files:
            return

        mode = str(ingestion_mode or "append").lower()
        if mode not in ("append", "skip", "replace", "upsert"):
            mode = "append"

        t0 = time.time()
        self._trigger('on_ingest_start', f"batch of {len(all_files)} files")
        
        telemetry.track('ingest_batch_started', {
            'file_count': len(all_files),
            'ingestion_mode': mode
        })

        all_documents = []
        all_chunks = []
        all_hashes = []
        all_metas = []
        file_info_list = []

        for file_path in all_files:
            abs_path = os.path.abspath(file_path)
            try:
                size = int(os.path.getsize(file_path))
                mtime = int(os.path.getmtime(file_path))
            except Exception:
                size = 0
                mtime = 0
            
            md5 = hashlib.md5()
            sha = hashlib.sha256()
            with open(file_path, 'rb') as f:
                while True:
                    b = f.read(8192)
                    if not b: break
                    md5.update(b)
                    sha.update(b)
            file_md5 = md5.hexdigest()
            file_sha256 = sha.hexdigest()
            
            exists = False
            if mode == "skip" and hasattr(self.vector_store, 'file_exists'):
                try:
                    exists = await self.vector_store.file_exists(file_sha256, size, mtime)
                except Exception:
                    exists = False
            
            if exists:
                continue

            raw_text = await self.processor.load_document(file_path)
            raw_text = await self._run_middlewares('on_before_chunk', raw_text, self.config)
            chunks = await self.processor.process(raw_text)
            hashes = [hashlib.sha256(c.encode('utf-8')).hexdigest() for c in chunks]
            metas = self.processor.compute_chunk_metadata(file_path, raw_text, chunks)
            
            file_info = {
                'path': file_path,
                'abs_path': abs_path,
                'md5': file_md5,
                'sha256': file_sha256,
                'size': size,
                'mtime': mtime,
                'chunk_range': (len(all_chunks), len(all_chunks) + len(chunks))
            }
            file_info_list.append(file_info)
            all_chunks.extend(chunks)
            all_hashes.extend(hashes)
            all_metas.extend(metas)

        if not all_chunks:
            duration_ms = int((time.time() - t0) * 1000)
            self._trigger('on_ingest_end', "batch", 0, duration_ms)
            return

        # Batch Embedding
        unique_hashes = list(set(all_hashes))
        uncached_hashes = [h for h in unique_hashes if h not in self._embedding_cache]
        
        if uncached_hashes:
            # Map hash to index in all_chunks for value retrieval
            hash_to_chunk = {}
            for i, h in enumerate(all_hashes):
                if h in uncached_hashes:
                    hash_to_chunk[h] = all_chunks[i]
            
            uncached_texts = [hash_to_chunk[h] for h in uncached_hashes]
            
            ing = self.config.ingestion
            enabled = ing.rate_limit_enabled
            default_limit = ing.concurrency_limit
            limit = default_limit if enabled else len(uncached_texts)
            
            new_embeds = [None] * len(uncached_texts)
            for i in range(0, len(uncached_texts), limit):
                batch = uncached_texts[i:i+limit]
                attempt = 0
                delay = 0.5
                while True:
                    try:
                        out = await self.embedder.embed_documents(batch)
                        # Middleware Hook: on_after_embed
                        batch, out = await self._run_middlewares('on_after_embed', batch, out)
                        for j, vec in enumerate(out):
                            new_embeds[i + j] = vec
                        break
                    except Exception as e:
                        attempt += 1
                        if attempt >= 3: raise e
                        await asyncio.sleep(delay)
                        delay = min(4.0, delay * 2)
            
            for h, vec in zip(uncached_hashes, new_embeds):
                self._embedding_cache[h] = vec

        # Prepare final documents
        documents = []
        chunk_idx_overall = 0
        for info in file_info_list:
            start, end = info['chunk_range']
            for i in range(start, end):
                meta = all_metas[i] if i < len(all_metas) else {}
                meta.update({
                    'source': info['path'],
                    'absolutePath': info['abs_path'],
                    'fileMD5': info['md5'],
                    'fileSHA256': info['sha256'],
                    'fileSize': info['size'],
                    'lastModified': info['mtime'],
                    'chunk_index': i - start,
                    'sha256': all_hashes[i]
                })
                # Filter None values
                meta = {k: v for k, v in meta.items() if v is not None}
                
                documents.append({
                    'content': all_chunks[i],
                    'embedding': self._embedding_cache[all_hashes[i]],
                    'metadata': meta
                })

        # Enrichment
        if getattr(self.config, 'metadata', None) and self.config.metadata.get('enrichment'):
            enriched = await self._enrich_chunk_metadata(all_chunks)
            for i in range(min(len(documents), len(enriched))):
                documents[i]['metadata'].update(enriched[i])

        # Store
        if hasattr(self.vector_store, 'ensure_indexes'):
            try: await self.vector_store.ensure_indexes()
            except Exception: pass

        if mode == "replace":
            abs_paths = list(set(info['abs_path'] for info in file_info_list))
            for ap in abs_paths:
                try: await self.vector_store.delete_documents({ "absolutePath": ap })
                except Exception: pass

        if mode == "upsert":
            to_add = []
            for d in documents:
                try:
                    updated = 0
                    if hasattr(self.vector_store, "update_documents"):
                        updated = await self.vector_store.update_documents(
                            { "sha256": d["metadata"].get("sha256") },
                            { "content": d["content"], "metadata": d["metadata"] },
                        )
                    if not updated and hasattr(self.vector_store, "delete_documents"):
                        await self.vector_store.delete_documents({ "sha256": d["metadata"].get("sha256") })
                    if not updated:
                        to_add.append(d)
                except Exception:
                    to_add.append(d)
            documents = to_add

        if documents:
            attempt = 0
            delay = 0.5
            while True:
                try:
                    await self.vector_store.add_documents(documents)
                    break
                except Exception as e:
                    attempt += 1
                    if attempt >= 3: raise e
                    await asyncio.sleep(delay)
                    delay = min(4.0, delay * 2)

        duration_ms = int((time.time() - t0) * 1000)
        self._trigger('on_ingest_end', "batch", len(all_chunks), duration_ms)
        
        telemetry.track('ingest_batch_completed', {
            'file_count': len(all_files),
            'chunk_count': len(all_chunks),
            'duration_ms': duration_ms
        })

    async def ingest_documents(self, file_path: str, ingestion_mode: str = "append"):
        await self.ingest_batch([file_path], ingestion_mode=ingestion_mode)

    async def list_documents(self, filter: Optional[Dict[str, Any]] = None, limit: int = 100, cursor: Optional[str] = None) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        return await self.vector_store.list_documents(filter=filter, limit=limit, cursor=cursor)

    async def delete_documents(self, filter: Dict[str, Any]) -> int:
        return await self.vector_store.delete_documents(filter)

    async def update_documents(self, filter: Dict[str, Any], update_data: Dict[str, Any]) -> int:
        return await self.vector_store.update_documents(filter, update_data)

    async def _generate_hyde_query(self, query: str) -> str:
        prompt = f'Please write a plausible passage that answers the question: "{query}".'
        return await self.retrieval_llm.generate(prompt)

    async def _generate_multi_queries(self, query: str) -> List[str]:
        prompt = f'Generate 3 different versions of the user question to retrieve relevant documents. Return them separated by newlines.\nOriginal: {query}'
        response = await self.retrieval_llm.generate(prompt)
        return [line.strip() for line in response.split('\n') if line.strip()][:3]

    async def _enrich_chunk_metadata(self, chunks: List[str], concurrency: int = 5) -> List[Dict[str, Any]]:
        sem = asyncio.Semaphore(concurrency)
        
        async def _enrich_one(c):
            async with sem:
                try:
                    prompt = f"Summarize and extract keywords and questions from the following text. Return STRICT JSON with keys: summary (string), keywords (array of strings), hypothetical_questions (array of strings).\nText:\n{c}"
                    out = await self.llm.generate(prompt, 'You return valid JSON only.')
                    clean = str(out).replace('```json','').replace('```','').strip()
                    p = json.loads(clean)
                    return { 'summary': p.get('summary',''), 'keywords': p.get('keywords', []), 'hypothetical_questions': p.get('hypothetical_questions', []) }
                except Exception:
                    # Heuristic fallback if LLM fails
                    words = re.findall(r"[a-zA-Z0-9]+", c.lower())
                    freq = {}
                    for w in words:
                        if len(w) > 3:
                            freq[w] = freq.get(w, 0) + 1
                    top = [w for w,_ in sorted(freq.items(), key=lambda kv: kv[1], reverse=True)[:10]]
                    return { 'summary': c[:300], 'keywords': top, 'hypothetical_questions': [] }
                    
        return await asyncio.gather(*[_enrich_one(c) for c in chunks])

    async def _generate_hypothetical_questions(self, query: str) -> List[str]:
        out = await self.retrieval_llm.generate(f"Generate 3 hypothetical questions related to the query. Return a VALID JSON array of strings.\nQuery: {query}")
        try:
            import json
            arr = json.loads(str(out).strip().replace('```json','').replace('```',''))
            return arr[:3] if isinstance(arr, list) else []
        except Exception:
            return []

    def _token_estimate(self, text: str) -> int:
        if not text:
            return 0
        ascii_chars = sum(1 for c in text if ord(c) < 128)
        non_ascii = len(text) - ascii_chars
        return max(1, (ascii_chars + 3) // 4 + non_ascii)

    def _build_context_parts(self, docs: List[Dict[str, Any]], query: str) -> Tuple[List[str], List[Dict[str, Any]]]:
        budget = int(self.config.query_planning.get('token_budget', 2048)) if getattr(self.config, 'query_planning', None) else 2048
        prefer_summ = int(self.config.query_planning.get('prefer_summaries_below', 1024)) if getattr(self.config, 'query_planning', None) else 1024
        parts: List[str] = []
        doc_map: List[Dict[str, Any]] = []
        used = 0
        for d in docs:
            md = d.get('metadata', {})
            t = md.get('docTitle') or ''
            sec = md.get('section') or ''
            pf = md.get('pageFrom'); pt = md.get('pageTo')
            pages = f"pages {pf}-{pt}" if pf and pt else ''
            summ = md.get('summary') or d.get('content','')[:800]
            chosen = summ if self._token_estimate(summ) <= prefer_summ else d.get('content','')[:1200]
            part = f"{t} {sec} {pages}\n{chosen}"
            est = self._token_estimate(part)
            if used + est > budget:
                break
            parts.append(part)
            doc_map.append({
                'source': md.get('source') or md.get('absolutePath', ''),
                'pageFrom': md.get('pageFrom'),
                'pageTo': md.get('pageTo'),
                'section': md.get('section'),
                'docTitle': md.get('docTitle'),
                '_content': chosen
            })
            used += est
        return parts, doc_map

    def _extract_first_sentence(self, text: str, max_len: int = 250) -> str:
        if not text:
            return ''
        m = re.match(r'^(.*?[.!?])\s', text, re.S)
        sent = m.group(1) if m else text
        return sent[:max_len] + '...' if len(sent) > max_len else sent

    def _parse_citations(self, answer: str, doc_map: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        citations: List[Dict[str, Any]] = []
        seen: set = set()
        for m in re.finditer(r'\[(\d+)\]', answer):
            idx = int(m.group(1))
            if idx in seen or idx < 1 or idx > len(doc_map):
                continue
            seen.add(idx)
            doc = doc_map[idx - 1]
            citations.append({
                'index': idx,
                'source': doc.get('source') or '',
                'page': doc.get('pageFrom'),
                'section': doc.get('section'),
                'quote': self._extract_first_sentence(doc.get('_content') or '', 250)
            })
        return citations

    async def _extract_snippets(self, docs: List[Dict[str, Any]], query: str, query_vector: List[float], max_snippets: int) -> List[str]:
        all_sentences = []
        for d in docs:
            # Use sentence splitting to extract candidates
            sents = re.split(r"(?<=[.!?])\s+", d.get('content',''))
            for s in sents:
                all_sentences.append({'doc': d, 'text': s})
                
        if not all_sentences: return []
        
        # Batch embed sentences for semantic evaluation
        sentence_texts = [s['text'] for s in all_sentences]
        sentence_embeddings = await self.embedder.embed_documents(sentence_texts)
        
        scored = []
        for i, emb in enumerate(sentence_embeddings):
            # Compute cosine similarity with query vector
            score = sum(a*b for a, b in zip(emb, query_vector))
            scored.append({'text': all_sentences[i]['text'], 'doc': all_sentences[i]['doc'], 'score': score})
            
        scored.sort(key=lambda x: x['score'], reverse=True)
        
        out = []
        for s in scored[:max_snippets]:
            d = s['doc']
            md = d.get('metadata', {})
            pf = md.get('pageFrom'); pt = md.get('pageTo')
            pages = f"pages {pf}-{pt}" if pf and pt else ''
            out.append(f"{md.get('docTitle') or ''} {md.get('section') or ''} {pages}\n{s['text']}")
        return out

    def _reciprocal_rank_fusion(self, doc_lists: List[List[Dict]], k=60) -> List[Dict]:
        """Simple RRF implementation for merging results."""
        scores = {}
        content_map = {}
        
        for doc_list in doc_lists:
            for rank, doc in enumerate(doc_list):
                content = doc['content']
                if content not in content_map:
                    content_map[content] = doc
                if content not in scores:
                    scores[content] = 0
                scores[content] += 1 / (k + rank + 1)
        
        sorted_content = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        return [content_map[c] for c in sorted_content]

    def _mmr_select(self, candidates: List[Dict[str, Any]], k: int, mmr_lambda: float) -> List[Dict[str, Any]]:
        if not candidates:
            return []
        k_int = max(1, int(k))
        lam = max(0.0, min(1.0, float(mmr_lambda)))

        def tokens(text: str) -> set:
            return set(t for t in re.findall(r"[a-zA-Z0-9]+", (text or "").lower()) if len(t) > 2)

        cand = []
        for d in candidates:
            dd = dict(d)
            dd["_tokens"] = tokens(dd.get("content", ""))
            dd["_rel"] = float(dd.get("score", 0.0) or 0.0)
            cand.append(dd)

        cand.sort(key=lambda x: x.get("_rel", 0.0), reverse=True)
        selected: List[Dict[str, Any]] = []
        selected_tokens: List[set] = []

        first = cand.pop(0)
        selected.append(first)
        selected_tokens.append(first.get("_tokens") or set())

        def jaccard(a: set, b: set) -> float:
            if not a or not b:
                return 0.0
            inter = len(a & b)
            if inter == 0:
                return 0.0
            union = len(a | b)
            return inter / union if union else 0.0

        while cand and len(selected) < k_int:
            best_idx = -1
            best_score = None
            for i, d in enumerate(cand):
                rel = d.get("_rel", 0.0)
                dt = d.get("_tokens") or set()
                div = 0.0
                for st in selected_tokens:
                    div = max(div, jaccard(dt, st))
                score = lam * rel - (1.0 - lam) * div
                if best_score is None or score > best_score:
                    best_score = score
                    best_idx = i
            if best_idx < 0:
                break
            picked = cand.pop(best_idx)
            selected.append(picked)
            selected_tokens.append(picked.get("_tokens") or set())

        out = []
        for d in selected[:k_int]:
            dd = dict(d)
            dd.pop("_tokens", None)
            dd.pop("_rel", None)
            out.append(dd)
        return out

    async def query_rag(self, query: str, filter: Optional[Dict] = None, stream: bool = False, session_id: Optional[str] = None) -> Dict[str, Any] | AsyncGenerator[str, None]:
        trace_id = str(uuid.uuid4())
        root_span_id = str(uuid.uuid4())
        
        provider = self.config.llm.provider
        model_name = self.config.llm.model_name

        if session_id:
             self.logger.update_session(session_id, None, {'last_query': query})

        try:
            t_start = time.time()
            t_ret = time.time()
            self._trigger('on_retrieval_start', query)
            
            strategy = self.config.retrieval.strategy
            docs = []
            k = self.config.reranking.window_size if (self.config.reranking and self.config.reranking.enabled) else 5
            
            query_vector = await self.embedder.embed_query(query)
            # Middleware Hook: on_before_retrieve
            query, query_vector = await self._run_middlewares('on_before_retrieve', query, query_vector)

            if strategy == RetrievalStrategy.HYDE:
                hypothetical_doc = await self._generate_hyde_query(query)
                hyde_vector = await self.embedder.embed_query(hypothetical_doc)
                # Weighted average: 70% HyDE, 30% original query
                combined_vector = [(0.7 * h + 0.3 * q) for h, q in zip(hyde_vector, query_vector)]
                docs = await self.vector_store.similarity_search(combined_vector, k, filter)
                
            elif strategy == RetrievalStrategy.MULTI_QUERY:
                queries = await self._generate_multi_queries(query)
                if getattr(self.config, 'query_planning', None):
                    hyps = await self._generate_hypothetical_questions(query)
                    queries.extend(hyps)
                queries.append(query)
                
                # Concurrent embedding of all queries
                embed_tasks = [self.embedder.embed_query(q) for q in queries]
                vectors = await asyncio.gather(*embed_tasks)
                
                # Concurrent similarity search for all generated vectors
                search_tasks = [self.vector_store.similarity_search(v, k, filter) for v in vectors]
                result_lists = await asyncio.gather(*search_tasks)
                
                docs = self._reciprocal_rank_fusion(result_lists, k=60)
                
            elif strategy == RetrievalStrategy.HYBRID:
                 # Hybrid = Vector Search + (Simulated) Keyword Search via specialized Store method
                 docs = await self.vector_store.hybrid_search(query, query_vector, k, filter)

            elif strategy == RetrievalStrategy.MMR:
                fetch_k = int(getattr(self.config.retrieval, "mmr_fetch_k", 20))
                mmr_lam = float(getattr(self.config.retrieval, "mmr_lambda", 0.5))
                candidates = await self.vector_store.similarity_search(query_vector, max(fetch_k, k), filter)
                docs = self._mmr_select(candidates, k, mmr_lam)

            else: # NAIVE
                docs = await self.vector_store.similarity_search(query_vector, k, filter)
                
            if self.config.reranking and self.config.reranking.enabled and self.reranker:
                self._trigger('on_reranking_start', len(docs))
                docs = await self.reranker.rerank(query, docs)
                self._trigger('on_reranking_end', len(docs))
                
            self._trigger('on_retrieval_end', len(docs), int((time.time() - t_ret) * 1000))
            
            embedding_provider = self.config.embedding.provider
            embedding_model_name = self.config.embedding.model_name
            self.logger.log_trace({
                'trace_id': trace_id,
                'span_id': str(uuid.uuid4()),
                'parent_span_id': root_span_id,
                'name': 'retrieval',
                'start_time': int(t_ret * 1000),
                'end_time': int(time.time() * 1000),
                'input': {'query': query, 'filter': filter, 'strategy': strategy},
                'output': {'documents_found': len(docs)},
                'provider': embedding_provider,
                'model_name': embedding_model_name
            })

            terms = [t for t in re.findall(r"[a-zA-Z0-9]+", query.lower()) if len(t) > 2]
            boosted = []
            for d in docs:
                kws = [str(k).lower() for k in (d.get('metadata',{}).get('keywords') or [])]
                match = sum(1 for t in terms if t in kws)
                dd = dict(d)
                dd['_boost'] = match
                boosted.append(dd)
            boosted.sort(key=lambda x: (x.get('score', 0) + 0.1 * x.get('_boost', 0)), reverse=True)

            gen_conf = getattr(self.config, 'generation', None) or {}
            citations_enabled = gen_conf.get('structured_output') == 'citations' \
                and not (getattr(self.config, 'grounding', None) and self.config.grounding.get('strict'))

            context_parts, doc_map = self._build_context_parts(boosted, query)
            if citations_enabled:
                context_parts = [f"[{i + 1}] {p}" for i, p in enumerate(context_parts)]
            if getattr(self.config, 'grounding', None) and self.config.grounding.get('enabled'):
                max_snippets = int(self.config.grounding.get('max_snippets', 3))
                snippets = await self._extract_snippets(boosted, query, query_vector, max_snippets)
                if self.config.grounding.get('strict'):
                    context_parts = snippets
                else:
                    context_parts.extend(snippets)
            context = "\n---\n".join(context_parts)
            import inspect
            history_text = ""
            if self.history and session_id:
                get_recent = getattr(self.history, 'get_recent', None)
                if callable(get_recent):
                    if inspect.iscoroutinefunction(get_recent):
                        recent = await get_recent(session_id, int((getattr(self.config, 'memory', {}) or {}).get('max_messages', 10)))
                    else:
                        recent = get_recent(session_id, int((getattr(self.config, 'memory', {}) or {}).get('max_messages', 10)))
                    history_text = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in recent])
            if getattr(self.config, 'prompts', None) and self.config.prompts.get('query'):
                prompt = str(self.config.prompts.get('query')).replace('{{context}}', context).replace('{{question}}', query)
                if history_text:
                    prompt = f"Conversation:\n{history_text}\n\n{prompt}"
            else:
                conv_section = f"Conversation:\n{history_text}\n\n" if history_text else ""
                if citations_enabled:
                    prompt = f"Answer the question using the provided context. Cite sources using inline markers like [1], [2], etc., matching the numbered context chunks. Every factual claim must have a citation.\nContext:\n{context}\n\n{conv_section}Question: {query}"
                else:
                    prompt = f"Answer the question using the provided summaries and cite titles/sections/pages where relevant.\nContext:\n{context}\n\n{conv_section}Question: {query}"
            
            t_gen = time.time()
            self._trigger('on_generation_start', prompt)
            
            system_inst = "You are a helpful RAG assistant. When answering, cite sources using inline markers like [1], [2], etc., matching the numbered context chunks provided. Every factual claim must have a citation." if citations_enabled else "You are a helpful RAG assistant."
            if stream:
                # Streaming Logic
                self.logger.log_trace({
                    'trace_id': trace_id,
                    'span_id': str(uuid.uuid4()),
                    'parent_span_id': root_span_id,
                    'name': 'generation_stream_start',
                    'start_time': int(t_gen * 1000),
                    'end_time': int(time.time() * 1000),
                    'input': {'prompt': prompt},
                    'output': {'stream': True},
                    'provider': provider,
                    'model_name': model_name
                })
                original_stream = self.llm.generate_stream(prompt, system_inst)
                _self = self
                _citations_enabled = citations_enabled
                _doc_map = doc_map

                async def _wrapped_stream():
                    full_answer = ''
                    async for chunk in original_stream:
                        delta = chunk.get('delta', '') if isinstance(chunk, dict) else (chunk if isinstance(chunk, str) else '')
                        full_answer += delta
                        yield chunk
                    if _citations_enabled:
                        yield {'type': 'citations', 'citations': _self._parse_citations(full_answer, _doc_map)}

                return _wrapped_stream()
            else:
                answer = await self.llm.generate(prompt, system_inst)
                # Middleware Hook: on_after_generate
                sources = [d['metadata'] for d in docs]
                answer, sources = await self._run_middlewares('on_after_generate', answer, sources)
                if self.history and session_id:
                    add_msg = getattr(self.history, 'add_message', None)
                    if callable(add_msg):
                        if inspect.iscoroutinefunction(add_msg):
                            await add_msg(session_id, 'user', query)
                            await add_msg(session_id, 'assistant', str(answer))
                        else:
                            add_msg(session_id, 'user', query)
                            add_msg(session_id, 'assistant', str(answer))
                
                gen_ms = int((time.time() - t_gen) * 1000)
                self._trigger('on_generation_end', answer, gen_ms)
                
                prompt_chars = len(prompt)
                answer_chars = len(str(answer))

                self.logger.log_trace({
                    'trace_id': trace_id,
                    'span_id': str(uuid.uuid4()),
                    'parent_span_id': root_span_id,
                    'name': 'generation',
                    'start_time': int(t_gen * 1000),
                    'end_time': int(time.time() * 1000),
                    'input': {'prompt': prompt},
                    'output': {'answer': str(answer)[:1000]},
                    'attributes': {'prompt_chars': prompt_chars, 'completion_chars': answer_chars},
                    'provider': provider,
                    'model_name': model_name
                })
                
                self.logger.log_metric({'name': 'prompt_chars', 'value': prompt_chars})
                self.logger.log_metric({'name': 'completion_chars', 'value': answer_chars})

                self.logger.log_trace({
                    'trace_id': trace_id,
                    'span_id': root_span_id,
                    'name': 'query_rag',
                    'start_time': int(t_start * 1000),
                    'end_time': int(time.time() * 1000),
                    'input': {'query': query, 'session_id': session_id},
                    'output': {'success': True},
                    'attributes': {'retrieval_ms': int((t_gen - t_ret) * 1000), 'gen_ms': gen_ms, 'doc_count': len(docs)},
                    'provider': provider,
                    'model_name': model_name
                })
                self.logger.log_metric({'name': 'query_latency', 'value': int((time.time() - t_start) * 1000), 'tags': {'type': 'total'}})

                telemetry.track('query_executed', {
                    'query_mode': 'rag',
                    'retrieval_strategy': strategy,
                    'reranking_enabled': bool(self.config.reranking and self.config.reranking.enabled),
                    'streaming': stream,
                    'memory_used': bool(self.history and session_id),
                    'result_count': len(docs)
                })

                if gen_conf.get('output_format') == 'json':
                    try:
                        parsed = json.loads(str(answer))
                        result = {'answer': parsed, 'sources': [d['metadata'] for d in docs]}
                        if citations_enabled:
                            result['citations'] = self._parse_citations(str(answer), doc_map)
                        return result
                    except Exception:
                        result = {'answer': answer, 'sources': [d['metadata'] for d in docs]}
                        if citations_enabled:
                            result['citations'] = self._parse_citations(str(answer), doc_map)
                        return result
                if citations_enabled:
                    citations = self._parse_citations(str(answer), doc_map)
                    return {'answer': answer, 'citations': citations, 'sources': [d['metadata'] for d in docs]}
                return {'answer': answer, 'sources': [d['metadata'] for d in docs]}
            
        except Exception as e:
            telemetry.track('error_occurred', {
                'stage': 'retrieval' if 't_ret' in locals() and 't_gen' not in locals() else 'generation',
                'error_type': 'unknown'
            })
            self._trigger('on_error', e)
            if 'trace_id' in locals():
                self.logger.log_trace({
                    'trace_id': trace_id,
                    'span_id': root_span_id,
                    'name': 'query_rag',
                    'start_time': int(t_start * 1000) if 't_start' in locals() else int(time.time() * 1000),
                    'end_time': int(time.time() * 1000),
                    'input': {'query': query},
                    'error': {'message': str(e)},
                    'status': 'error'
                })
            raise e

    async def evaluate(self, test_set: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        telemetry.track('evaluation_run', {
            'dataset_size_bucket': '1-5' if len(test_set) <= 5 else '5-20' if len(test_set) <= 20 else '20+'
        })
        report: List[Dict[str, Any]] = []
        for item in test_set:
            query = item['question']
            ground_truth = item.get('expectedGroundTruth', '')
            res = await self.query_rag(query)
            answer = res.get('answer', '')
            sources = res.get('sources', [])
            context = "\n".join([f"[Source {i+1}] {s.get('content', s.get('summary', ''))}" for i, s in enumerate(sources)])

            # 1. Faithfulness (Claim-level)
            faith_prompt = f"""Given the context and the answer, determine if every claim in the answer is supported by the context.
Context: {context}
Answer: {answer}
Return JSON: {{"claims": [{{"claim": "...", "supported": true/false, "evidence": "..."}}], "score": 0.0-1.0}}"""
            
            # 2. Context Precision (Chunk relevance)
            precision_results = []
            for i, src in enumerate(sources):
                chunk = src.get('content', src.get('summary', ''))
                prec_prompt = f"Query: {query}\nChunk: {chunk}\nIs this chunk relevant to the query? Return JSON: {{\"relevant\": true/false}}"
                try:
                    p_res = await self.llm.generate(prec_prompt, "Return valid JSON.")
                    p_json = json.loads(re.search(r'\{.*\}', p_res, re.S).group(0))
                    precision_results.append(1.0 if p_json.get('relevant') else 0.0)
                except: precision_results.append(0.0)
            context_precision = sum(precision_results) / len(precision_results) if precision_results else 0.0

            # 3. Context Recall (GT facts in Context)
            recall_prompt = f"Ground Truth: {ground_truth}\nContext: {context}\nDoes the context contain the facts needed for the ground truth? Return JSON: {{\"facts\": [{{\"fact\": \"...\", \"present\": true/false}}], \"score\": 0.0-1.0}}"
            
            # 4. Answer Correctness (LLM-as-judge)
            correctness_prompt = f"Question: {query}\nGenerated Answer: {answer}\nGround Truth: {ground_truth}\nRate correctness (0-1). Return JSON: {{\"score\": 0.0-1.0, \"reason\": \"...\"}}"

            metrics = {'faithfulness': 0.0, 'context_recall': 0.0, 'answer_correctness': 0.0}
            for key, prompt in [('faithfulness', faith_prompt), ('context_recall', recall_prompt), ('answer_correctness', correctness_prompt)]:
                try:
                    m_res = await self.llm.generate(prompt, "Return valid JSON.")
                    m_json = json.loads(re.search(r'\{.*\}', m_res, re.S).group(0))
                    metrics[key] = m_json.get('score', 0.0)
                except: pass

            report.append({
                'question': query,
                'expectedGroundTruth': ground_truth,
                'answer': answer,
                'metrics': {
                    'faithfulness': metrics['faithfulness'],
                    'relevance': metrics['answer_correctness'], # Map to existing field for compat
                    'context_precision': context_precision,
                    'context_recall': metrics['context_recall'],
                    'answer_correctness': metrics['answer_correctness']
                }
            })
        return report
