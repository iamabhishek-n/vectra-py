from typing import List, Dict, Any, Optional, AsyncGenerator
import re
import hashlib
import asyncio
import os
import uuid
from .config import VectraConfig, ProviderType, ChunkingStrategy, RetrievalStrategy
from .observability import SQLiteLogger
from .processor import DocumentProcessor
from .backends.openai import OpenAIBackend
from .backends.gemini import GeminiBackend
from .backends.anthropic import AnthropicBackend
from .backends.openrouter import OpenRouterBackend
from .backends.prisma_store import PrismaVectorStore
from .backends.chroma_store import ChromaVectorStore
from .backends.qdrant_store import QdrantVectorStore
from .backends.milvus_store import MilvusVectorStore
from .backends.huggingface import HuggingFaceBackend
from .reranker import LLMReranker
from .memory import InMemoryHistory, RedisHistory, PostgresHistory
from .backends.ollama import OllamaBackend

class VectraClient:
    def __init__(self, config: VectraConfig):
        self.config = config
        self.callbacks = config.callbacks or []
        self._embedding_cache: Dict[str, List[float]] = {}
        
        # Initialize observability
        self.logger = SQLiteLogger(config.observability)

        self.embedder = self._create_embedder()
        self.llm = self._create_llm(config.llm)
        
        if config.database.type == 'prisma':
            self.vector_store = PrismaVectorStore(config.database)
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
            self.reranker = LLMReranker(rerank_llm, config.reranking)

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

    def _is_temporary_file(self, path: str) -> bool:
        name = os.path.basename(path)
        if name.startswith('~$'): return True
        if name.endswith('.tmp') or name.endswith('.temp'): return True
        if name.endswith('.crdownload') or name.endswith('.part'): return True
        if name.startswith('.'): return True
        return False
    
    async def ingest_documents(self, file_path: str, ingestion_mode: str = "append"):
        if os.path.isdir(file_path):
            summary = { 'processed': 0, 'succeeded': 0, 'failed': 0, 'errors': [] }
            for root, _, files in os.walk(file_path):
                for file in files:
                    full = os.path.join(root, file)
                    if self._is_temporary_file(full): 
                        continue
                    summary['processed'] += 1
                    try:
                        await self.ingest_documents(full, ingestion_mode=ingestion_mode)
                        summary['succeeded'] += 1
                    except Exception as e:
                        summary['failed'] += 1
                        summary['errors'].append({ 'file': full, 'message': str(e) })
                        self._trigger('on_error', e)
            self._trigger('on_ingest_summary', summary)
            return

        try:
            import time
            t0 = time.time()
            trace_id = str(uuid.uuid4())
            root_span_id = str(uuid.uuid4())
            provider = self.config.embedding.provider
            model_name = self.config.embedding.model_name
            
            self._trigger('on_ingest_start', file_path)
            
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
            validation = { 'absolutePath': abs_path, 'fileMD5': file_md5, 'fileSHA256': file_sha256, 'fileSize': size, 'lastModified': mtime, 'timestamp': int(time.time()*1000) }
            self._trigger('on_pre_ingestion_validation', validation)
            exists = False
            if hasattr(self.vector_store, 'file_exists'):
                try:
                    exists = await self.vector_store.file_exists(file_sha256, size, mtime)
                except Exception:
                    exists = False
            mode = str(ingestion_mode or "append").lower()
            if mode not in ("append", "skip", "replace", "upsert"):
                mode = "append"
            if exists and mode in ("append", "skip"):
                self._trigger('on_ingest_skipped', validation)
                return
            raw_text = await self.processor.load_document(file_path)
            
            self._trigger('on_chunking_start', self.config.chunking.strategy)
            chunks = await self.processor.process(raw_text)
            
            self._trigger('on_embedding_start', len(chunks))
            hashes = [hashlib.sha256(c.encode('utf-8')).hexdigest() for c in chunks]
            uncached = [chunks[i] for i,h in enumerate(hashes) if h not in self._embedding_cache]
            if len(uncached) > 0:
                ing = (getattr(self.config, 'ingestion', {}) or {})
                enabled = bool(ing.get('rate_limit_enabled', False))
                default_limit = int(ing.get('concurrency_limit', 5))
                limit = default_limit if enabled else len(uncached)
                new_embeds: List[List[float]] = []
                for i in range(0, len(uncached), limit):
                    batch = uncached[i:i+limit]
                    attempt = 0
                    delay = 0.5
                    while True:
                        try:
                            out = await self.embedder.embed_documents(batch)
                            new_embeds.extend(out)
                            break
                        except Exception as e:
                            attempt += 1
                            if attempt >= 3:
                                raise e
                            await asyncio.sleep(delay)
                            delay = min(4.0, delay * 2)
                j = 0
                for i,h in enumerate(hashes):
                    if h not in self._embedding_cache:
                        self._embedding_cache[h] = new_embeds[j]
                        j += 1
            embeddings = [self._embedding_cache[h] for h in hashes]
            metas = self.processor.compute_chunk_metadata(file_path, raw_text, chunks)
            documents = []
            for i, content in enumerate(chunks):
                md = metas[i] if i < len(metas) else {}
                documents.append({
                    'content': content,
                    'embedding': embeddings[i],
                    'metadata': {
                        'source': file_path,
                        'absolutePath': abs_path,
                        'fileMD5': file_md5,
                        'fileSHA256': file_sha256,
                        'fileSize': size,
                        'lastModified': mtime,
                        'chunk_index': i,
                        'sha256': hashes[i],
                        'fileType': md.get('fileType'),
                        'docTitle': md.get('docTitle'),
                        'pageFrom': md.get('pageFrom'),
                        'pageTo': md.get('pageTo'),
                        'section': md.get('section')
                    }
                })

            if getattr(self.config, 'metadata', None) and self.config.metadata.get('enrichment'): 
                import json
                enriched = []
                for c in chunks:
                    try:
                        prompt = f"Summarize and extract keywords and questions from the following text. Return STRICT JSON with keys: summary (string), keywords (array of strings), hypothetical_questions (array of strings).\nText:\n{c}"
                        out = await self.llm.generate(prompt, 'You return valid JSON only.')
                        clean = str(out).replace('```json','').replace('```','').strip()
                        p = json.loads(clean)
                        enriched.append({ 'summary': p.get('summary',''), 'keywords': p.get('keywords', []), 'hypothetical_questions': p.get('hypothetical_questions', []) })
                    except Exception:
                        import re
                        words = re.findall(r"[a-zA-Z0-9]+", c.lower())
                        freq = {}
                        for w in words:
                            if len(w) > 3:
                                freq[w] = freq.get(w, 0) + 1
                        top = [w for w,_ in sorted(freq.items(), key=lambda kv: kv[1], reverse=True)[:10]]
                        enriched.append({ 'summary': c[:300], 'keywords': top, 'hypothetical_questions': [] })
                for i in range(len(documents)):
                    documents[i]['metadata']['summary'] = enriched[i]['summary']
                    documents[i]['metadata']['keywords'] = enriched[i]['keywords']
                    documents[i]['metadata']['hypothetical_questions'] = enriched[i]['hypothetical_questions']
                
            if hasattr(self.vector_store, 'ensure_indexes'):
                try:
                    await self.vector_store.ensure_indexes()
                except Exception:
                    pass
            exists_server = False
            if hasattr(self.vector_store, 'file_exists'):
                try:
                    exists_server = await self.vector_store.file_exists(file_sha256, size, mtime)
                except Exception:
                    exists_server = False
            if exists_server and mode in ("append", "skip"):
                self._trigger('on_ingest_skipped', validation)
                return

            if mode == "replace":
                try:
                    await self.vector_store.delete_documents({ "absolutePath": abs_path })
                except Exception:
                    pass

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
                if not documents:
                    duration_ms = int((time.time() - t0) * 1000)
                    self._trigger('on_ingest_end', file_path, 0, duration_ms)
                    return
            attempt = 0
            delay = 0.5
            while True:
                try:
                    await self.vector_store.add_documents(documents)
                    break
                except Exception as e:
                    attempt += 1
                    if attempt >= 3:
                        raise e
                    await asyncio.sleep(delay)
                    delay = min(4.0, delay * 2)
            duration_ms = int((time.time() - t0) * 1000)
            self._trigger('on_ingest_end', file_path, len(chunks), duration_ms)
            
            self.logger.log_trace({
                'trace_id': trace_id,
                'span_id': root_span_id,
                'name': 'ingest_documents',
                'start_time': int(t0 * 1000),
                'end_time': int(time.time() * 1000),
                'input': {'file_path': file_path},
                'output': {'chunks': len(chunks), 'duration_ms': duration_ms},
                'attributes': {'file_size': size},
                'provider': provider,
                'model_name': model_name
            })
            self.logger.log_metric({'name': 'ingest_latency', 'value': duration_ms, 'tags': {'type': 'single_file'}})
            
        except Exception as e:
            self._trigger('on_error', e)
            if 'trace_id' in locals():
                 self.logger.log_trace({
                    'trace_id': trace_id,
                    'span_id': root_span_id,
                    'name': 'ingest_documents',
                    'start_time': int(t0 * 1000),
                    'end_time': int(time.time() * 1000),
                    'input': {'file_path': file_path},
                    'error': {'message': str(e)},
                    'status': 'error',
                    'provider': provider,
                    'model_name': model_name
                })
            raise e

    async def list_documents(self, filter: Optional[Dict[str, Any]] = None, limit: int = 100) -> List[Dict[str, Any]]:
        return await self.vector_store.list_documents(filter=filter, limit=limit)

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

    async def _generate_hypothetical_questions(self, query: str) -> List[str]:
        out = await self.retrieval_llm.generate(f"Generate 3 hypothetical questions related to the query. Return a VALID JSON array of strings.\nQuery: {query}")
        try:
            import json
            arr = json.loads(str(out).strip().replace('```json','').replace('```',''))
            return arr[:3] if isinstance(arr, list) else []
        except Exception:
            return []

    def _token_estimate(self, text: str) -> int:
        return max(1, (len(text or '') + 3) // 4)

    def _build_context_parts(self, docs: List[Dict[str, Any]], query: str) -> List[str]:
        budget = int(self.config.query_planning.get('token_budget', 2048)) if getattr(self.config, 'query_planning', None) else 2048
        prefer_summ = int(self.config.query_planning.get('prefer_summaries_below', 1024)) if getattr(self.config, 'query_planning', None) else 1024
        parts: List[str] = []
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
            used += est
        return parts

    def _extract_snippets(self, docs: List[Dict[str, Any]], query: str, max_snippets: int) -> List[str]:
        terms = [t for t in re.findall(r"[a-zA-Z0-9]+", query.lower()) if len(t) > 2]
        out: List[str] = []
        for d in docs:
            sents = re.split(r"(?<=[.!?])\s+", d.get('content',''))
            for s in sents:
                l = s.lower()
                score = sum(1 for t in terms if t in l)
                if score > 0:
                    md = d.get('metadata', {})
                    pf = md.get('pageFrom'); pt = md.get('pageTo')
                    pages = f"pages {pf}-{pt}" if pf and pt else ''
                    out.append(f"{md.get('docTitle') or ''} {md.get('section') or ''} {pages}\n{s}")
                    if len(out) >= max_snippets:
                        return out
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
            import time
            t_start = time.time()
            t_ret = time.time()
            self._trigger('on_retrieval_start', query)
            
            strategy = self.config.retrieval.strategy
            docs = []
            k = self.config.reranking.window_size if (self.config.reranking and self.config.reranking.enabled) else 5
            
            query_vector = await self.embedder.embed_query(query)

            if strategy == RetrievalStrategy.HYDE:
                hypothetical_doc = await self._generate_hyde_query(query)
                hyde_vector = await self.embedder.embed_query(hypothetical_doc)
                docs = await self.vector_store.similarity_search(hyde_vector, k, filter)
                
            elif strategy == RetrievalStrategy.MULTI_QUERY:
                queries = await self._generate_multi_queries(query)
                if getattr(self.config, 'query_planning', None):
                    hyps = await self._generate_hypothetical_questions(query)
                    queries.extend(hyps)
                queries.append(query)
                result_lists = []
                for q in queries:
                    vec = await self.embedder.embed_query(q)
                    result_lists.append(await self.vector_store.similarity_search(vec, k, filter))
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
            boosted.sort(key=lambda x: x.get('_boost',0), reverse=True)
            context_parts = self._build_context_parts(boosted, query)
            if getattr(self.config, 'grounding', None) and self.config.grounding.get('enabled'):
                max_snippets = int(self.config.grounding.get('max_snippets', 3))
                snippets = self._extract_snippets(boosted, query, max_snippets)
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
                prompt = f"Answer the question using the provided summaries and cite titles/sections/pages where relevant.\nContext:\n{context}\n\n{f'Conversation:\n{history_text}\n\n' if history_text else ''}Question: {query}"
            
            t_gen = time.time()
            self._trigger('on_generation_start', prompt)
            
            system_inst = "You are a helpful RAG assistant."
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
                return self.llm.generate_stream(prompt, system_inst) # Need to implement this in LLMs
            else:
                answer = await self.llm.generate(prompt, system_inst)
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

                if getattr(self.config, 'generation', None) and self.config.generation.get('output_format') == 'json':
                    try:
                        import json
                        parsed = json.loads(str(answer))
                        return {'answer': parsed, 'sources': [d['metadata'] for d in docs]}
                    except Exception:
                        return {'answer': answer, 'sources': [d['metadata'] for d in docs]}
                return {'answer': answer, 'sources': [d['metadata'] for d in docs]}
            
        except Exception as e:
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
        report: List[Dict[str, Any]] = []
        for item in test_set:
            res = await self.query_rag(item['question'])
            ctx = "\n".join([s.get('summary','') for s in res.get('sources', [])])
            fp = f"Rate 0-1: Is the following Answer derived only from the Context?\nContext:\n{ctx}\n\nAnswer:\n{res.get('answer') if isinstance(res.get('answer'), str) else str(res.get('answer'))}"
            rp = f"Rate 0-1: Does the Answer correctly answer the Question?\nQuestion:\n{item['question']}\n\nAnswer:\n{res.get('answer') if isinstance(res.get('answer'), str) else str(res.get('answer'))}"
            faith = 0.0; rel = 0.0
            try:
                faith = max(0.0, min(1.0, float(str(await self.llm.generate(fp, 'You return a single number between 0 and 1.')))))
            except Exception:
                faith = 0.0
            try:
                rel = max(0.0, min(1.0, float(str(await self.llm.generate(rp, 'You return a single number between 0 and 1.')))))
            except Exception:
                rel = 0.0
            report.append({ 'question': item['question'], 'expectedGroundTruth': item.get('expectedGroundTruth',''), 'faithfulness': faith, 'relevance': rel })
        return report
