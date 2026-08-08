import re
import json
import os
import aiohttp
from typing import List, Dict, Any, Union
from .config import RerankingConfig, RerankingProvider

class LLMReranker:
    def __init__(self, llm, config: RerankingConfig):
        self.llm = llm
        self.config = config

    async def rerank(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not documents:
            return []
        
        # Use window_size to limit documents for ranking
        docs_to_rank = documents[:self.config.window_size]
        
        doc_list = "\n".join([f"[{i+1}] {d['content'][:500]}" for i, d in enumerate(docs_to_rank)])
        prompt = f"""Identify the most relevant documents to the following query. 
Rank them from most relevant to least relevant by their IDs (e.g., [1], [2]).
Query: "{query}"

Documents:
{doc_list}

Return a VALID JSON array of indices (starting from 1) in order of relevance. 
Example result format: [3, 1, 2]
"""
        try:
            res = await self.llm.generate(prompt)
            match = re.search(r'\[[\d,\s]+\]', res)
            if match:
                indices = json.loads(match.group(0))
                ranked = []
                for idx in indices:
                    if isinstance(idx, int) and 1 <= idx <= len(docs_to_rank):
                        ranked.append(docs_to_rank[idx-1])
                
                # Add any missing docs from the original window at the end
                seen_ids = {id(d) for d in ranked}
                for d in docs_to_rank:
                    if id(d) not in seen_ids:
                        ranked.append(d)
                
                # Append docs outside the window
                ranked.extend(documents[self.config.window_size:])
                return ranked[:self.config.top_n]
        except Exception:
            pass
        
        return documents[:self.config.top_n]

class CrossEncoderReranker:
    """Dedicated reranker model (Cohere, Jina, or local Cross-Encoder)"""
    def __init__(self, config: RerankingConfig):
        self.config = config

    async def rerank(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not documents:
            return []

        docs_to_rank = documents[:self.config.window_size]

        try:
            if self.config.provider == RerankingProvider.COHERE:
                return await self._cohere_rerank(query, docs_to_rank)
            if self.config.provider == RerankingProvider.JINA:
                return await self._jina_rerank(query, docs_to_rank)
            if self.config.provider == RerankingProvider.CROSS_ENCODER:
                raise NotImplementedError(
                    "RerankingProvider.CROSS_ENCODER (local model) is not implemented in vectra-py. "
                    "Use RerankingProvider.COHERE, RerankingProvider.JINA, or RerankingProvider.LLM instead."
                )
            return documents[:self.config.top_n]
        except NotImplementedError:
            raise
        except Exception:
            return docs_to_rank[:self.config.top_n]

    async def _cohere_rerank(self, query: str, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        api_key = getattr(self.config, "api_key", None) or os.getenv("COHERE_API_KEY")
        if not api_key:
            return docs[:self.config.top_n]
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.cohere.com/v2/rerank",
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
                json={
                    "model": getattr(self.config, "model_name", None) or "rerank-v3.5",
                    "query": query,
                    "documents": [d["content"] for d in docs],
                    "top_n": min(self.config.top_n, len(docs)),
                },
                timeout=aiohttp.ClientTimeout(total=10),
            ) as res:
                if res.status != 200:
                    raise Exception(f"Cohere rerank API error: {res.status}")
                data = await res.json()
                return [docs[r["index"]] for r in data["results"]]

    async def _jina_rerank(self, query: str, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # Implemented in Task 2.
        return docs[:self.config.top_n]

def get_reranker(config: RerankingConfig, llm=None):
    if config.provider == RerankingProvider.LLM:
        return LLMReranker(llm, config)
    return CrossEncoderReranker(config)
