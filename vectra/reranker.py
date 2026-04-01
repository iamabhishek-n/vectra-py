import re
import json
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
        
        # This is a placeholder for specific provider implementations (Cohere, Jina, etc.)
        # In a real implementation, we would call the respective API here.
        # For now, we'll implement a generic pattern.
        
        if self.config.provider == RerankingProvider.COHERE:
            # Placeholder for Cohere API call
            return await self._mock_api_rerank(query, docs_to_rank)
        
        # Default fallback
        return documents[:self.config.top_n]

    async def _mock_api_rerank(self, query, docs):
        # Implementation of API-based reranking would go here
        return docs[:self.config.top_n]

def get_reranker(config: RerankingConfig, llm=None):
    if config.provider == RerankingProvider.LLM:
        return LLMReranker(llm, config)
    return CrossEncoderReranker(config)
