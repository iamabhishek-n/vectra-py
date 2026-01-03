import re
from typing import List, Dict, Any
from .config import RerankingConfig

class LLMReranker:
    def __init__(self, llm, config: RerankingConfig):
        self.llm = llm
        self.config = config

    async def rerank(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not documents:
            return []
        
        scored = []
        for doc in documents:
            score = await self.score_document(query, doc['content'])
            doc_copy = doc.copy()
            doc_copy['score'] = score
            scored.append(doc_copy)
            
        scored.sort(key=lambda x: x['score'], reverse=True)
        return scored[:self.config.top_n]

    async def score_document(self, query: str, content: str) -> int:
        prompt = f'Analyze relevance (0-10) of document to query. Return ONLY integer.\nQuery: "{query}"\nDoc: "{content[:1000]}..."'
        try:
            res = await self.llm.generate(prompt)
            match = re.search(r'\d+', res)
            return int(match.group(0)) if match else 0
        except Exception:
            return 0
