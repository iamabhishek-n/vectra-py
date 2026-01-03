from typing import List, Dict, Any, Optional
from ..interfaces import VectorStore

class MilvusVectorStore(VectorStore):
    def __init__(self, config):
        self.config = config
        self.client = config.client_instance
        self.collection = config.table_name or 'rag_collection'

    async def add_documents(self, documents: List[Dict[str, Any]]):
        data = [{ 'vector': d['embedding'], 'content': d['content'], 'metadata': d['metadata'] } for d in documents]
        await self.client.insert(collection_name=self.collection, fields_data=data)

    async def similarity_search(self, vector: List[float], limit: int = 5, filter: Optional[Dict] = None) -> List[Dict[str, Any]]:
        expr = self._filter_to_expr(filter)
        if expr:
            try:
                res = await self.client.search(collection_name=self.collection, data=[vector], limit=limit, expr=expr)
            except TypeError:
                res = await self.client.search(collection_name=self.collection, data=[vector], limit=limit, filter=expr)
        else:
            res = await self.client.search(collection_name=self.collection, data=[vector], limit=limit)
        hits = res.get('results', []) if isinstance(res, dict) else res
        return [{ 'content': h.get('content', ''), 'metadata': h.get('metadata', {}), 'score': h.get('distance', 0.0) } for h in hits]

    async def hybrid_search(self, text: str, vector: List[float], limit: int = 5, filter: Optional[Dict] = None) -> List[Dict[str, Any]]:
        return await self.similarity_search(vector, limit, filter)

    def _filter_to_expr(self, filter: Optional[Dict[str, Any]]) -> str:
        if not filter:
            return ""
        parts: List[str] = []
        for k, v in filter.items():
            if isinstance(v, str):
                parts.append(f'metadata["{k}"] == "{v}"')
            elif isinstance(v, bool):
                parts.append(f'metadata["{k}"] == {str(v).lower()}')
            elif isinstance(v, (int, float)):
                parts.append(f'metadata["{k}"] == {v}')
        return " and ".join(parts)

    async def list_documents(self, filter: Optional[Dict[str, Any]] = None, limit: int = 100) -> List[Dict[str, Any]]:
        expr = self._filter_to_expr(filter)
        if not hasattr(self.client, "query"):
            raise NotImplementedError("Milvus client does not support query()")
        res = await self.client.query(
            collection_name=self.collection,
            expr=expr,
            output_fields=["content", "metadata"],
            limit=max(1, int(limit)),
        )
        out: List[Dict[str, Any]] = []
        for r in res or []:
            out.append({"content": r.get("content", ""), "metadata": r.get("metadata") or {}})
        return out

    async def delete_documents(self, filter: Dict[str, Any]) -> int:
        expr = self._filter_to_expr(filter)
        if not hasattr(self.client, "delete"):
            raise NotImplementedError("Milvus client does not support delete()")
        await self.client.delete(collection_name=self.collection, expr=expr)
        return 0

    async def update_documents(self, filter: Dict[str, Any], update_data: Dict[str, Any]) -> int:
        raise NotImplementedError("Milvus update_documents is not implemented")
