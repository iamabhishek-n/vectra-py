from typing import List, Dict, Any, Optional, Tuple
from ..interfaces import VectorStore

class QdrantVectorStore(VectorStore):
    def __init__(self, config):
        self.config = config
        self.client = config.client_instance
        self.collection = config.table_name or 'rag_collection'

    async def add_documents(self, documents: List[Dict[str, Any]]):
        points = [{
            'id': i,
            'vector': doc['embedding'],
            'payload': {'content': doc['content'], 'metadata': doc['metadata']}
        } for i, doc in enumerate(documents)]
        await self.client.upsert(self.collection, points=points)

    async def upsert_documents(self, documents: List[Dict[str, Any]]):
        return await self.add_documents(documents)

    def _normalize_filter(self, filter: Optional[Dict[str, Any]]):
        if not filter:
            return None
        if not isinstance(filter, dict):
            return filter
        if any(k in filter for k in ("must", "should", "must_not")):
            return filter
        must = []
        for k, v in filter.items():
            if isinstance(v, (str, int, float, bool)):
                must.append({"key": f"metadata.{k}", "match": {"value": v}})
        return {"must": must} if must else None

    async def similarity_search(self, vector: List[float], limit: int = 5, filter: Optional[Dict] = None) -> List[Dict[str, Any]]:
        q_filter = self._normalize_filter(filter)
        res = await self.client.search(self.collection, vector=vector, limit=limit, filter=q_filter)
        return [{'content': r['payload']['content'], 'metadata': r['payload'].get('metadata'), 'score': r.get('score', 1.0)} for r in res]

    async def hybrid_search(self, text: str, vector: List[float], limit: int = 5, filter: Optional[Dict] = None) -> List[Dict[str, Any]]:
        return await self.similarity_search(vector, limit, filter)

    async def file_exists(self, sha256: str, size: int, last_modified: int) -> bool:
        q_filter = self._normalize_filter({'fileSHA256': sha256, 'fileSize': size, 'lastModified': last_modified})
        try:
            res = await self.client.scroll(
                self.collection,
                scroll_filter=q_filter,
                limit=1,
                with_payload=False,
                with_vectors=False
            )
            points = res[0] # res is (points, next_page_offset)
            return len(points) > 0
        except TypeError:
             # Older client
            try:
                res = await self.client.scroll(
                    self.collection,
                    filter=q_filter,
                    limit=1,
                    with_payload=False,
                    with_vectors=False
                )
                points = res[0]
                return len(points) > 0
            except Exception:
                return False
        except Exception:
            return False

    async def _scroll_points(self, filter: Optional[Dict[str, Any]], limit: int, offset=None):
        if hasattr(self.client, "scroll"):
            q_filter = self._normalize_filter(filter)
            try:
                return await self.client.scroll(
                    self.collection,
                    scroll_filter=q_filter,
                    limit=limit,
                    with_payload=True,
                    with_vectors=False,
                    offset=offset,
                )
            except TypeError:
                return await self.client.scroll(
                    self.collection,
                    filter=q_filter,
                    limit=limit,
                    with_payload=True,
                    with_vectors=False,
                    offset=offset,
                )
        raise NotImplementedError("Qdrant client does not support scroll()")

    async def list_documents(self, filter: Optional[Dict[str, Any]] = None, limit: int = 100, cursor: Optional[str] = None) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        limit_int = max(1, int(limit))
        res = await self._scroll_points(filter, limit_int, offset=cursor)
        if isinstance(res, tuple) and len(res) == 2:
            points, next_cursor = res
        else:
            points = res
            next_cursor = None
        
        out: List[Dict[str, Any]] = []
        for p in points or []:
            payload = p.get("payload") or {}
            out.append(
                {
                    "id": p.get("id"),
                    "content": payload.get("content", ""),
                    "metadata": payload.get("metadata") or {},
                }
            )
        return out, next_cursor

    async def delete_documents(self, filter: Dict[str, Any]) -> int:
        deleted = 0
        docs = await self.list_documents(filter=filter, limit=100000)
        ids = [d.get("id") for d in docs if d.get("id") is not None]
        if not ids:
            return 0
        if hasattr(self.client, "delete"):
            try:
                await self.client.delete(self.collection, points_selector={"points": ids})
            except TypeError:
                await self.client.delete(self.collection, ids=ids)
        deleted = len(ids)
        return deleted

    async def update_documents(self, filter: Dict[str, Any], update_data: Dict[str, Any]) -> int:
        if not update_data:
            return 0
        docs = await self.list_documents(filter=filter, limit=100000)
        ids = [d.get("id") for d in docs if d.get("id") is not None]
        if not ids:
            return 0
        payload: Dict[str, Any] = {}
        if "content" in update_data and update_data["content"] is not None:
            payload["content"] = str(update_data["content"])
        if "metadata" in update_data and isinstance(update_data["metadata"], dict):
            payload["metadata"] = update_data["metadata"]
        if not payload:
            return 0
        if hasattr(self.client, "set_payload"):
            try:
                await self.client.set_payload(self.collection, payload=payload, points=ids)
            except TypeError:
                await self.client.set_payload(collection_name=self.collection, payload=payload, points=ids)
            return len(ids)
        if hasattr(self.client, "upsert"):
            for d in docs:
                pid = d.get("id")
                if pid is None:
                    continue
                merged_md = d.get("metadata") or {}
                if "metadata" in payload and isinstance(payload["metadata"], dict):
                    merged_md = payload["metadata"]
                merged_content = payload.get("content", d.get("content", ""))
                await self.client.upsert(
                    self.collection,
                    points=[{"id": pid, "payload": {"content": merged_content, "metadata": merged_md}}],
                )
            return len(ids)
        raise NotImplementedError("Qdrant client does not support payload update")
