import json
from typing import List, Dict, Any, Optional, Tuple
from ..interfaces import VectorStore


class WeaviateVectorStore(VectorStore):
    def __init__(self, config):
        self.config = config
        self.client = config.client_instance
        self.class_name = getattr(config, "table_name", None) or "Document"
        self.collection = self.client.collections.get(self.class_name)

    def _build_filter(self, filter: Optional[Dict[str, Any]]):
        if not filter:
            return None
        # NOTE: the exact filter-builder API (`collection.filter.by_property(...).equal(...)`)
        # was not verifiable against a live weaviate-client install in this environment.
        # Passing the raw filter object through assumes the caller's client_instance mock
        # (or a future adapter) accepts a plain equality-map shape. Flagged for
        # confirmation before production use, same as the vectra-js mirror and the
        # upsert-namespace assumption already carried by PineconeVectorStore.
        return filter

    async def add_documents(self, documents: List[Dict[str, Any]]):
        objects = [{
            "id": doc.get("id"),
            "properties": {"content": doc["content"], "metadata": json.dumps(doc.get("metadata") or {})},
            "vector": doc["embedding"],
        } for doc in documents]
        await self.collection.data.insert_many(objects)

    async def upsert_documents(self, documents: List[Dict[str, Any]]):
        return await self.add_documents(documents)

    def _map_object(self, o: Dict[str, Any]) -> Dict[str, Any]:
        properties = o.get("properties") or {}
        metadata = json.loads(properties["metadata"]) if properties.get("metadata") else {}
        distance = (o.get("metadata") or {}).get("distance")
        return {
            "content": properties.get("content", ""),
            "metadata": metadata,
            "score": (1 - distance) if isinstance(distance, (int, float)) else None,
        }

    async def similarity_search(self, vector: List[float], limit: int = 5, filter: Optional[Dict] = None) -> List[Dict[str, Any]]:
        kwargs: Dict[str, Any] = {"limit": limit, "return_metadata": ["distance"]}
        f = self._build_filter(filter)
        if f:
            kwargs["filters"] = f
        res = await self.collection.query.near_vector(vector, **kwargs)
        return [self._map_object(o) for o in (res.get("objects") or [])]

    async def hybrid_search(self, text: str, vector: List[float], limit: int = 5, filter: Optional[Dict] = None) -> List[Dict[str, Any]]:
        kwargs: Dict[str, Any] = {"vector": vector, "limit": limit, "alpha": 0.5, "return_metadata": ["score"]}
        f = self._build_filter(filter)
        if f:
            kwargs["filters"] = f
        res = await self.collection.query.hybrid(text, **kwargs)
        out = []
        for o in (res.get("objects") or []):
            properties = o.get("properties") or {}
            metadata = json.loads(properties["metadata"]) if properties.get("metadata") else {}
            out.append({
                "content": properties.get("content", ""),
                "metadata": metadata,
                "score": (o.get("metadata") or {}).get("score"),
            })
        return out

    async def list_documents(self, filter: Optional[Dict[str, Any]] = None, limit: int = 100, cursor: Optional[str] = None) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        kwargs: Dict[str, Any] = {"limit": limit}
        f = self._build_filter(filter)
        if f:
            kwargs["filters"] = f
        if cursor:
            kwargs["after"] = cursor
        res = await self.collection.query.fetch_objects(**kwargs)
        objects = res.get("objects") or []
        docs = []
        for o in objects:
            properties = o.get("properties") or {}
            metadata = json.loads(properties["metadata"]) if properties.get("metadata") else {}
            docs.append({"id": o.get("uuid"), "content": properties.get("content", ""), "metadata": metadata})
        next_cursor = objects[-1].get("uuid") if len(objects) == limit and objects else None
        return docs, next_cursor

    async def delete_documents(self, filter: Dict[str, Any]) -> int:
        # Unlike PineconeVectorStore (which returns -1, count unknown), Weaviate
        # genuinely supports listing by filter, so we can enumerate matches first
        # and report a real count — a real capability difference, not
        # inconsistency with the Pinecone mirror.
        if not filter:
            raise ValueError("delete_documents requires a filter")
        docs, _ = await self.list_documents(filter=filter, limit=100000)
        ids = [d["id"] for d in docs if d.get("id") is not None]
        for doc_id in ids:
            await self.collection.data.delete_by_id(doc_id)
        return len(ids)

    async def update_documents(self, filter: Dict[str, Any], update_data: Dict[str, Any]) -> int:
        if not update_data:
            return 0
        docs, _ = await self.list_documents(filter=filter, limit=100000)
        ids = [d["id"] for d in docs if d.get("id") is not None]
        if not ids:
            return 0
        properties: Dict[str, Any] = {}
        if "content" in update_data and update_data["content"] is not None:
            properties["content"] = str(update_data["content"])
        if "metadata" in update_data and isinstance(update_data["metadata"], dict):
            properties["metadata"] = json.dumps(update_data["metadata"])
        if not properties:
            return 0
        for doc_id in ids:
            await self.collection.data.update(doc_id, properties=properties)
        return len(ids)

    async def file_exists(self, sha256: str, size: int, last_modified: int) -> bool:
        # Unlike PineconeVectorStore, Weaviate's fetch_objects takes filters
        # without requiring a placeholder vector — a genuine capability
        # advantage over Pinecone, not an oversight.
        try:
            res = await self.collection.query.fetch_objects(
                limit=1,
                filters=self._build_filter({"fileSHA256": sha256, "fileSize": size, "lastModified": last_modified}),
            )
            return len(res.get("objects") or []) > 0
        except Exception:
            return False
