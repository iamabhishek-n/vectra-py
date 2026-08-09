import re
from typing import List, Dict, Any, Optional, Tuple
from ..interfaces import VectorStore


class PineconeVectorStore(VectorStore):
    def __init__(self, config):
        self.config = config
        self.client = config.client_instance
        self.namespace = getattr(config, "table_name", None) or None

    async def add_documents(self, documents: List[Dict[str, Any]]):
        vectors = [{
            "id": doc["id"],
            "values": doc["embedding"],
            "metadata": {**doc.get("metadata", {}), "content": doc["content"]},
        } for doc in documents]
        kwargs = {"namespace": self.namespace} if self.namespace else {}
        await self.client.upsert(vectors, **kwargs)

    async def upsert_documents(self, documents: List[Dict[str, Any]]):
        return await self.add_documents(documents)

    async def similarity_search(self, vector: List[float], limit: int = 5, filter: Optional[Dict] = None) -> List[Dict[str, Any]]:
        query_kwargs: Dict[str, Any] = {"vector": vector, "top_k": limit, "include_metadata": True}
        if filter:
            query_kwargs["filter"] = filter
        if self.namespace:
            query_kwargs["namespace"] = self.namespace
        res = await self.client.query(**query_kwargs)
        matches = (res or {}).get("matches", [])
        results = []
        for m in matches:
            metadata = dict(m.get("metadata") or {})
            content = metadata.pop("content", "")
            results.append({"content": content, "metadata": metadata, "score": m.get("score")})
        return results

    def _lexical_overlap(self, query: str, content: str) -> float:
        def tokenize(s):
            return set(t for t in re.findall(r"[a-zA-Z0-9]+", (s or "").lower()) if len(t) > 2)
        query_tokens = tokenize(query)
        if not query_tokens:
            return 0.0
        content_tokens = tokenize(content)
        matches = len(query_tokens & content_tokens)
        return matches / len(query_tokens)

    async def hybrid_search(self, text: str, vector: List[float], limit: int = 5, filter: Optional[Dict] = None) -> List[Dict[str, Any]]:
        pool = await self.similarity_search(vector, max(limit * 4, 20), filter)
        if not pool:
            return []
        with_lexical = [{**d, "_lexical": self._lexical_overlap(text, d["content"])} for d in pool]
        semantic_ranked = sorted(with_lexical, key=lambda d: d["score"], reverse=True)
        lexical_ranked = sorted(with_lexical, key=lambda d: d["_lexical"], reverse=True)
        rrf_scores: Dict[str, float] = {}
        for ranked in (semantic_ranked, lexical_ranked):
            for idx, d in enumerate(ranked):
                key = d["content"]
                rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (60 + idx + 1)
        seen: Dict[str, Dict[str, Any]] = {}
        for d in with_lexical:
            if d["content"] not in seen:
                seen[d["content"]] = d
        ordered = sorted(seen.values(), key=lambda d: rrf_scores.get(d["content"], 0.0), reverse=True)
        return [{k: v for k, v in d.items() if k != "_lexical"} for d in ordered[:limit]]

    async def list_documents(self, filter: Optional[Dict[str, Any]] = None, limit: int = 100, cursor: Optional[str] = None) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        raise NotImplementedError(
            "list_documents is not supported for Pinecone — the API has no arbitrary "
            "listing/scroll endpoint. Use file_exists or similarity_search with a broad query instead."
        )

    async def update_documents(self, filter: Dict[str, Any], update_data: Dict[str, Any]) -> int:
        raise NotImplementedError(
            "update_documents is not supported for Pinecone — updating by filter requires "
            "enumerating matching vectors first, and Pinecone has no listing/scroll endpoint."
        )

    async def delete_documents(self, filter: Dict[str, Any]) -> int:
        # Pinecone's delete-by-filter is server-side and native (unlike list/update,
        # which would need enumeration Pinecone can't do), but the API doesn't report
        # how many vectors matched. Returning an exact count here isn't possible
        # without a listing capability the SDK doesn't have. -1 is a deliberate
        # sentinel for "delete accepted, count unknown" — never silently 0, which
        # would misreport a real deletion as a no-op.
        if not filter:
            raise ValueError("delete_documents requires a filter")
        kwargs = {"namespace": self.namespace} if self.namespace else {}
        await self.client.delete(filter=filter, **kwargs)
        return -1

    async def file_exists(self, sha256: str, size: int, last_modified: int) -> bool:
        # Pinecone's query API requires a vector even for a pure metadata-filter
        # existence check — there's no listing/count-by-filter endpoint. A dummy
        # zero vector works for filter-only matching but MUST match the real
        # index's configured dimension, which this class doesn't otherwise know.
        # config.dimensions is used if the caller supplied it; otherwise this
        # will likely error against a real Pinecone index with a different
        # dimension — a real, documented limitation of this backend, not a bug
        # to silently paper over.
        dim = getattr(self.config, "dimensions", None) or 1536
        try:
            kwargs: Dict[str, Any] = {
                "vector": [0.0] * dim,
                "top_k": 1,
                "filter": {"fileSHA256": sha256, "fileSize": size, "lastModified": last_modified},
                "include_metadata": False,
            }
            if self.namespace:
                kwargs["namespace"] = self.namespace
            res = await self.client.query(**kwargs)
            matches = (res or {}).get("matches", [])
            return len(matches) > 0
        except Exception:
            return False
