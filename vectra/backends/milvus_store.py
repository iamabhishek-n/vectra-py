from typing import List, Dict, Any, Optional, Tuple
import re
from ..interfaces import VectorStore

_SAFE_FILTER_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _escape_milvus_string(value: str) -> str:
    return value.replace('\\', '\\\\').replace('"', '\\"')

class MilvusVectorStore(VectorStore):
    def __init__(self, config):
        self.config = config
        self.client = config.client_instance
        self.collection = config.table_name or 'rag_collection'

    async def add_documents(self, documents: List[Dict[str, Any]]):
        data = [{ 'vector': d['embedding'], 'content': d['content'], 'metadata': d['metadata'] } for d in documents]
        await self.client.insert(collection_name=self.collection, fields_data=data)

    async def upsert_documents(self, documents: List[Dict[str, Any]]):
        data = [{ 'vector': d['embedding'], 'content': d['content'], 'metadata': d['metadata'] } for d in documents]
        # Try upsert if available, else insert
        if hasattr(self.client, 'upsert'):
             await self.client.upsert(collection_name=self.collection, fields_data=data)
        else:
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

    async def file_exists(self, sha256: str, size: int, last_modified: int) -> bool:
        expr = self._filter_to_expr({'fileSHA256': sha256, 'fileSize': size, 'lastModified': last_modified})
        try:
            res = await self.client.query(
                collection_name=self.collection,
                expr=expr,
                output_fields=["count(*)"], # Just check existence
                limit=1
            )
            return len(res) > 0
        except Exception:
            # Fallback if count(*) not supported or other error
            try:
                res = await self.client.query(
                    collection_name=self.collection,
                    expr=expr,
                    output_fields=["metadata"], 
                    limit=1
                )
                return len(res) > 0
            except Exception:
                return False

    def _filter_to_expr(self, filter: Optional[Dict[str, Any]]) -> str:
        if not filter:
            return ""
        parts: List[str] = []
        for k, v in filter.items():
            if not _SAFE_FILTER_KEY_RE.fullmatch(str(k)):
                raise ValueError(f"Unsafe filter key for Milvus expression: {k!r}")
            if isinstance(v, str):
                parts.append(f'metadata["{k}"] == "{_escape_milvus_string(v)}"')
            elif isinstance(v, bool):
                parts.append(f'metadata["{k}"] == {str(v).lower()}')
            elif isinstance(v, (int, float)):
                parts.append(f'metadata["{k}"] == {v}')
            else:
                raise ValueError(f"Unsupported filter value type for Milvus expression: {k!r}={type(v).__name__}")
        return " and ".join(parts)

    async def list_documents(self, filter: Optional[Dict[str, Any]] = None, limit: int = 100, cursor: Optional[str] = None) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        expr = self._filter_to_expr(filter)
        if not hasattr(self.client, "query"):
            raise NotImplementedError("Milvus client does not support query()")
        
        limit_int = max(1, int(limit))
        offset = int(cursor) if cursor and cursor.isdigit() else 0
        
        res = await self.client.query(
            collection_name=self.collection,
            expr=expr,
            output_fields=["content", "metadata", "id"],
            limit=limit_int,
            offset=offset
        )
        out: List[Dict[str, Any]] = []
        for r in res or []:
            out.append({"id": r.get("id"), "content": r.get("content", ""), "metadata": r.get("metadata") or {}})
        
        next_cursor = str(offset + len(out)) if len(out) == limit_int else None
        return out, next_cursor

    async def delete_documents(self, filter: Dict[str, Any]) -> int:
        expr = self._filter_to_expr(filter)
        if not hasattr(self.client, "delete"):
            raise NotImplementedError("Milvus client does not support delete()")
        result = await self.client.delete(collection_name=self.collection, expr=expr)
        if isinstance(result, dict):
            return result.get("delete_count", 0)
        return getattr(result, "delete_count", 0)

    async def update_documents(self, filter: Dict[str, Any], update_data: Dict[str, Any]) -> int:
        if not update_data:
            return 0
        expr = self._filter_to_expr(filter)
        if not hasattr(self.client, "query"):
            raise NotImplementedError("Milvus client does not support query()")
        docs = await self.client.query(
            collection_name=self.collection,
            expr=expr,
            output_fields=["id", "vector", "content", "metadata"],
            limit=100000,
        )
        docs = docs or []
        if not docs:
            return 0
        new_content = update_data.get("content")
        update_meta = update_data.get("metadata")
        data = []
        for d in docs:
            vector = d.get("vector")
            if vector is None:
                # Skip documents missing their vector rather than upserting a
                # null embedding over a working one.
                continue
            metadata = d.get("metadata") or {}
            if isinstance(update_meta, dict):
                metadata = {**metadata, **update_meta}
            data.append({
                "id": d.get("id"),
                "vector": vector,
                "content": new_content if isinstance(new_content, str) else d.get("content", ""),
                "metadata": metadata,
            })
        if not data:
            return 0
        if hasattr(self.client, "upsert"):
            await self.client.upsert(collection_name=self.collection, fields_data=data)
        else:
            await self.client.insert(collection_name=self.collection, fields_data=data)
        return len(data)
