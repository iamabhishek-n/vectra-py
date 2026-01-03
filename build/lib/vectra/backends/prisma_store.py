import json
import uuid
from typing import List, Dict, Any, Optional
from ..interfaces import VectorStore
from ..config import DatabaseConfig

class PrismaVectorStore(VectorStore):
    def __init__(self, config: DatabaseConfig):
        self.config = config

    def _safe_ident(self, value: str) -> str:
        import re
        v = str(value or "")
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", v):
            raise ValueError(f"Unsafe SQL identifier: {v!r}")
        return v
    
    async def ensure_indexes(self):
        client = self.config.client_instance
        table = self._safe_ident(self.config.table_name or "Document")
        c_vec = self._safe_ident(self.config.column_map.get('vector', 'embedding'))
        c_content = self._safe_ident(self.config.column_map.get('content', 'content'))
        try:
            await client.execute_raw("CREATE EXTENSION IF NOT EXISTS vector")
            await client.execute_raw(f"""
                CREATE INDEX IF NOT EXISTS "{table}_embedding_ivfflat"
                ON "{table}" USING ivfflat ("{c_vec}" vector_cosine_ops)
                WITH (lists = 100);
            """)
            await client.execute_raw(f"""
                CREATE INDEX IF NOT EXISTS "{table}_content_fts_gin"
                ON "{table}" USING GIN (to_tsvector('english', "{c_content}"));
            """)
        except Exception:
            pass

    def normalize_vector(self, v: List[float]) -> List[float]:
        norm = sum(x * x for x in v) ** 0.5
        if norm == 0: return v
        return [x / norm for x in v]

    async def add_documents(self, documents: List[Dict[str, Any]]):
        client = self.config.client_instance
        table = self._safe_ident(self.config.table_name or "Document")
        # Assuming flexible column mapping, defaulting to standard names
        col_content = self._safe_ident(self.config.column_map.get('content', 'content'))
        col_meta = self._safe_ident(self.config.column_map.get('metadata', 'metadata'))
        col_vec = self._safe_ident(self.config.column_map.get('vector', 'embedding'))

        for doc in documents:
            vec_str = json.dumps(self.normalize_vector(doc['embedding']))
            query = f"""
            INSERT INTO "{table}" ("id", "{col_content}", "{col_meta}", "{col_vec}", "createdAt")
            VALUES ($1::uuid, $2, $3::jsonb, $4::vector, NOW())
            """
            await client.execute_raw(
                query,
                str(uuid.uuid4()),
                doc['content'],
                json.dumps(doc['metadata']),
                vec_str
            )
    
    async def file_exists(self, sha256: str, size: int, last_modified: int) -> bool:
        client = self.config.client_instance
        table = self._safe_ident(self.config.table_name or "Document")
        col_meta = self._safe_ident(self.config.column_map.get('metadata', 'metadata'))
        payload = json.dumps({"fileSHA256": sha256, "fileSize": size, "lastModified": last_modified})
        q = f'SELECT 1 FROM "{table}" WHERE "{col_meta}" @> $1::jsonb LIMIT 1'
        try:
            res = await client.query_raw(q, payload)
            return len(res) > 0
        except Exception:
            return False

    async def similarity_search(self, vector: List[float], limit: int = 5, filter: Optional[Dict] = None) -> List[Dict[str, Any]]:
        client = self.config.client_instance
        table = self._safe_ident(self.config.table_name or "Document")
        col_content = self._safe_ident(self.config.column_map.get('content', 'content'))
        col_meta = self._safe_ident(self.config.column_map.get('metadata', 'metadata'))
        col_vec = self._safe_ident(self.config.column_map.get('vector', 'embedding'))
        
        vec_str = json.dumps(self.normalize_vector(vector))
        where_clause = ""
        params = [vec_str]
        
        if filter:
            where_clause = f'WHERE "{col_meta}" @> $2::jsonb'
            params.append(json.dumps(filter))
            
        query = f"""
        SELECT "{col_content}" as content, "{col_meta}" as metadata, 1 - ("{col_vec}" <=> $1::vector) as score
        FROM "{table}"
        {where_clause}
        ORDER BY score DESC
        LIMIT {limit}
        """
        
        if len(params) == 1: res = await client.query_raw(query, params[0])
        else: res = await client.query_raw(query, params[0], params[1])
            
        return [
            {
                'content': r['content'], 
                'metadata': r['metadata'] if isinstance(r['metadata'], dict) else json.loads(r['metadata']), 
                'score': float(r['score'])
            } for r in res
        ]
    
    async def hybrid_search(self, text: str, vector: List[float], limit: int = 5, filter: Optional[Dict] = None) -> List[Dict[str, Any]]:
        semantic = await self.similarity_search(vector, limit * 2, filter)
        client = self.config.client_instance
        table = self._safe_ident(self.config.table_name or "Document")
        c_content = self._safe_ident(self.config.column_map.get('content', 'content'))
        c_metadata = self._safe_ident(self.config.column_map.get('metadata', 'metadata'))
        where = f"to_tsvector('simple', \"{c_content}\") @@ plainto_tsquery($1)"
        params: List[Any] = [text]
        if filter:
            where = where + f' AND "{c_metadata}" @> $2::jsonb'
            params.append(json.dumps(filter))
        q = f"SELECT \"{c_content}\" as content, \"{c_metadata}\" as metadata FROM \"{table}\" WHERE {where} LIMIT {limit * 2}"
        lexical = []
        try:
            res = await client.query_raw(q, *params)
            lexical = [{ 'content': r['content'], 'metadata': r.get('metadata'), 'score': 1.0 } for r in res]
        except Exception:
            lexical = []
        combined: Dict[str, Dict[str, Any]] = {}
        def add(lst: List[Dict[str, Any]], weight: float = 1.0):
            for idx, doc in enumerate(lst):
                key = doc['content']
                score = (1.0 / (60 + idx + 1)) * weight
                if key not in combined:
                    combined[key] = { **doc, 'score': 0.0 }
                combined[key]['score'] += score
        add(semantic, 1.0)
        add(lexical, 1.0)
        return sorted(combined.values(), key=lambda d: d['score'], reverse=True)[:limit]

    async def list_documents(self, filter: Optional[Dict[str, Any]] = None, limit: int = 100) -> List[Dict[str, Any]]:
        client = self.config.client_instance
        table = self._safe_ident(self.config.table_name or "Document")
        c_content = self._safe_ident(self.config.column_map.get("content", "content"))
        c_metadata = self._safe_ident(self.config.column_map.get("metadata", "metadata"))
        limit_int = max(1, int(limit))
        where = ""
        params: List[Any] = []
        if filter:
            where = f'WHERE "{c_metadata}" @> $1::jsonb'
            params.append(json.dumps(filter))
        q = f"""
        SELECT "id" as id, "{c_content}" as content, "{c_metadata}" as metadata
        FROM "{table}"
        {where}
        ORDER BY "createdAt" DESC
        LIMIT {limit_int}
        """
        if params:
            rows = await client.query_raw(q, params[0])
        else:
            rows = await client.query_raw(q)
        out: List[Dict[str, Any]] = []
        for r in rows:
            md = r.get("metadata")
            out.append(
                {
                    "id": r.get("id"),
                    "content": r.get("content"),
                    "metadata": md if isinstance(md, dict) else json.loads(md) if isinstance(md, str) else md,
                }
            )
        return out

    async def delete_documents(self, filter: Dict[str, Any]) -> int:
        client = self.config.client_instance
        table = self._safe_ident(self.config.table_name or "Document")
        c_metadata = self._safe_ident(self.config.column_map.get("metadata", "metadata"))
        q = f'DELETE FROM "{table}" WHERE "{c_metadata}" @> $1::jsonb RETURNING 1'
        rows = await client.query_raw(q, json.dumps(filter))
        return len(rows)

    async def update_documents(self, filter: Dict[str, Any], update_data: Dict[str, Any]) -> int:
        client = self.config.client_instance
        table = self._safe_ident(self.config.table_name or "Document")
        c_content = self._safe_ident(self.config.column_map.get("content", "content"))
        c_metadata = self._safe_ident(self.config.column_map.get("metadata", "metadata"))

        sets: List[str] = []
        params: List[Any] = []
        idx = 1

        if "content" in update_data and update_data["content"] is not None:
            sets.append(f'"{c_content}" = ${idx}')
            params.append(str(update_data["content"]))
            idx += 1

        if "metadata" in update_data and isinstance(update_data["metadata"], dict):
            sets.append(f'"{c_metadata}" = COALESCE("{c_metadata}", \'{{}}\'::jsonb) || ${idx}::jsonb')
            params.append(json.dumps(update_data["metadata"]))
            idx += 1

        if not sets:
            return 0

        where = f'"{c_metadata}" @> ${idx}::jsonb'
        params.append(json.dumps(filter))

        q = f'UPDATE "{table}" SET {", ".join(sets)} WHERE {where} RETURNING 1'
        rows = await client.query_raw(q, *params)
        return len(rows)
