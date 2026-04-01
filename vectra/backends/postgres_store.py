import json
import logging
import uuid
from typing import List, Dict, Any, Optional, Tuple
import asyncio
from ..interfaces import VectorStore

def to_db_vector(vector: List[float]) -> str:
    return f"[{','.join(map(str, vector))}]"

class PostgresVectorStore(VectorStore):
    def __init__(self, config: Any):
        self.config = config
        self.client = config.client_instance
        self.table_name = getattr(config, 'table_name', 'document')
        self.column_map = getattr(config, 'column_map', {})
        self.c_content = self.column_map.get('content', 'content')
        self.c_meta = self.column_map.get('metadata', 'metadata')
        self.c_vector = self.column_map.get('vector', 'vector')
        
    def _get_connection(self):
        """Helper to support both single asyncpg connection and Pool."""
        if hasattr(self.client, 'acquire'):
            return self.client.acquire()
        
        class DummyContext:
            def __init__(self, conn): self.conn = conn
            async def __aenter__(self): return self.conn
            async def __aexit__(self, *args): pass
        return DummyContext(self.client)

    async def ensure_indexes(self, dimensions: int = 1536):
        async with self._get_connection() as conn:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            
            # 1. Ensure Table Exists
            create_table_sql = f"""
            CREATE TABLE IF NOT EXISTS "{self.table_name}" (
                "id" TEXT PRIMARY KEY,
                "{self.c_content}" TEXT,
                "{self.c_meta}" JSONB,
                "{self.c_vector}" vector({dimensions}),
                "createdAt" TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            );
            """
            await conn.execute(create_table_sql)

            # 2. Schema Check & Migration
            try:
                cols_sql = """
                    SELECT column_name, data_type, udt_name 
                    FROM information_schema.columns 
                    WHERE table_name = $1
                """
                rows = await conn.fetch(cols_sql, self.table_name)
                existing_cols = {r['column_name']: r for r in rows}
                
                if self.c_vector in existing_cols:
                    col_info = existing_cols[self.c_vector]
                    if col_info['udt_name'] != 'vector' and 'array' in (col_info['data_type'] or '').lower():
                         raise ValueError(f"Postgres schema mismatch: '{self.c_vector}' is array. Use vector({dimensions}).")

                alter_stmts = []
                if self.c_content not in existing_cols: alter_stmts.append(f'ADD COLUMN "{self.c_content}" TEXT')
                if self.c_meta not in existing_cols: alter_stmts.append(f'ADD COLUMN "{self.c_meta}" JSONB')
                if self.c_vector not in existing_cols: alter_stmts.append(f'ADD COLUMN "{self.c_vector}" vector({dimensions})')
                if "createdAt" not in existing_cols: alter_stmts.append('ADD COLUMN "createdAt" TIMESTAMP WITH TIME ZONE DEFAULT NOW()')
                
                if alter_stmts:
                    await conn.execute(f'ALTER TABLE "{self.table_name}" {", ".join(alter_stmts)}')
            except Exception as e:
                if "schema mismatch" in str(e): raise e
                logging.warning(f"Schema check failed: {e}")

            # 3. Indexes
            index_sql = f'CREATE INDEX IF NOT EXISTS "{self.table_name}_vec_idx" ON "{self.table_name}" USING hnsw ("{self.c_vector}" vector_cosine_ops);'
            try:
                await conn.execute(index_sql)
            except Exception:
                try:
                    await conn.execute(f'CREATE INDEX IF NOT EXISTS "{self.table_name}_vec_idx" ON "{self.table_name}" USING ivfflat ("{self.c_vector}" vector_cosine_ops);')
                except Exception: pass

    async def add_documents(self, documents: List[Dict[str, Any]]):
        sql = f"""
        INSERT INTO "{self.table_name}" ("id", "{self.c_content}", "{self.c_meta}", "{self.c_vector}", "createdAt")
        VALUES ($1, $2, $3, $4, NOW())
        ON CONFLICT ("id") DO NOTHING;
        """
        data = []
        for doc in documents:
            doc_id = doc.get('id') or str(uuid.uuid4())
            content = doc.get('content')
            meta = json.dumps(doc.get('metadata', {}))
            vec = to_db_vector(doc.get('embedding', []))
            data.append((doc_id, content, meta, vec))
            
        async with self._get_connection() as conn:
            await conn.executemany(sql, data)

    async def upsert_documents(self, documents: List[Dict[str, Any]]):
        sql = f"""
        INSERT INTO "{self.table_name}" ("id", "{self.c_content}", "{self.c_meta}", "{self.c_vector}", "createdAt")
        VALUES ($1, $2, $3, $4, NOW())
        ON CONFLICT ("id") DO UPDATE SET
            "{self.c_content}" = EXCLUDED."{self.c_content}",
            "{self.c_meta}" = EXCLUDED."{self.c_meta}",
            "{self.c_vector}" = EXCLUDED."{self.c_vector}",
            "createdAt" = NOW();
        """
        data = []
        for doc in documents:
            doc_id = doc.get('id') or str(uuid.uuid4())
            content = doc.get('content')
            meta = json.dumps(doc.get('metadata', {}))
            vec = to_db_vector(doc.get('embedding', []))
            data.append((doc_id, content, meta, vec))
            
        async with self._get_connection() as conn:
            await conn.executemany(sql, data)

    async def file_exists(self, sha256: str, size: int, last_modified: int) -> bool:
        meta_filter = json.dumps({ 'fileSHA256': sha256, 'fileSize': size, 'lastModified': last_modified })
        sql = f'SELECT 1 FROM "{self.table_name}" WHERE "{self.c_meta}" @> $1::jsonb LIMIT 1'
        try:
            async with self._get_connection() as conn:
                rows = await conn.fetch(sql, meta_filter)
                return len(rows) > 0
        except Exception:
            return False

    async def similarity_search(self, vector: List[float], limit: int = 5, filter: Optional[Dict] = None) -> List[Dict[str, Any]]:
        vec_str = to_db_vector(vector)
        where_clause = ""
        params = [vec_str, limit]
        
        if filter:
            where_clause = f'WHERE "{self.c_meta}" @> $3::jsonb'
            params.append(json.dumps(filter))

        sql = f"""
        SELECT "id", "{self.c_content}" as content, "{self.c_meta}" as metadata, 
               ("{self.c_vector}" <=> $1) as distance
        FROM "{self.table_name}"
        {where_clause}
        ORDER BY distance ASC
        LIMIT $2
        """
        
        async with self._get_connection() as conn:
            rows = await conn.fetch(sql, *params)
        
        results = []
        for row in rows:
            results.append({
                'id': row['id'],
                'content': row['content'],
                'metadata': json.loads(row['metadata']) if isinstance(row['metadata'], str) else row['metadata'],
                'score': 1 - row['distance']
            })
        return results

    async def hybrid_search(self, text: str, vector: List[float], limit: int = 5, filter: Optional[Dict] = None) -> List[Dict[str, Any]]:
        # Fallback to similarity search for now
        return await self.similarity_search(vector, limit, filter)

    async def delete_documents(self, filter: Dict[str, Any]) -> int:
        sql = f'DELETE FROM "{self.table_name}" WHERE "{self.c_meta}" @> $1::jsonb'
        async with self._get_connection() as conn:
            res = await conn.execute(sql, json.dumps(filter))
            return int(res.split(' ')[1]) if res else 0
        
    async def update_documents(self, filter: Dict[str, Any], update_data: Dict[str, Any]) -> int:
        raise NotImplementedError

    async def list_documents(self, filter: Optional[Dict[str, Any]] = None, limit: int = 100, cursor: Optional[str] = None) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        where_parts = []
        params = []
        if filter:
            where_parts.append(f'"{self.c_meta}" @> ${len(params)+1}::jsonb')
            params.append(json.dumps(filter))
        if cursor:
            where_parts.append(f'"id" > ${len(params)+1}')
            params.append(cursor)
            
        where_clause = "WHERE " + " AND ".join(where_parts) if where_parts else ""
        sql = f'SELECT "id", "{self.c_content}" as content, "{self.c_meta}" as metadata FROM "{self.table_name}" {where_clause} ORDER BY "id" ASC LIMIT ${len(params)+1}'
        params.append(limit)
        
        async with self._get_connection() as conn:
            rows = await conn.fetch(sql, *params)
            
        docs = [{
            'id': r['id'],
            'content': r['content'],
            'metadata': json.loads(r['metadata']) if isinstance(r['metadata'], str) else r['metadata']
        } for r in rows]
        
        next_cursor = docs[-1]['id'] if len(docs) == limit else None
        return docs, next_cursor
