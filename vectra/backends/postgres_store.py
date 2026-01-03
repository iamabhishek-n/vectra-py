import json
import logging
import uuid
from typing import List, Dict, Any, Optional
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
        
    async def ensure_indexes(self, dimensions: int = 1536):
        await self.client.execute("CREATE EXTENSION IF NOT EXISTS vector")
        
        # Default dimension 1536 (OpenAI/Gemini-ish), but user should manage schema for other dims
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS "{self.table_name}" (
            "id" TEXT PRIMARY KEY,
            "{self.c_content}" TEXT,
            "{self.c_meta}" JSONB,
            "{self.c_vector}" vector({dimensions}),
            "createdAt" TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );
        """
        await self.client.execute(create_table_sql)
        
        index_sql = f"""
        CREATE INDEX IF NOT EXISTS "{self.table_name}_vec_idx" 
        ON "{self.table_name}" USING hnsw ("{self.c_vector}" vector_cosine_ops);
        """
        await self.client.execute(index_sql)

    async def add_documents(self, documents: List[Dict[str, Any]]):
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
            
        await self.client.executemany(sql, data)

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
        
        rows = await self.client.fetch(sql, *params)
        
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
        res = await self.client.execute(sql, json.dumps(filter))
        return int(res.split(' ')[1]) if res else 0
        
    async def update_documents(self, filter: Dict[str, Any], update_data: Dict[str, Any]) -> int:
        raise NotImplementedError

    async def list_documents(self, filter: Optional[Dict[str, Any]] = None, limit: int = 100) -> List[Dict[str, Any]]:
        where_clause = ""
        params = [limit]
        if filter:
            where_clause = f'WHERE "{self.c_meta}" @> $2::jsonb'
            params.append(json.dumps(filter))
            
        sql = f"""
        SELECT "id", "{self.c_content}" as content, "{self.c_meta}" as metadata
        FROM "{self.table_name}"
        {where_clause}
        LIMIT $1
        """
        rows = await self.client.fetch(sql, *params)
        return [{
            'id': r['id'],
            'content': r['content'],
            'metadata': json.loads(r['metadata']) if isinstance(r['metadata'], str) else r['metadata']
        } for r in rows]
