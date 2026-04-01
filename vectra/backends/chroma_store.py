import uuid
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from ..interfaces import VectorStore
from ..config import DatabaseConfig

class ChromaVectorStore(VectorStore):
    def __init__(self, config: DatabaseConfig):
        # Expecting 'client_instance' to be a chromadb.Client or PersistentClient
        self.client = config.client_instance
        if self.client is None:
            import chromadb
            path = getattr(config, 'path', './chroma_db')
            self.client = chromadb.PersistentClient(path=path)
            
        self.collection_name = config.table_name or getattr(config, 'collection_name', "rag_collection")
        self.collection = self.client.get_or_create_collection(name=self.collection_name)

    async def add_documents(self, documents: List[Dict[str, Any]]):
        ids = [str(uuid.uuid4()) for _ in documents]
        embeddings = [doc['embedding'] for doc in documents]
        metadatas = [doc['metadata'] for doc in documents]
        documents_text = [doc['content'] for doc in documents]
        
        await asyncio.to_thread(
            self.collection.add,
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents_text
        )

    async def upsert_documents(self, documents: List[Dict[str, Any]]):
        ids = [doc.get('id') or str(uuid.uuid4()) for doc in documents]
        embeddings = [doc['embedding'] for doc in documents]
        metadatas = [doc['metadata'] for doc in documents]
        documents_text = [doc['content'] for doc in documents]
        
        await asyncio.to_thread(
            self.collection.upsert,
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents_text
        )

    async def file_exists(self, sha256: str, size: int, last_modified: int) -> bool:
        try:
            res = await asyncio.to_thread(
                self.collection.get,
                where={'fileSHA256': sha256, 'fileSize': size, 'lastModified': last_modified},
                limit=1,
                include=[] # Don't fetch embeddings/docs
            )
            ids = res.get('ids') or []
            return len(ids) > 0
        except Exception:
            return False

    async def similarity_search(self, vector: List[float], limit: int = 5, filter: Optional[Dict] = None) -> List[Dict[str, Any]]:
        results = await asyncio.to_thread(
            self.collection.query,
            query_embeddings=[vector],
            n_results=limit,
            where=filter
        )
        
        # Unpack Chroma structure
        # results['documents'][0] is list of docs for first query
        if not results['documents']: return []
        
        out = []
        for i in range(len(results['documents'][0])):
            out.append({
                'content': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'score': 1.0 - (results['distances'][0][i] if 'distances' in results else 0) 
            })
        return out

    async def hybrid_search(self, text: str, vector: List[float], limit: int = 5, filter: Optional[Dict] = None) -> List[Dict[str, Any]]:
        return await self.similarity_search(vector, limit, filter)

    async def list_documents(self, filter: Optional[Dict[str, Any]] = None, limit: int = 100, cursor: Optional[str] = None) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        kwargs: Dict[str, Any] = {}
        if filter:
            kwargs["where"] = filter
        limit_int = max(1, int(limit))
        offset = int(cursor) if cursor and cursor.isdigit() else 0
        try:
            res = await asyncio.to_thread(self.collection.get, limit=limit_int, offset=offset, **kwargs)
        except TypeError:
            res = await asyncio.to_thread(self.collection.get, **kwargs)
        ids = res.get("ids") or []
        documents = res.get("documents") or []
        metadatas = res.get("metadatas") or []
        out: List[Dict[str, Any]] = []
        for i in range(min(len(ids), len(documents))):
            out.append(
                {
                    "id": ids[i],
                    "content": documents[i],
                    "metadata": metadatas[i] if i < len(metadatas) else {},
                }
            )
        
        next_cursor = str(offset + len(out)) if len(out) == limit_int else None
        return out, next_cursor

    async def delete_documents(self, filter: Dict[str, Any]) -> int:
        deleted = 0
        offset = 0
        batch = 1000
        while True:
            try:
                res = await asyncio.to_thread(self.collection.get, where=filter, limit=batch, offset=offset)
            except TypeError:
                res = await asyncio.to_thread(self.collection.get, where=filter)
            ids = res.get("ids") or []
            if not ids:
                break
            await asyncio.to_thread(self.collection.delete, ids=ids)
            deleted += len(ids)
            if len(ids) < batch:
                break
            offset += batch
        return deleted

    async def update_documents(self, filter: Dict[str, Any], update_data: Dict[str, Any]) -> int:
        if not update_data:
            return 0
        docs = await self.list_documents(filter=filter, limit=100000)
        if not docs:
            return 0
        new_content = update_data.get("content", None)
        update_meta = update_data.get("metadata", None)
        ids: List[str] = []
        documents: List[str] = []
        metadatas: List[Dict[str, Any]] = []
        for d in docs:
            ids.append(d["id"])
            documents.append(new_content if isinstance(new_content, str) else d.get("content", ""))
            md = d.get("metadata") or {}
            if isinstance(update_meta, dict):
                merged = dict(md)
                merged.update(update_meta)
                metadatas.append(merged)
            else:
                metadatas.append(md)
        await asyncio.to_thread(self.collection.update, ids=ids, documents=documents, metadatas=metadatas)
        return len(ids)
