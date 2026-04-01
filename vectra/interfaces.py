from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple

class VectorStore(ABC):
    @abstractmethod
    async def add_documents(self, documents: List[Dict[str, Any]]):
        pass

    async def upsert_documents(self, documents: List[Dict[str, Any]]):
        """
        Upsert documents (insert or update on conflict).
        """
        raise NotImplementedError("Method 'upsert_documents' must be implemented.")

    @abstractmethod
    async def similarity_search(self, vector: List[float], limit: int = 5, filter: Optional[Dict] = None) -> List[Dict[str, Any]]:
        pass
    
    # New method for Hybrid Search support
    @abstractmethod
    async def hybrid_search(self, text: str, vector: List[float], limit: int = 5, filter: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Perform hybrid search (Sparse + Dense). 
        Default implementation falls back to similarity_search if not supported.
        """
        return await self.similarity_search(vector, limit, filter)
    
    async def file_exists(self, sha256: str, size: int, last_modified: int) -> bool:
        return False

    @abstractmethod
    async def delete_documents(self, filter: Dict[str, Any]) -> int:
        raise NotImplementedError

    @abstractmethod
    async def update_documents(self, filter: Dict[str, Any], update_data: Dict[str, Any]) -> int:
        raise NotImplementedError

    @abstractmethod
    async def list_documents(self, filter: Optional[Dict[str, Any]] = None, limit: int = 100, cursor: Optional[str] = None) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        raise NotImplementedError

class VectraMiddleware:
    """Base class for Vectra middlewares to intercept and transform data."""
    async def on_before_chunk(self, text: str, config: Any) -> str:
        """Transform text before chunking"""
        return text
    
    async def on_after_embed(self, chunks: List[str], embeddings: List[List[float]]) -> Tuple[List[str], List[List[float]]]:
        """Filter/transform after embedding"""
        return chunks, embeddings
    
    async def on_before_retrieve(self, query: str, vector: List[float]) -> Tuple[str, List[float]]:
        """Modify query or vector before retrieval"""
        return query, vector
    
    async def on_after_generate(self, answer: str, sources: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
        """Post-process generated answer"""
        return answer, sources
