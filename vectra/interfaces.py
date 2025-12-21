from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class VectorStore(ABC):
    @abstractmethod
    async def add_documents(self, documents: List[Dict[str, Any]]):
        pass

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
    async def list_documents(self, filter: Optional[Dict[str, Any]] = None, limit: int = 100) -> List[Dict[str, Any]]:
        raise NotImplementedError
