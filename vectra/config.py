from enum import Enum
from typing import List, Optional, Any, Dict
from pydantic import BaseModel, Field, field_validator

class ProviderType(str, Enum):
    OPENAI = 'openai'
    ANTHROPIC = 'anthropic'
    GEMINI = 'gemini'
    OPENROUTER = 'openrouter'
    HUGGINGFACE = 'huggingface'
    OLLAMA = 'ollama'

class ChunkingStrategy(str, Enum):
    RECURSIVE = 'recursive'
    AGENTIC = 'agentic'

class RetrievalStrategy(str, Enum):
    NAIVE = 'naive'
    HYDE = 'hyde'
    MULTI_QUERY = 'multi_query'
    HYBRID = 'hybrid'  # New Strategy
    MMR = 'mmr'

class EmbeddingConfig(BaseModel):
    provider: ProviderType
    api_key: Optional[str] = None
    model_name: str = 'text-embedding-3-small'
    dimensions: Optional[int] = None

class LLMConfig(BaseModel):
    provider: ProviderType
    api_key: Optional[str] = None
    model_name: str
    temperature: float = 0.0
    max_tokens: int = 1024
    base_url: Optional[str] = None
    default_headers: Optional[Dict[str, str]] = None

class ChunkingConfig(BaseModel):
    strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE
    chunk_size: int = 1000
    chunk_overlap: int = 200
    separators: List[str] = ['\n\n', '\n', ' ', '']
    agentic_llm: Optional[LLMConfig] = None

    @field_validator('agentic_llm')
    def check_agentic_llm(cls, v, info):
        if info.data.get('strategy') == ChunkingStrategy.AGENTIC and not v:
            raise ValueError('agentic_llm required for AGENTIC strategy')
        return v

class RerankingConfig(BaseModel):
    enabled: bool = False
    provider: str = 'llm'
    llm_config: Optional[LLMConfig] = None
    top_n: int = 5
    window_size: int = 20

class RetrievalConfig(BaseModel):
    strategy: RetrievalStrategy = RetrievalStrategy.NAIVE
    llm_config: Optional[LLMConfig] = None
    hybrid_alpha: float = 0.5 # Alpha 0-1 (0 = keyword, 1 = dense)
    mmr_lambda: float = 0.5
    mmr_fetch_k: int = 20

    @field_validator('llm_config')
    def check_llm_config(cls, v, info):
        strategy = info.data.get('strategy')
        if strategy in [RetrievalStrategy.HYDE, RetrievalStrategy.MULTI_QUERY] and not v:
            raise ValueError('llm_config required for advanced retrieval')
        return v

class DatabaseConfig(BaseModel):
    type: str # 'prisma', 'chroma', 'custom'
    table_name: Optional[str] = None
    column_map: Optional[Dict[str, str]] = {"content": "content", "vector": "vector", "metadata": "metadata"}
    client_instance: Any  # Prisma client, Chroma client, etc.

class VectraConfig(BaseModel):
    embedding: EmbeddingConfig
    llm: LLMConfig
    database: DatabaseConfig
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    reranking: RerankingConfig = Field(default_factory=RerankingConfig)
    callbacks: List[Any] = []
    metadata: Optional[Dict[str, Any]] = None
    ingestion: Optional[Dict[str, Any]] = Field(default_factory=lambda: { 'rate_limit_enabled': False, 'concurrency_limit': 5 })
    memory: Optional[Dict[str, Any]] = Field(default_factory=lambda: { 'enabled': False, 'type': 'in-memory', 'max_messages': 20 })
    query_planning: Optional[Dict[str, Any]] = None
    grounding: Optional[Dict[str, Any]] = None
    generation: Optional[Dict[str, Any]] = None
    prompts: Optional[Dict[str, Any]] = None
    tracing: Optional[Dict[str, Any]] = None
