from enum import Enum
from typing import List, Optional, Any, Dict, Union
from pydantic import BaseModel, Field, field_validator, ConfigDict

class ProviderType(str, Enum):
    OPENAI = 'openai'
    ANTHROPIC = 'anthropic'
    GEMINI = 'gemini'
    OPENROUTER = 'openrouter'
    HUGGINGFACE = 'huggingface'
    OLLAMA = 'ollama'

class RerankingProvider(str, Enum):
    LLM = 'llm'
    CROSS_ENCODER = 'cross-encoder'
    COHERE = 'cohere'
    JINA = 'jina'

class ChunkingStrategy(str, Enum):
    RECURSIVE = 'recursive'
    AGENTIC = 'agentic'

class RetrievalStrategy(str, Enum):
    NAIVE = 'naive'
    HYDE = 'hyde'
    MULTI_QUERY = 'multi_query'
    HYBRID = 'hybrid'  # New Strategy
    MMR = 'mmr'

class SessionType(str, Enum):
    CLI = 'cli'
    API = 'api'
    CHAT = 'chat'

class IngestionConfig(BaseModel):
    rate_limit_enabled: bool = False
    concurrency_limit: int = 5

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
    provider: RerankingProvider = RerankingProvider.LLM
    llm_config: Optional[LLMConfig] = None
    model_name: Optional[str] = None
    api_key: Optional[str] = None
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
    model_config = ConfigDict(extra='allow')
    type: str # 'prisma', 'chroma', 'custom'
    table_name: Optional[str] = None
    column_map: Optional[Dict[str, str]] = {"content": "content", "vector": "vector", "metadata": "metadata"}
    client_instance: Optional[Any] = None

class ObservabilityConfig(BaseModel):
    enabled: bool = False
    sqlite_path: str = 'vectra-observability.db'
    project_id: str = 'default'
    track_metrics: bool = True
    track_traces: bool = True
    track_logs: bool = True
    session_tracking: bool = True

class TelemetryConfig(BaseModel):
    enabled: bool = True

class GuardrailConfig(BaseModel):
    block_pii: bool = False           # Detect and redact PII in outputs
    block_off_topic: bool = False     # Reject queries unrelated to ingested docs
    max_query_length: int = 2000
    content_filter: bool = False      # Block harmful content generation
    hallucination_check: bool = False # Verify claims against retrieved context

class VectraConfig(BaseModel):
    embedding: EmbeddingConfig
    llm: LLMConfig
    database: DatabaseConfig
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    reranking: RerankingConfig = Field(default_factory=RerankingConfig)
    callbacks: List[Any] = []
    middlewares: List[Any] = []
    metadata: Optional[Dict[str, Any]] = None
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    memory: Optional[Dict[str, Any]] = Field(default_factory=lambda: { 'enabled': False, 'type': 'in-memory', 'max_messages': 20 })
    query_planning: Optional[Dict[str, Any]] = None
    grounding: Optional[Dict[str, Any]] = None
    generation: Optional[Dict[str, Any]] = None
    prompts: Optional[Dict[str, Any]] = None
    tracing: Optional[Dict[str, Any]] = None
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)
    guardrails: GuardrailConfig = Field(default_factory=GuardrailConfig)
    session_type: SessionType = SessionType.API
    max_cache_size: int = 10000
