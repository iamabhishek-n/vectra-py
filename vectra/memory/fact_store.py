from typing import Any

from ..backends.postgres_store import assert_safe_identifier


class FactStore:
    def __init__(self, config: Any):
        self.client = config.client_instance
        self.table_name = assert_safe_identifier(getattr(config, 'table_name', 'VectraFact') or 'VectraFact', 'table_name')

    def _get_connection(self):
        if hasattr(self.client, 'acquire'):
            return self.client.acquire()

        class DummyContext:
            def __init__(self, conn): self.conn = conn
            async def __aenter__(self): return self.conn
            async def __aexit__(self, *args): pass
        return DummyContext(self.client)

    async def ensure_indexes(self, dimensions: int = 1536):
        t = self.table_name
        dim = dimensions or 1536
        async with self._get_connection() as conn:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            await conn.execute(f'''CREATE TABLE IF NOT EXISTS "{t}" (
                "id" TEXT PRIMARY KEY,
                "session_id" TEXT NOT NULL,
                "subject" TEXT NOT NULL,
                "predicate" TEXT NOT NULL,
                "object" TEXT NOT NULL,
                "embedding" vector({dim}),
                "valid_at" TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
                "invalid_at" TIMESTAMP WITH TIME ZONE,
                "source_message_id" TEXT,
                "created_at" TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )''')
            try:
                await conn.execute(f'CREATE INDEX IF NOT EXISTS "{t}_vec_idx" ON "{t}" USING hnsw ("embedding" vector_cosine_ops)')
            except Exception:
                try:
                    await conn.execute(f'CREATE INDEX IF NOT EXISTS "{t}_vec_idx" ON "{t}" USING ivfflat ("embedding" vector_cosine_ops)')
                except Exception:
                    pass
            await conn.execute(f'CREATE INDEX IF NOT EXISTS "{t}_session_temporal_idx" ON "{t}" ("session_id", "valid_at", "invalid_at")')
            await conn.execute(f'CREATE INDEX IF NOT EXISTS "{t}_subject_predicate_idx" ON "{t}" ("session_id", "subject", "predicate")')
