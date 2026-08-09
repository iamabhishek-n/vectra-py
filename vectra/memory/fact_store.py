import json
import re
import time
import uuid
from typing import Any, Dict

from ..backends.postgres_store import assert_safe_identifier

EXTRACTION_PROMPT = """Extract factual (subject, predicate, object) triples from the conversation turn below. Only extract clear, stated facts about the user or entities discussed — not questions, greetings, or the assistant's own commentary. Return strict JSON only, no prose: {{"facts": [{{"subject": "...", "predicate": "...", "object": "..."}}]}}. If there are no clear facts, return {{"facts": []}}.

User: {user}
Assistant: {assistant}"""

_FENCE_RE = re.compile(r'^```(?:json)?\s*|\s*```$', re.IGNORECASE)


class FactStore:
    def __init__(self, config: Any):
        self.client = config.client_instance
        self.table_name = assert_safe_identifier(getattr(config, 'table_name', 'VectraFact') or 'VectraFact', 'table_name')
        self.llm = getattr(config, 'llm', None)
        self.embedder = getattr(config, 'embedder', None)
        self.cache_ttl_ms = getattr(config, 'cache_ttl_ms', 30000)
        self._read_cache: Dict[str, Any] = {}

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

    async def write(self, session_id: str, turn: Dict[str, str]):
        if not session_id or not self.llm or not self.embedder:
            return

        prompt = EXTRACTION_PROMPT.format(
            user=turn.get("user_message", ""),
            assistant=turn.get("assistant_message", ""),
        )

        try:
            raw = await self.llm.generate(prompt, "You extract structured facts as strict JSON.")
            cleaned = _FENCE_RE.sub('', str(raw).strip()).strip()
            parsed = json.loads(cleaned)
            facts = parsed.get("facts", []) if isinstance(parsed, dict) else []
        except Exception:
            return

        facts = [f for f in facts if f.get("subject") and f.get("predicate") and f.get("object")]
        if not facts:
            return

        texts = [f"{f['subject']} {f['predicate']} {f['object']}" for f in facts]
        try:
            embeddings = await self.embedder.embed_documents(texts)
        except Exception:
            return

        t = self.table_name
        async with self._get_connection() as conn:
            for i, f in enumerate(facts):
                vec = f"[{','.join(map(str, embeddings[i]))}]"

                existing = await conn.fetchrow(
                    f'SELECT "id","object" FROM "{t}" WHERE "session_id" = $1 AND "subject" = $2 AND "predicate" = $3 AND "invalid_at" IS NULL',
                    session_id, f["subject"], f["predicate"],
                )

                if existing and existing["object"] == f["object"]:
                    continue
                if existing:
                    try:
                        await conn.execute(f'UPDATE "{t}" SET "invalid_at" = NOW() WHERE "id" = $1', existing["id"])
                    except Exception:
                        pass

                fact_id = str(uuid.uuid4())
                try:
                    await conn.execute(
                        f'INSERT INTO "{t}" ("id","session_id","subject","predicate","object","embedding","source_message_id") VALUES ($1,$2,$3,$4,$5,$6,$7)',
                        fact_id, session_id, f["subject"], f["predicate"], f["object"], vec, turn.get("source_message_id"),
                    )
                except Exception:
                    pass

        self._invalidate_session_cache(session_id)

    def _invalidate_session_cache(self, session_id: str):
        prefix = f"{session_id}:"
        for key in [k for k in self._read_cache if k.startswith(prefix)]:
            del self._read_cache[key]

    async def read(self, session_id: str, query: str, limit: int = 10):
        if not session_id or not self.embedder:
            return []

        cache_key = f"{session_id}:{query}"
        cached = self._read_cache.get(cache_key)
        if cached and (time.time() * 1000 - cached['ts']) < self.cache_ttl_ms:
            return cached['value']

        vector = await self.embedder.embed_query(query)
        vec = f"[{','.join(map(str, vector))}]"
        t = self.table_name

        async with self._get_connection() as conn:
            rows = await conn.fetch(
                f'''SELECT "id","subject","predicate","object","valid_at","invalid_at"
                    FROM "{t}"
                    WHERE "session_id" = $1 AND "invalid_at" IS NULL
                    ORDER BY "embedding" <=> $2
                    LIMIT $3''',
                session_id, vec, max(1, limit),
            )
        result = [dict(r) for r in rows]
        self._read_cache[cache_key] = {'ts': time.time() * 1000, 'value': result}
        return result
