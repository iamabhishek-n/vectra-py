from typing import List, Dict, Any, Optional
import json

class InMemoryHistory:
    def __init__(self, max_messages: int = 20):
        self.sessions: Dict[str, List[Dict[str, Any]]] = {}
        self.max_messages = max_messages
    def add_message(self, session_id: str, role: str, content: str):
        if not session_id:
            return
        arr = self.sessions.get(session_id, [])
        arr.append({ 'role': role, 'content': content })
        start = max(0, len(arr) - self.max_messages)
        self.sessions[session_id] = arr[start:]
    def get_recent(self, session_id: str, n: int = 10) -> List[Dict[str, Any]]:
        arr = self.sessions.get(session_id, [])
        start = max(0, len(arr) - n)
        return arr[start:]

class RedisHistory:
    def __init__(self, client: Any, key_prefix: str = 'vectra:chat:', max_messages: int = 20):
        self.client = client
        self.key_prefix = key_prefix
        self.max_messages = max_messages
    async def add_message(self, session_id: str, role: str, content: str):
        if not session_id or not self.client:
            return
        key = f"{self.key_prefix}{session_id}"
        payload = json.dumps({ 'role': role, 'content': content })
        try:
            if hasattr(self.client, 'rpush'):
                await self.client.rpush(key, payload)
            elif hasattr(self.client, 'lpush'):
                await self.client.lpush(key, payload)
            if hasattr(self.client, 'ltrim'):
                await self.client.ltrim(key, -self.max_messages, -1)
        except Exception:
            pass
    async def get_recent(self, session_id: str, n: int = 10) -> List[Dict[str, Any]]:
        if not session_id or not self.client:
            return []
        key = f"{self.key_prefix}{session_id}"
        try:
            arr = []
            if hasattr(self.client, 'lrange'):
                arr = await self.client.lrange(key, -n, -1)
            elif hasattr(self.client, 'lRange'):
                arr = await self.client.lRange(key, -n, -1)
            out: List[Dict[str, Any]] = []
            for x in arr:
                try:
                    out.append(json.loads(x))
                except Exception:
                    out.append({ 'role': 'assistant', 'content': str(x) })
            return out
        except Exception:
            return []

class PostgresHistory:
    def __init__(self, client: Any, table_name: str = 'ChatMessage', column_map: Optional[Dict[str, str]] = None, max_messages: int = 20):
        self.client = client
        self.table_name = table_name
        self.column_map = column_map or { 'sessionId': 'sessionId', 'role': 'role', 'content': 'content', 'createdAt': 'createdAt' }
        self.max_messages = max_messages
    async def add_message(self, session_id: str, role: str, content: str):
        if not session_id or not self.client:
            return
        t = self.table_name
        c = self.column_map
        q = f'INSERT INTO "{t}" ("{c["sessionId"]}","{c["role"]}","{c["content"]}","{c["createdAt"]}") VALUES ($1,$2,$3,NOW())'
        try:
            if hasattr(self.client, 'execute_raw'):
                await self.client.execute_raw(q, session_id, role, content)
            elif hasattr(self.client, '$executeRawUnsafe'):
                await self.client.$executeRawUnsafe(q, session_id, role, content)
        except Exception:
            pass
    async def get_recent(self, session_id: str, n: int = 10) -> List[Dict[str, Any]]:
        if not session_id or not self.client:
            return []
        t = self.table_name
        c = self.column_map
        q = f'SELECT "{c["role"]}" as role, "{c["content"]}" as content FROM "{t}" WHERE "{c["sessionId"]}" = $1 ORDER BY "{c["createdAt"]}" DESC LIMIT {max(1,n)}'
        try:
            rows = []
            if hasattr(self.client, 'query_raw'):
                rows = await self.client.query_raw(q, session_id)
            elif hasattr(self.client, '$queryRawUnsafe'):
                rows = await self.client.$queryRawUnsafe(q, session_id)
            if isinstance(rows, list):
                rows.reverse()
                return [{ 'role': r.get('role'), 'content': r.get('content') } for r in rows]
            return []
        except Exception:
            return []
