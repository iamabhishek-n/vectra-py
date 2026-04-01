import json
import asyncio
import aiohttp

class OllamaBackend:
    def __init__(self, config):
        self.config = config
        self.base_url = getattr(config, 'base_url', None) or 'http://localhost:11434'

    async def embed_documents(self, texts):
        out = []
        async with aiohttp.ClientSession() as session:
            for t in texts:
                async with session.post(f"{self.base_url}/api/embeddings", json={ 'model': self.config.model_name, 'prompt': t }) as resp:
                    j = await resp.json()
                    out.append(j.get('embedding') or j.get('data') or [])
        return out

    async def embed_query(self, text):
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{self.base_url}/api/embeddings", json={ 'model': self.config.model_name, 'prompt': text }) as resp:
                j = await resp.json()
                return j.get('embedding') or j.get('data') or []

    async def generate(self, prompt, sys=None):
        async with aiohttp.ClientSession() as session:
            payload = {
                'model': self.config.model_name,
                'prompt': (f"{sys}\n\n{prompt}" if sys else prompt),
                'stream': False
            }
            async with session.post(f"{self.base_url}/api/generate", json=payload) as resp:
                j = await resp.json()
                return j.get('response', '')

    async def generate_stream(self, prompt, sys=None):
        async with aiohttp.ClientSession() as session:
            payload = {
                'model': self.config.model_name,
                'prompt': (f"{sys}\n\n{prompt}" if sys else prompt),
                'stream': True
            }
            async with session.post(f"{self.base_url}/api/generate", json=payload) as resp:
                async for line in resp.content:
                    if line:
                        try:
                            obj = json.loads(line)
                            d = obj.get('response', '')
                            if d:
                                yield { 'delta': d, 'finish_reason': 'stop' if obj.get('done') else None, 'usage': None }
                        except Exception:
                            continue
