import json
import asyncio
import requests

class OllamaBackend:
    def __init__(self, config):
        self.config = config
        self.base_url = getattr(config, 'base_url', None) or 'http://localhost:11434'
    async def embed_documents(self, texts):
        out = []
        for t in texts:
            r = requests.post(f"{self.base_url}/api/embeddings", json={ 'model': self.config.model_name, 'prompt': t })
            j = r.json()
            out.append(j.get('embedding') or j.get('data') or [])
        return out
    async def embed_query(self, text):
        r = requests.post(f"{self.base_url}/api/embeddings", json={ 'model': self.config.model_name, 'prompt': text })
        j = r.json()
        return j.get('embedding') or j.get('data') or []
    async def generate(self, prompt, sys=None):
        r = requests.post(f"{self.base_url}/api/generate", json={ 'model': self.config.model_name, 'prompt': (f"{sys}\n\n{prompt}" if sys else prompt), 'stream': False })
        j = r.json()
        return j.get('response','')
    async def generate_stream(self, prompt, sys=None):
        text = await self.generate(prompt, sys)
        for i in range(0, len(text), 64):
            yield { 'delta': text[i:i+64], 'finish_reason': None, 'usage': None }
