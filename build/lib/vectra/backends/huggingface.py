import os
import asyncio
from typing import List, AsyncGenerator
import json
import urllib.request
from ..config import LLMConfig

class HuggingFaceBackend:
    def __init__(self, config: LLMConfig):
        self.config = config
        self.api_key = config.api_key or os.getenv('HUGGINGFACE_API_KEY')
        if not self.api_key:
            raise ValueError('HuggingFace API Key missing. Set HUGGINGFACE_API_KEY.')
        self.base_url = 'https://api-inference.huggingface.co/models'

    async def _post(self, model: str, payload: dict):
        def _do():
            req = urllib.request.Request(
                url=f"{self.base_url}/{model}",
                data=json.dumps(payload).encode('utf-8'),
                headers={ 'Authorization': f'Bearer {self.api_key}', 'Content-Type': 'application/json' },
                method='POST'
            )
            with urllib.request.urlopen(req) as resp:
                return json.loads(resp.read().decode('utf-8'))
        return await asyncio.to_thread(_do)

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        out: List[List[float]] = []
        for text in texts:
            r = await self._post(self.config.model_name, { 'inputs': text, 'options': { 'wait_for_model': True } })
            vec = r if isinstance(r, list) else r.get('embedding')
            flat = []
            def _flatten(x):
                if isinstance(x, list):
                    for i in x: _flatten(i)
                else:
                    try: flat.append(float(x))
                    except: pass
            _flatten(vec)
            out.append(flat)
        return out

    async def embed_query(self, text: str) -> List[float]:
        r = await self._post(self.config.model_name, { 'inputs': text, 'options': { 'wait_for_model': True } })
        vec = r if isinstance(r, list) else r.get('embedding')
        flat: List[float] = []
        def _flatten(x):
            if isinstance(x, list):
                for i in x: _flatten(i)
            else:
                try: flat.append(float(x))
                except: pass
        _flatten(vec)
        return flat

    async def generate(self, prompt: str, system_instruction: str = '') -> str:
        inputs = f"{system_instruction}\n{prompt}" if system_instruction else prompt
        r = await self._post(self.config.model_name, { 'inputs': inputs, 'parameters': { 'temperature': self.config.temperature }, 'options': { 'wait_for_model': True } })
        if isinstance(r, list) and len(r) and isinstance(r[0], dict) and 'generated_text' in r[0]:
            return r[0]['generated_text']
        if isinstance(r, str):
            return r
        if isinstance(r, dict) and 'generated_text' in r:
            return r['generated_text']
        return json.dumps(r)

    async def generate_stream(self, prompt: str, system_instruction: str = '') -> AsyncGenerator[dict, None]:
        text = await self.generate(prompt, system_instruction)
        yield { 'delta': text, 'finish_reason': None, 'usage': None }
