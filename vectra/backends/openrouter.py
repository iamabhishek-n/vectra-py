import os
from typing import List, AsyncGenerator
from openai import AsyncOpenAI
from ..config import LLMConfig

class OpenRouterBackend:
    def __init__(self, config: LLMConfig):
        self.config = config
        api_key = config.api_key or os.getenv('OPENROUTER_API_KEY')
        if not api_key:
            raise ValueError('OpenRouter API Key missing. Set OPENROUTER_API_KEY.')
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=config.base_url or 'https://openrouter.ai/api/v1',
            default_headers=config.default_headers or {
                'HTTP-Referer': os.getenv('OPENROUTER_REFERER', ''),
                'X-Title': os.getenv('OPENROUTER_TITLE', '')
            }
        )

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError('OpenRouter embeddings are not supported via this adapter.')

    async def embed_query(self, text: str) -> List[float]:
        raise NotImplementedError('OpenRouter embeddings are not supported via this adapter.')

    async def generate(self, prompt: str, system_instruction: str = 'You are a helpful assistant.') -> str:
        res = await self.client.chat.completions.create(
            model=self.config.model_name,
            messages=[
                { 'role': 'system', 'content': system_instruction },
                { 'role': 'user', 'content': prompt }
            ],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )
        return res.choices[0].message.content

    async def generate_stream(self, prompt: str, system_instruction: str = 'You are a helpful assistant.') -> AsyncGenerator[dict, None]:
        stream = await self.client.chat.completions.create(
            model=self.config.model_name,
            messages=[
                { 'role': 'system', 'content': system_instruction },
                { 'role': 'user', 'content': prompt }
            ],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            stream=True
        )
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield { 'delta': chunk.choices[0].delta.content, 'finish_reason': None, 'usage': None }
