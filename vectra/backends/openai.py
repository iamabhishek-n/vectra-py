import os
import asyncio
from typing import List, AsyncGenerator
from openai import AsyncOpenAI
from ..config import EmbeddingConfig, LLMConfig

class OpenAIBackend:
    def __init__(self, config: LLMConfig | EmbeddingConfig):
        self.config = config
        api_key = config.api_key or os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API Key missing.")
        self.client = AsyncOpenAI(api_key=api_key)

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        res = await self.client.embeddings.create(
            input=texts,
            model=self.config.model_name,
            dimensions=self.config.dimensions or 1536
        )
        return [d.embedding for d in res.data]

    async def embed_query(self, text: str) -> List[float]:
        res = await self.client.embeddings.create(
            input=[text],
            model=self.config.model_name,
            dimensions=self.config.dimensions
        )
        return res.data[0].embedding

    async def generate(self, prompt: str, system_instruction: str = "You are a helpful assistant.") -> str:
        res = await self.client.chat.completions.create(
            model=self.config.model_name,
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt}
            ],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )
        return res.choices[0].message.content

    async def generate_stream(self, prompt: str, system_instruction: str = "You are a helpful assistant.") -> AsyncGenerator[dict, None]:
        stream = await self.client.chat.completions.create(
            model=self.config.model_name,
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt}
            ],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            stream=True
        )
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield { 'delta': chunk.choices[0].delta.content, 'finish_reason': None, 'usage': None }
