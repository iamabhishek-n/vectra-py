import os
import asyncio
from typing import AsyncGenerator
from anthropic import AsyncAnthropic
from ..config import LLMConfig

class AnthropicBackend:
    def __init__(self, config: LLMConfig):
        self.config = config
        api_key = config.api_key or os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            raise ValueError("Anthropic API Key missing.")
        self.client = AsyncAnthropic(api_key=api_key)

    async def embed_documents(self, texts):
        raise NotImplementedError("Anthropic does not provide embedding models.")

    async def embed_query(self, text):
        raise NotImplementedError("Anthropic does not provide embedding models.")

    async def generate(self, prompt: str, system_instruction: str = "You are a helpful assistant.") -> str:
        res = await self.client.messages.create(
            model=self.config.model_name,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            system=system_instruction,
            messages=[{"role": "user", "content": prompt}]
        )
        return res.content[0].text
    
    async def generate_stream(self, prompt: str, system_instruction: str = "You are a helpful assistant.") -> AsyncGenerator[dict, None]:
        stream = await self.client.messages.create(
            model=self.config.model_name,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            system=system_instruction,
            messages=[{"role": "user", "content": prompt}],
            stream=True
        )
        async for chunk in stream:
            if chunk.type == 'content_block_delta':
                yield { 'delta': chunk.delta.text, 'finish_reason': None, 'usage': None }
