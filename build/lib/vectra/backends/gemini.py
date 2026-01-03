import os
import asyncio
from typing import List, AsyncGenerator
from google import genai
from google.genai import types
from ..config import EmbeddingConfig, LLMConfig

class GeminiBackend:
    def __init__(self, config: LLMConfig | EmbeddingConfig):
        self.config = config
        api_key = config.api_key or os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("Gemini API Key missing.")
        self.client = genai.Client(api_key=api_key)

    async def _retry(self, fn, retries=3):
        for i in range(retries):
            try:
                return await fn()
            except Exception as e:
                if i == retries - 1: raise e
                await asyncio.sleep(1 * (2 ** i))

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        BATCH_SIZE = 100
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i:i + BATCH_SIZE]
            
            async def _call_embed():
                 response = await self.client.aio.models.embed_content(
                    model=self.config.model_name,
                    contents=batch,
                    config=types.EmbedContentConfig(
                         task_type="RETRIEVAL_DOCUMENT",
                         output_dimensionality=self.config.dimensions if getattr(self.config, 'dimensions', None) else None
                    )
                )
                 return response

            res = await self._retry(_call_embed)
            # The response structure for batch embedding in google-genai
            # res.embeddings is a list of EmbedContentResponse or similar objects, each has 'values'
            embeddings.extend([e.values for e in res.embeddings])
        return embeddings

    async def embed_query(self, text: str) -> List[float]:
        async def _call_embed():
             response = await self.client.aio.models.embed_content(
                model=self.config.model_name,
                contents=text,
                config=types.EmbedContentConfig(
                     task_type="RETRIEVAL_QUERY",
                     output_dimensionality=self.config.dimensions if getattr(self.config, 'dimensions', None) else None
                )
            )
             return response

        res = await self._retry(_call_embed)
        return res.embeddings[0].values

    async def generate(self, prompt: str, system_instruction: str = "") -> str:
        config_args = {}
        if self.config.temperature is not None:
             config_args['temperature'] = self.config.temperature
        if self.config.max_tokens is not None:
             config_args['max_output_tokens'] = self.config.max_tokens
        
        if system_instruction:
             config_args['system_instruction'] = system_instruction

        async def _call_generate():
            response = await self.client.aio.models.generate_content(
                model=self.config.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(**config_args)
            )
            return response

        response = await self._retry(_call_generate)
        return response.text

    async def generate_stream(self, prompt: str, system_instruction: str = "") -> AsyncGenerator[dict, None]:
        config_args = {}
        if self.config.temperature is not None:
             config_args['temperature'] = self.config.temperature
        if self.config.max_tokens is not None:
             config_args['max_output_tokens'] = self.config.max_tokens
        
        if system_instruction:
             config_args['system_instruction'] = system_instruction

        # google-genai aio generate_content_stream returns an async iterator
        async for chunk in await self.client.aio.models.generate_content_stream(
            model=self.config.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(**config_args)
        ):
            if chunk.text:
                yield { 'delta': chunk.text, 'finish_reason': None, 'usage': None }
