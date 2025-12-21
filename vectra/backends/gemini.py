import os
import asyncio
from typing import List, AsyncGenerator
import google.generativeai as genai
from ..config import EmbeddingConfig, LLMConfig

class GeminiBackend:
    def __init__(self, config: LLMConfig | EmbeddingConfig):
        self.config = config
        api_key = config.api_key or os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("Gemini API Key missing.")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(config.model_name)

    async def _retry(self, fn, retries=3):
        for i in range(retries):
            try:
                return await fn()
            except Exception as e:
                if i == retries - 1: raise e
                await asyncio.sleep(1 * (2 ** i))

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            res = await self._retry(lambda: asyncio.to_thread(
                genai.embed_content,
                model=self.config.model_name,
                content=text,
                task_type="retrieval_document",
                output_dimensionality=self.config.dimensions if getattr(self.config, 'dimensions', None) else None
            ))
            embeddings.append(res['embedding'])
        return embeddings

    async def embed_query(self, text: str) -> List[float]:
        res = await self._retry(lambda: asyncio.to_thread(
            genai.embed_content,
            model=self.config.model_name,
            content=text,
            task_type="retrieval_query",
            output_dimensionality=self.config.dimensions if getattr(self.config, 'dimensions', None) else None
        ))
        return res['embedding']

    async def generate(self, prompt: str, system_instruction: str = "") -> str:
        generation_config = genai.types.GenerationConfig(
            temperature=self.config.temperature,
            max_output_tokens=self.config.max_tokens
        )
        model = genai.GenerativeModel(
            self.config.model_name,
            system_instruction=system_instruction if system_instruction else None
        )
        response = await self._retry(lambda: model.generate_content_async(
            prompt, 
            generation_config=generation_config
        ))
        return response.text

    async def generate_stream(self, prompt: str, system_instruction: str = "") -> AsyncGenerator[dict, None]:
        generation_config = genai.types.GenerationConfig(
            temperature=self.config.temperature,
            max_output_tokens=self.config.max_tokens
        )
        model = genai.GenerativeModel(
            self.config.model_name,
            system_instruction=system_instruction if system_instruction else None
        )
        # Google's stream is async iterable
        response = await model.generate_content_async(
            prompt, 
            generation_config=generation_config,
            stream=True
        )
        async for chunk in response:
            if hasattr(chunk, 'text') and chunk.text:
                yield { 'delta': chunk.text, 'finish_reason': None, 'usage': None }
