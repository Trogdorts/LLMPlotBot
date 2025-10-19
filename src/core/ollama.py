"""Async connector for interacting with Ollama's OpenAI-compatible API."""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, Mapping

import httpx


class OllamaConnector:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        timeout: float,
        logger,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.logger = logger
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=timeout)
        self._lock = asyncio.Lock()

    async def generate(self, prompt: str, headline: str) -> Dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": headline},
            ],
            "temperature": 0.7,
            "stream": False,
        }
        async with self._lock:
            start = time.perf_counter()
            response = await self._client.post("/v1/chat/completions", json=payload)
            elapsed = time.perf_counter() - start
        response.raise_for_status()
        data = response.json()
        text = self.extract_text(data)
        tokens = 0
        usage = data.get("usage")
        if isinstance(usage, Mapping):
            tokens = int(usage.get("total_tokens") or 0)
        self.logger.debug("%s responded in %.2fs", self.model, elapsed)
        return {"text": text, "raw": data, "elapsed": elapsed, "tokens": tokens}

    async def aclose(self) -> None:
        await self._client.aclose()

    @staticmethod
    def extract_text(payload: Mapping[str, Any]) -> str:
        choices = payload.get("choices") if isinstance(payload, Mapping) else None
        if isinstance(choices, list) and choices:
            message = choices[0]
            if isinstance(message, Mapping):
                content = message.get("message")
                if isinstance(content, Mapping):
                    text = content.get("content")
                    if isinstance(text, str):
                        return text.strip()
        return ""


__all__ = ["OllamaConnector"]
