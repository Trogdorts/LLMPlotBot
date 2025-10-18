"""HTTP connectors for interacting with chat-completion style models."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Dict, Mapping, Sequence

import requests


@dataclass
class ModelRequest:
    prompt: str
    items: Sequence[Mapping[str, object]]


class ModelConnector:
    def __init__(self, model: str, endpoint: str, *, timeout: int, logger: logging.Logger) -> None:
        self.model = model
        self.endpoint = endpoint
        self.timeout = timeout
        self.logger = logger

    def invoke(self, request: ModelRequest) -> Dict[str, object]:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": request.prompt}],
            "temperature": 0.7,
            "stream": False,
        }
        start = time.perf_counter()
        response = requests.post(self.endpoint, json=payload, timeout=self.timeout)
        elapsed = time.perf_counter() - start
        try:
            response.raise_for_status()
        except requests.HTTPError:
            self.logger.error("%s responded with HTTP %s", self.endpoint, response.status_code)
            raise
        self.logger.debug("%s completed in %.2fs", self.model, elapsed)
        data = response.json()
        return {"raw": data, "elapsed": elapsed}

    @staticmethod
    def extract_text(payload: Mapping[str, object]) -> str:
        choices = payload.get("choices") if isinstance(payload, Mapping) else None
        if isinstance(choices, Sequence) and choices:
            message = choices[0]
            if isinstance(message, Mapping):
                content = message.get("message")
                if isinstance(content, Mapping):
                    text = content.get("content")
                    if isinstance(text, str):
                        return text.strip()
        return ""

    def shutdown(self) -> None:
        self.logger.debug("Connector for %s shutdown", self.model)


__all__ = ["ModelConnector", "ModelRequest"]
