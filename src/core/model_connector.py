"""Simplified ModelConnector for direct, synchronous calls to LM Studio."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

import requests


Payload = Dict[str, Any]
ResponseData = Tuple[Dict[str, Any], float]


@dataclass(slots=True)
class ModelConnector:
    """Thin wrapper around the LM Studio HTTP API."""

    model: str
    url: str
    timeout: int
    logger: logging.Logger
    session: requests.Session = field(default_factory=requests.Session)

    def send_to_model(self, prompt: str, title_id: str) -> ResponseData:
        """Send ``prompt`` to the backing model and return its parsed JSON body."""

        payload: Payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "stream": False,
        }

        start = time.perf_counter()
        try:
            response = self.session.post(self.url, json=payload, timeout=self.timeout)
        except requests.RequestException as exc:
            self.logger.error("Connection error for %s: %s", title_id, exc)
            raise

        elapsed = time.perf_counter() - start
        body_size = len(response.content)

        if not response.ok:
            snippet = response.text[:200]
            self.logger.error("HTTP %s for %s: %s", response.status_code, title_id, snippet)
            response.raise_for_status()

        try:
            data: Dict[str, Any] = response.json()
        except ValueError as exc:
            self.logger.error("JSON decode error for %s: %s", title_id, exc)
            self.logger.debug("Raw HTTP body for %s:\n%s", title_id, response.text)
            raise

        self.logger.info(
            "%s: %.2fs | %s bytes | HTTP %s",
            title_id,
            elapsed,
            f"{body_size:,}",
            response.status_code,
        )
        return data, elapsed

    @staticmethod
    def extract_content(response: Dict[str, Any]) -> str:
        """Pull the assistant message content from an LM Studio JSON response."""

        return (
            response.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )
