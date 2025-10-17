"""HTTP connector for interacting with chat-completion style LLM endpoints."""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Dict, List, Optional

import requests

from src.core.prompt_spec import PromptSpecification


class ModelConnector:
    """Thin wrapper around the LM Studio HTTP API with light QoL helpers."""

    def __init__(
        self,
        model: str,
        url: str,
        timeout: int,
        compliance_interval: int = 0,
        logger: Optional[logging.Logger] = None,
        expected_language: Optional[str] = None,
        *,
        prompt_spec: Optional[PromptSpecification] = None,
        temperature: float = 0.7,
    ) -> None:
        self.model = model
        self.url = url
        self.timeout = max(1, int(timeout or 1))
        self.temperature = float(temperature)
        self.logger = logger or logging.getLogger(__name__)
        self.expected_language = (expected_language or "").strip()
        self.prompt_spec = prompt_spec

        self._session = requests.Session()
        self._lock = threading.Lock()
        self._requests_sent = 0
        self._compliance_interval = max(0, int(compliance_interval or 0))
        self._compliance_message = self._build_compliance_message()
        self._closed = False

    # ------------------------------------------------------------------
    def _build_compliance_message(self) -> Optional[str]:
        parts: List[str] = []
        if self.prompt_spec is not None:
            field_list = ", ".join(self.prompt_spec.required_field_names)
            if field_list:
                parts.append(f"Return a JSON array with objects containing: {field_list}.")
            if self.prompt_spec.formatting_rules:
                parts.append(self.prompt_spec.formatting_rules[0])
        if self.expected_language:
            parts.append(f"Respond in {self.expected_language}.")
        if not parts:
            return None
        return " ".join(parts)

    # ------------------------------------------------------------------
    def _build_messages(self, prompt: str, *, include_compliance: bool) -> List[Dict[str, str]]:
        messages: List[Dict[str, str]] = []
        if include_compliance and self._compliance_message:
            messages.append({"role": "system", "content": self._compliance_message})
        messages.append({"role": "user", "content": prompt})
        return messages

    def _build_payload(self, prompt: str, *, include_compliance: bool) -> Dict[str, object]:
        return {
            "model": self.model,
            "messages": self._build_messages(prompt, include_compliance=include_compliance),
            "temperature": self.temperature,
            "stream": False,
        }

    # ------------------------------------------------------------------
    def send_to_model(self, prompt: str, title_id: str):
        """Submit ``prompt`` to the configured endpoint and return the raw JSON."""

        with self._lock:
            if self._closed:
                raise RuntimeError("Connector already shut down.")
            self._requests_sent += 1
            include_compliance = (
                self._compliance_interval > 0
                and (self._requests_sent == 1 or self._requests_sent % self._compliance_interval == 0)
            )

        payload = self._build_payload(prompt, include_compliance=include_compliance)
        start = time.perf_counter()
        try:
            response = self._session.post(self.url, json=payload, timeout=self.timeout)
            response.raise_for_status()
        except requests.RequestException as exc:
            if self.logger:
                self.logger.error("Connection error for %s: %s", title_id, exc)
            raise

        elapsed = time.perf_counter() - start
        size = len(response.content)
        if self.logger:
            self.logger.info(
                "%s: %.2fs | %s bytes | HTTP %s",
                title_id,
                elapsed,
                f"{size:,}",
                response.status_code,
            )

        try:
            data = response.json()
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive
            if self.logger:
                snippet = response.text[:200]
                self.logger.error("JSON decode error for %s: %s | %s", title_id, exc, snippet)
            raise
        return data, elapsed

    # ------------------------------------------------------------------
    @staticmethod
    def extract_content(response: dict) -> str:
        return response.get("choices", [{}])[0].get("message", {}).get("content", "").strip()

    # ------------------------------------------------------------------
    def shutdown(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._session.close()


__all__ = ["ModelConnector"]
