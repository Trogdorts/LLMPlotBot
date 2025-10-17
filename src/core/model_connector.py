"""Simplified ModelConnector for direct, synchronous calls to LM Studio."""
import json
import logging
import time
import requests


class ModelConnector:
    def __init__(self, model: str, url: str, timeout: int, logger: logging.Logger):
        self.model = model
        self.url = url
        self.timeout = timeout
        self.logger = logger

    def send_to_model(self, prompt: str, title_id: str):
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "stream": False,
        }

        start = time.perf_counter()
        try:
            r = requests.post(self.url, json=payload, timeout=self.timeout)
        except requests.RequestException as e:
            self.logger.error(f"Connection error for {title_id}: {e}")
            raise

        elapsed = time.perf_counter() - start
        size = len(r.content)
        if not r.ok:
            self.logger.error(f"HTTP {r.status_code} for {title_id}: {r.text[:200]}")
            r.raise_for_status()

        try:
            data = r.json()
            self.logger.info(f"{title_id}: {elapsed:.2f}s | {size:,} bytes | HTTP {r.status_code}")
            return data, elapsed
        except Exception as e:
            self.logger.error(f"JSON decode error for {title_id}: {e}")
            print(r.text)
            raise
    def shutdown(self):
        """No-op for compatibility with pipeline shutdown calls."""
        if self.logger:
            self.logger.debug("ModelConnector.shutdown() called — nothing to close.")

    @staticmethod
    def extract_content(response: dict) -> str:
        return response.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
