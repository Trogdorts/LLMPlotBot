"""
ResultWriter for immediate and buffered JSON persistence.
"""

import os
import json
import time
import threading
from typing import Any, Dict


class ResultWriter:
    """
    Buffered writer that persists each model response independently.
    Flushes either by buffer size or elapsed time.
    Uses atomic rename to prevent file corruption.
    """

    def __init__(self, base_dir: str, flush_interval: int = 50, flush_seconds: float = 5.0, logger=None):
        self.base = base_dir
        self.logger = logger
        self.flush_interval = flush_interval
        self.flush_seconds = flush_seconds
        self.buffer = []
        self.last_flush = time.time()
        self.lock = threading.Lock()
        os.makedirs(base_dir, exist_ok=True)

    def write(self, id: str, model: str, prompt_hash: str, response: Dict[str, Any]):
        """Queue a single write and flush if thresholds are met."""
        self.buffer.append((id, model, prompt_hash, response))
        if len(self.buffer) >= self.flush_interval or (time.time() - self.last_flush) > self.flush_seconds:
            self.flush()

    def flush(self):
        """Persist all buffered writes to disk with atomic file replacement."""
        with self.lock:
            for id, model, prompt_hash, response in self.buffer:
                self._write_one(id, model, prompt_hash, response)
            self.buffer.clear()
            self.last_flush = time.time()

    def _write_one(self, id, model, prompt_hash, response):
        path = os.path.join(self.base, f"{id}.json")
        tmp = path + ".tmp"

        data = {"id": id}

        # load previous record
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                data = {"id": id}

        # extract title once for top level
        title = response.get("title", "")
        data["title"] = title

        # remove it from nested response before writing
        if "title" in response:
            response = dict(response)
            response.pop("title")

        data.setdefault("llm_models", {}).setdefault(model, {})[prompt_hash] = response

        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.write("\n")
        os.replace(tmp, path)

        if self.logger:
            self.logger.debug(f"Saved {id} ({title}) -> {model}/{prompt_hash}")

