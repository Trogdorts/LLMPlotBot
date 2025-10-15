"""ResultWriter with configurable persistence strategy and file locking."""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Dict, List, Tuple

from src.util.file_lock import FileLock, FileLockTimeout


class ResultWriter:
    """Persist responses either immediately or via timed batching with file locks."""

    def __init__(
        self,
        base_dir: str,
        *,
        strategy: str = "immediate",
        flush_interval: int = 50,
        flush_seconds: float = 5.0,
        lock_timeout: float = 10.0,
        lock_poll_interval: float = 0.1,
        lock_stale_seconds: float = 300.0,
        logger=None,
    ) -> None:
        self.base = base_dir
        self.logger = logger
        self.strategy = strategy.lower()
        self.flush_interval = max(1, flush_interval)
        self.flush_seconds = max(0.1, flush_seconds)
        self.lock_timeout = max(0.1, lock_timeout)
        self.lock_poll_interval = max(0.01, lock_poll_interval)
        self.lock_stale_seconds = max(1.0, lock_stale_seconds)
        self.buffer: List[Tuple[str, str, str, Dict[str, Any]]] = []
        self.last_flush = time.time()
        self.lock = threading.Lock()
        os.makedirs(base_dir, exist_ok=True)

    # ------------------------------------------------------------------
    def write(self, id: str, model: str, prompt_hash: str, response: Dict[str, Any]):
        """Queue a single write and flush based on the configured strategy."""

        with self.lock:
            self.buffer.append((id, model, prompt_hash, response))
            if self.strategy == "immediate":
                self._flush_locked(force=True)
                return

            should_flush = (
                len(self.buffer) >= self.flush_interval
                or (time.time() - self.last_flush) > self.flush_seconds
            )
            if should_flush:
                self._flush_locked(force=True)

    # ------------------------------------------------------------------
    def flush(self):
        """Persist all buffered writes to disk with atomic file replacement."""

        with self.lock:
            self._flush_locked(force=True)

    # ------------------------------------------------------------------
    def _flush_locked(self, *, force: bool = False):
        if not self.buffer:
            return

        if not force and self.strategy != "immediate":
            elapsed = time.time() - self.last_flush
            if elapsed < self.flush_seconds and len(self.buffer) < self.flush_interval:
                return

        pending = list(self.buffer)
        self.buffer.clear()
        self.last_flush = time.time()

        failed: List[Tuple[str, str, str, Dict[str, Any]]] = []
        success_count = 0

        for id, model, prompt_hash, response in pending:
            try:
                self._write_one(id, model, prompt_hash, response)
                success_count += 1
            except FileLockTimeout as exc:
                failed.append((id, model, prompt_hash, response))
                if self.logger:
                    self.logger.warning(str(exc))
            except Exception as exc:  # pragma: no cover - defensive
                failed.append((id, model, prompt_hash, response))
                if self.logger:
                    self.logger.error(
                        "Failed to write %s for model %s: %s", id, model, exc, exc_info=True
                    )

        if failed:
            # Requeue failed entries for another attempt on the next flush.
            self.buffer = failed + self.buffer

        if self.logger:
            if success_count:
                self.logger.info("Flushed %s result(s) to disk.", success_count)
            if failed:
                self.logger.warning("Deferred %s result(s) due to file lock contention.", len(failed))

    # ------------------------------------------------------------------
    def _write_one(self, id, model, prompt_hash, response):
        path = os.path.join(self.base, f"{id}.json")
        tmp = path + ".tmp"
        lock_path = path + ".lock"

        with FileLock(
            lock_path,
            timeout=self.lock_timeout,
            poll_interval=self.lock_poll_interval,
            stale_seconds=self.lock_stale_seconds,
        ):
            data = {"id": id}

            if os.path.exists(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                except Exception:
                    data = {"id": id}

            title = response.get("title", "")
            data["title"] = title

            if "title" in response:
                response = dict(response)
                response.pop("title")

            data.setdefault("llm_models", {}).setdefault(model, {})[prompt_hash] = response

            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                f.write("\n")
            os.replace(tmp, path)

        if self.logger:
            self.logger.debug("Saved %s (%s) -> %s/%s", id, title, model, prompt_hash)

