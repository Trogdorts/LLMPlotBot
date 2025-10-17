"""ResultWriter with configurable persistence strategy and file locking."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any, Dict, List, Tuple

from pathlib import Path

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
        flush_retry_limit: int = 3,
        lock_timeout: float = 10.0,
        lock_poll_interval: float = 0.1,
        lock_stale_seconds: float = 300.0,
        logger: logging.Logger | None = None,
    ) -> None:
        self.base = Path(base_dir)
        self.logger = logger
        self.strategy = strategy.lower()
        self.flush_interval = max(1, flush_interval)
        self.flush_seconds = max(0.1, flush_seconds)
        self.flush_retry_limit = max(1, int(flush_retry_limit or 1))
        self.lock_timeout = max(0.1, lock_timeout)
        self.lock_poll_interval = max(0.01, lock_poll_interval)
        self.lock_stale_seconds = max(1.0, lock_stale_seconds)
        self.buffer: List[Tuple[str, str, str, Dict[str, Any]]] = []
        self.last_flush = time.time()
        self.lock = threading.Lock()
        self.base.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def write(
        self,
        title_id: str,
        model: str,
        prompt_hash: str,
        response: Dict[str, Any],
    ) -> None:
        """Queue a single write and flush based on the configured strategy."""

        with self.lock:
            self.buffer.append((title_id, model, prompt_hash, response))
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

        grouped: Dict[str, List[Tuple[str, str, str, Dict[str, Any]]]] = {}
        for item in pending:
            grouped.setdefault(item[0], []).append(item)

        for title_id, records in grouped.items():
            try:
                written = self._write_batch(title_id, records)
            except Exception as exc:  # pragma: no cover - defensive
                failed.extend(records)
                if self.logger:
                    self.logger.error(
                        "Failed to persist batch for %s: %s",
                        title_id,
                        exc,
                        exc_info=True,
                    )
                continue

            if written:
                success_count += written
            else:
                failed.extend(records)

        if failed:
            # Requeue failed entries for another attempt on the next flush.
            self.buffer = failed + self.buffer

        if self.logger:
            if success_count:
                self.logger.info("Flushed %s result(s) to disk.", success_count)
            if failed:
                self.logger.warning(
                    "Deferred %s result(s) due to file lock contention or errors.",
                    len(failed),
                )

    # ------------------------------------------------------------------
    def _write_batch(
        self, title_id: str, records: List[Tuple[str, str, str, Dict[str, Any]]]
    ) -> int:
        """Persist a batch of responses for the same identifier atomically."""

        attempts_remaining = self.flush_retry_limit
        while attempts_remaining > 0:
            attempts_remaining -= 1
            try:
                return self._persist_records(title_id, records)
            except FileLockTimeout as exc:
                if self.logger:
                    self.logger.warning(str(exc))
                if attempts_remaining == 0:
                    return 0
                # Small backoff before retrying the same file.
                time.sleep(min(0.5 * (self.flush_retry_limit - attempts_remaining), 2.0))
            except Exception:
                if attempts_remaining == 0:
                    raise
                if self.logger:
                    self.logger.warning(
                        "Retrying batch write for %s due to unexpected error.",
                        title_id,
                        exc_info=True,
                    )
                time.sleep(min(0.5 * (self.flush_retry_limit - attempts_remaining), 2.0))
        return 0

    # ------------------------------------------------------------------
    def _persist_records(
        self, title_id: str, records: List[Tuple[str, str, str, Dict[str, Any]]]
    ) -> int:
        path = self.base / f"{title_id}.json"
        tmp = path.with_name(path.name + ".tmp")
        lock_path = path.with_name(path.name + ".lock")

        with FileLock(
            str(lock_path),
            timeout=self.lock_timeout,
            poll_interval=self.lock_poll_interval,
            stale_seconds=self.lock_stale_seconds,
        ):
            data: Dict[str, Any] = {}

            if path.exists():
                try:
                    with path.open("r", encoding="utf-8") as f:
                        data = json.load(f)
                except Exception:
                    data = {}

            if not isinstance(data, dict):
                data = {}

            existing_title = data.get("title", "") if isinstance(data, dict) else ""
            models_section = data.setdefault("llm_models", {})
            if not isinstance(models_section, dict):
                models_section = {}
                data["llm_models"] = models_section

            written = 0
            chosen_title = existing_title

            for _, model, prompt_hash, response in records:
                title = response.get("title", "")
                if title and not chosen_title:
                    chosen_title = title
                elif (
                    title
                    and chosen_title
                    and title != chosen_title
                    and self.logger is not None
                ):
                    self.logger.warning(
                        "Conflicting titles for %s; keeping existing value '%s'.",
                        title_id,
                        chosen_title,
                    )

                payload = dict(response)
                payload.pop("title", None)

                model_entry = models_section.setdefault(model, {})
                if not isinstance(model_entry, dict):
                    model_entry = {}
                    models_section[model] = model_entry

                model_entry[prompt_hash] = payload
                written += 1

            if chosen_title:
                data["title"] = chosen_title

            ordered_data: Dict[str, Any] = {}

            ordered_data["title"] = data.get("title", "")
            ordered_data["llm_models"] = data.get("llm_models", {})

            for key, value in data.items():
                if key in {"title", "llm_models"}:
                    continue
                ordered_data[key] = value

            with tmp.open("w", encoding="utf-8") as f:
                json.dump(ordered_data, f, ensure_ascii=False, indent=2)
                f.write("\n")
            os.replace(tmp, path)

        if self.logger:
            self.logger.debug(
                "Saved %s batch with %s record(s).", title_id, len(records)
            )

        return len(records)

