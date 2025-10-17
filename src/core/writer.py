"""ResultWriter with configurable persistence strategy and file locking."""

from __future__ import annotations

import json
import os
import threading
import time
<<<<<<< HEAD
from typing import Any, Dict, List, Tuple
=======
from typing import Any, Dict, List, Optional, Tuple, Mapping, Sequence

>>>>>>> origin/main
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
        logger=None,
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
    def write(self, id: str, model: str, prompt_hash: str, response: Dict[str, Any]):
        """Queue a single write and flush based on the configured strategy."""

<<<<<<< HEAD
        with self.lock:
            self.buffer.append((id, model, prompt_hash, response))
            if self.strategy == "immediate":
                self._flush_locked(force=True)
                return
=======
        payload = dict(response)
        title = str(payload.pop("title", "") or "")
        self._persist_records_with_retry(
            id,
            [(model, prompt_hash, payload, title)],
            debug_label=f"model {model}",
        )

    # ------------------------------------------------------------------
    def write_many(
        self,
        id: str,
        records: Sequence[Tuple[str, str, Mapping[str, Any]]],
    ) -> None:
        """Persist multiple responses for the same identifier in one operation."""

        if not records:
            return

        prepared: List[Tuple[str, str, Dict[str, Any], str]] = []
        for model, prompt_hash, response in records:
            payload = dict(response)
            title = str(payload.pop("title", "") or "")
            prepared.append((model, prompt_hash, payload, title))

        self._persist_records_with_retry(
            id,
            prepared,
            debug_label=f"batch of {len(prepared)} record(s)",
        )

    # ------------------------------------------------------------------
    def flush(self):
        """Compatibility shim for previous API; immediate writes need no flushing."""
        return

    # ------------------------------------------------------------------
    def _persist_records_with_retry(
        self,
        id: str,
        prepared_records: Sequence[Tuple[str, str, Dict[str, Any], str]],
        *,
        debug_label: str,
    ) -> None:
        attempts_remaining = self.retry_limit
        backoff_step = 0
>>>>>>> origin/main

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
        """Flush queued records in memory to disk safely."""
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

        for id, records in grouped.items():
            try:
                written = self._write_batch(id, records)
            except Exception as exc:  # defensive: avoid breaking flush loop
                failed.extend(records)
                if self.logger:
                    self.logger.error(
                        "Failed to persist batch for %s: %s", id, exc, exc_info=True
                    )
                continue

            if written:
                success_count += written
            else:
                failed.extend(records)

        if failed:
            # Requeue failed entries for another attempt on the next flush.
            self.buffer = failed + self.buffer

        if self.logger and success_count:
            self.logger.debug("Flushed %s record(s) successfully.", success_count)

    # ------------------------------------------------------------------
    def _write_batch(
        self, id: str, records: List[Tuple[str, str, str, Dict[str, Any]]]
    ) -> int:
        """Persist a batch of responses for the same identifier atomically."""

        attempts_remaining = self.flush_retry_limit
        while attempts_remaining > 0:
            attempts_remaining -= 1
            try:
<<<<<<< HEAD
                return self._persist_records(id, records)
=======
                written = self._persist_records(id, prepared_records)
                if self.logger:
                    self.logger.debug(
                        "Saved %s with %s (wrote %s record(s)).",
                        id,
                        debug_label,
                        written,
                    )
                return
>>>>>>> origin/main
            except FileLockTimeout as exc:
                if self.logger:
                    self.logger.warning(str(exc))
                if attempts_remaining == 0:
                    return 0
                # Small backoff before retrying the same file.
                time.sleep(
                    min(0.5 * (self.flush_retry_limit - attempts_remaining), 2.0)
                )
            except Exception:
                if attempts_remaining == 0:
                    raise
                if self.logger:
                    self.logger.warning(
                        "Retrying batch write for %s due to unexpected error.",
                        id,
                        exc_info=True,
                    )
                time.sleep(
                    min(0.5 * (self.flush_retry_limit - attempts_remaining), 2.0)
                )
        return 0

    # ------------------------------------------------------------------
    def _persist_records(
<<<<<<< HEAD
        self, id: str, records: List[Tuple[str, str, str, Dict[str, Any]]]
    ) -> int:
        """Perform the atomic JSON write with file lock and merge existing data."""
=======
        self, id: str, prepared_records: Sequence[Tuple[str, str, Dict[str, Any], str]]
    ) -> int:
>>>>>>> origin/main
        path = self.base / f"{id}.json"
        tmp = path.with_name(path.name + ".tmp")
        lock_path = path.with_name(path.name + ".lock")

<<<<<<< HEAD
        with FileLock(
            str(lock_path),
            timeout=self.lock_timeout,
            poll_interval=self.lock_poll_interval,
            stale_seconds=self.lock_stale_seconds,
        ):
=======
        def _load_existing(
            target: Path,
        ) -> Tuple[Dict[str, Any], str, Optional[float]]:
>>>>>>> origin/main
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

<<<<<<< HEAD
            for _, model, prompt_hash, response in records:
                title = response.get("title", "")
=======
            for record_model, record_prompt_hash, payload, title in prepared_records:
>>>>>>> origin/main
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
                        id,
                        chosen_title,
                    )

<<<<<<< HEAD
                payload = dict(response)
                payload.pop("title", None)

                model_entry = models_section.setdefault(model, {})
=======
                model_entry = models_section.setdefault(record_model, {})
>>>>>>> origin/main
                if not isinstance(model_entry, dict):
                    model_entry = {}
                    models_section[record_model] = model_entry

<<<<<<< HEAD
                model_entry[prompt_hash] = payload
                written += 1
=======
                existing_payload = model_entry.get(record_prompt_hash)
                if existing_payload != payload:
                    model_entry[record_prompt_hash] = payload
                    written += 1
>>>>>>> origin/main

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

<<<<<<< HEAD
        if self.logger:
            self.logger.debug("Saved %s batch with %s record(s).", id, len(records))
=======
            with FileLock(
                str(lock_path),
                timeout=self.lock_timeout,
                poll_interval=self.lock_poll_interval,
                stale_seconds=self.lock_stale_seconds,
            ):
                current_mtime: Optional[float]
                try:
                    current_mtime = path.stat().st_mtime
                except FileNotFoundError:
                    current_mtime = None

                if current_mtime != last_mtime:
                    continue

                try:
                    tmp.unlink()
                except FileNotFoundError:
                    pass

                with tmp.open("w", encoding="utf-8") as handle:
                    handle.write(serialized)

                os.replace(tmp, path)

            if self.logger:
                self.logger.debug(
                    "Saved %s batch with %s record(s).", id, written
                )

            return written
>>>>>>> origin/main

        return len(records)
