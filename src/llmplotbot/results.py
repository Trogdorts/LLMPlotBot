"""Disk persistence helpers for generated model results."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Dict, Iterable, List, MutableMapping, Tuple


# ---------------------------------------------------------------------------
# File lock
# ---------------------------------------------------------------------------


class FileLockTimeout(TimeoutError):
    """Raised when the lock cannot be acquired within the configured timeout."""


@dataclass
class FileLock:
    path: Path
    timeout: float
    poll_interval: float
    stale_seconds: float

    def __post_init__(self) -> None:
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")
        if self.poll_interval <= 0:
            raise ValueError("poll_interval must be positive")
        if self.stale_seconds <= 0:
            raise ValueError("stale_seconds must be positive")
        self._fd: int | None = None

    def acquire(self) -> None:
        deadline = time.time() + self.timeout
        while True:
            try:
                self._fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(self._fd, str(os.getpid()).encode("utf-8"))
                return
            except FileExistsError:
                if self._is_stale():
                    self._clear_stale()
                    continue
                if time.time() >= deadline:
                    raise FileLockTimeout(f"Timed out waiting for lock {self.path}")
                time.sleep(self.poll_interval)

    def release(self) -> None:
        if self._fd is not None:
            try:
                os.close(self._fd)
            finally:
                self._fd = None
        try:
            os.unlink(self.path)
        except FileNotFoundError:
            pass

    def _is_stale(self) -> bool:
        try:
            stat = os.stat(self.path)
        except FileNotFoundError:
            return False
        age = time.time() - stat.st_mtime
        return age >= self.stale_seconds

    def _clear_stale(self) -> None:
        try:
            os.unlink(self.path)
        except FileNotFoundError:
            pass

    def __enter__(self) -> "FileLock":
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()


# ---------------------------------------------------------------------------
# Existing result inspection
# ---------------------------------------------------------------------------


class ResultStore:
    """Read-only helpers for inspecting generated result files."""

    def __init__(self, base_dir: str | Path) -> None:
        self.base = Path(base_dir)
        self.base.mkdir(parents=True, exist_ok=True)

    def record_path(self, identifier: str) -> Path:
        return self.base / f"{identifier}.json"

    def load_record(self, identifier: str) -> Dict[str, object] | None:
        path = self.record_path(identifier)
        if not path.exists():
            return None
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception:
            return None
        return data if isinstance(data, MutableMapping) else None

    def has_entry(self, identifier: str, model: str, prompt_hash: str) -> bool:
        record = self.load_record(identifier)
        if not record:
            return False
        models = record.get("llm_models")
        if not isinstance(models, MutableMapping):
            return False
        payload = models.get(model)
        if not isinstance(payload, MutableMapping):
            return False
        return prompt_hash in payload


# ---------------------------------------------------------------------------
# Result writer
# ---------------------------------------------------------------------------


class ResultWriter:
    """Persist model responses to disk with optional batching."""

    def __init__(
        self,
        base_dir: str | Path,
        *,
        strategy: str = "immediate",
        flush_interval: int = 25,
        flush_seconds: float = 5.0,
        retry_limit: int = 3,
        lock_timeout: float = 10.0,
        lock_poll_interval: float = 0.1,
        lock_stale_seconds: float = 300.0,
        logger=None,
    ) -> None:
        self.base = Path(base_dir)
        self.base.mkdir(parents=True, exist_ok=True)
        self.strategy = strategy.lower()
        self.flush_interval = max(1, int(flush_interval))
        self.flush_seconds = max(0.1, float(flush_seconds))
        self.retry_limit = max(1, int(retry_limit))
        self.lock_timeout = float(lock_timeout)
        self.lock_poll_interval = float(lock_poll_interval)
        self.lock_stale_seconds = float(lock_stale_seconds)
        self.logger = logger
        self._buffer: List[Tuple[str, str, str, Dict[str, object]]] = []
        self._last_flush = time.time()
        self._lock = Lock()

    def write(self, identifier: str, model: str, prompt_hash: str, payload: Dict[str, object]) -> None:
        with self._lock:
            self._buffer.append((identifier, model, prompt_hash, dict(payload)))
            if self.strategy == "immediate":
                self._flush_locked(force=True)
                return
            should_flush = len(self._buffer) >= self.flush_interval or (
                time.time() - self._last_flush
            ) >= self.flush_seconds
            if should_flush:
                self._flush_locked(force=True)

    def flush(self) -> None:
        with self._lock:
            self._flush_locked(force=True)

    # ------------------------------------------------------------------
    def _flush_locked(self, *, force: bool = False) -> None:
        if not self._buffer:
            return
        if not force and self.strategy != "immediate":
            elapsed = time.time() - self._last_flush
            if elapsed < self.flush_seconds and len(self._buffer) < self.flush_interval:
                return
        pending = list(self._buffer)
        self._buffer.clear()
        self._last_flush = time.time()
        failures: List[Tuple[str, str, str, Dict[str, object]]] = []

        grouped: Dict[str, List[Tuple[str, str, str, Dict[str, object]]]] = {}
        for record in pending:
            grouped.setdefault(record[0], []).append(record)

        for identifier, records in grouped.items():
            try:
                self._persist(identifier, records)
            except Exception as exc:  # pragma: no cover - defensive
                if self.logger:
                    self.logger.error("Failed to persist %s: %s", identifier, exc, exc_info=True)
                failures.extend(records)

        if failures:
            self._buffer.extend(failures)

    # ------------------------------------------------------------------
    def _persist(self, identifier: str, records: Iterable[Tuple[str, str, str, Dict[str, object]]]) -> None:
        path = self.base / f"{identifier}.json"
        tmp = path.with_suffix(".tmp")
        lock_path = path.with_suffix(".lock")
        lock = FileLock(lock_path, self.lock_timeout, self.lock_poll_interval, self.lock_stale_seconds)

        attempts = self.retry_limit
        while attempts:
            attempts -= 1
            try:
                with lock:
                    data = self._read_existing(path)
                    written = self._merge(data, records)
                    if written:
                        self._write_atomic(path, tmp, data)
                return
            except FileLockTimeout:
                if attempts == 0:
                    raise
                time.sleep(min(2.0, self.lock_poll_interval * 5))
            except Exception:
                if attempts == 0:
                    raise
                time.sleep(min(2.0, self.lock_poll_interval * 5))

    # ------------------------------------------------------------------
    def _read_existing(self, path: Path) -> Dict[str, object]:
        if not path.exists():
            return {"llm_models": {}}
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception:
            return {"llm_models": {}}
        if not isinstance(data, MutableMapping):
            return {"llm_models": {}}
        if "llm_models" not in data or not isinstance(data["llm_models"], MutableMapping):
            data["llm_models"] = {}
        return data

    def _merge(
        self,
        data: Dict[str, object],
        records: Iterable[Tuple[str, str, str, Dict[str, object]]],
    ) -> bool:
        changed = False
        models = data.setdefault("llm_models", {})
        for identifier, model, prompt_hash, payload in records:
            entry = models.setdefault(model, {})
            if not isinstance(entry, MutableMapping):
                entry = {}
                models[model] = entry
            cleaned = dict(payload)
            title = cleaned.pop("title", None)
            if title:
                data.setdefault("title", title)
            entry[prompt_hash] = cleaned
            changed = True
        return changed

    def _write_atomic(self, path: Path, tmp: Path, data: Dict[str, object]) -> None:
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tmp.replace(path)


__all__ = ["ResultStore", "ResultWriter", "FileLock", "FileLockTimeout"]
