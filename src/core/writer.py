"""ResultWriter that persists responses immediately using file locks."""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from pathlib import Path

from src.util.file_lock import FileLock, FileLockTimeout


class ResultWriter:
    """Persist responses immediately with retry-aware file locking."""

    def __init__(
        self,
        base_dir: str,
        *,
        retry_limit: int = 3,
        lock_timeout: float = 10.0,
        lock_poll_interval: float = 0.1,
        lock_stale_seconds: float = 300.0,
        logger=None,
    ) -> None:
        self.base = Path(base_dir)
        self.logger = logger
        self.retry_limit = max(1, int(retry_limit or 1))
        self.lock_timeout = max(0.1, lock_timeout)
        self.lock_poll_interval = max(0.01, lock_poll_interval)
        self.lock_stale_seconds = max(1.0, lock_stale_seconds)
        self.base.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def write(self, id: str, model: str, prompt_hash: str, response: Dict[str, Any]):
        """Persist a single response with retry handling for lock contention."""

        attempts_remaining = self.retry_limit
        backoff_step = 0

        while attempts_remaining > 0:
            attempts_remaining -= 1
            try:
                self._persist_record(id, model, prompt_hash, response)
                if self.logger:
                    self.logger.debug("Saved %s for model %s.", id, model)
                return
            except FileLockTimeout as exc:
                if self.logger:
                    self.logger.warning(str(exc))
                if attempts_remaining == 0:
                    break
            except Exception:
                if attempts_remaining == 0:
                    raise
                if self.logger:
                    self.logger.warning(
                        "Retrying write for %s due to unexpected error.",
                        id,
                        exc_info=True,
                    )

            backoff_step += 1
            time.sleep(min(0.5 * backoff_step, 2.0))

        if self.logger:
            self.logger.error(
                "Failed to persist result for %s after %s attempt(s).",
                id,
                self.retry_limit,
            )

    # ------------------------------------------------------------------
    def flush(self):
        """Compatibility shim for previous API; immediate writes need no flushing."""
        return

    # ------------------------------------------------------------------
    def _persist_record(
        self, id: str, model: str, prompt_hash: str, response: Dict[str, Any]
    ) -> None:
        path = self.base / f"{id}.json"
        tmp = path.with_name(path.name + ".tmp")
        lock_path = path.with_name(path.name + ".lock")

        prepared_records: List[Tuple[str, str, Dict[str, Any], str]] = []
        for _, model, prompt_hash, response in records:
            payload = dict(response)
            title = str(payload.pop("title", "") or "")
            prepared_records.append((model, prompt_hash, payload, title))

        def _load_existing(
            target: Path,
        ) -> Tuple[Dict[str, Any], str, Optional[float]]:
            data: Dict[str, Any] = {}
            mtime: Optional[float] = None

            try:
                stat_result = target.stat()
                mtime = stat_result.st_mtime
                with target.open("r", encoding="utf-8") as handle:
                    loaded = json.load(handle)
                if isinstance(loaded, dict):
                    data = loaded
            except FileNotFoundError:
                data = {}
                mtime = None
            except Exception:
                data = {}

            if not isinstance(data, dict):
                data = {}

            existing_title = str(data.get("title", "") or "")
            models_section = data.get("llm_models")
            if not isinstance(models_section, dict):
                models_section = {}
                data["llm_models"] = models_section

            return data, existing_title, mtime

        while True:
            data, existing_title, last_mtime = _load_existing(path)

            models_section = data.setdefault("llm_models", {})
            written = 0
            chosen_title = existing_title

            for model, prompt_hash, payload, title in prepared_records:
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

                model_entry = models_section.setdefault(model, {})
                if not isinstance(model_entry, dict):
                    model_entry = {}
                    models_section[model] = model_entry

            model_entry[prompt_hash] = payload

            if chosen_title:
                data["title"] = chosen_title

            ordered_data: Dict[str, Any] = {
                "title": data.get("title", ""),
                "llm_models": data.get("llm_models", {}),
            }

            for key, value in data.items():
                if key in {"title", "llm_models"}:
                    continue
                ordered_data[key] = value

            serialized = json.dumps(ordered_data, ensure_ascii=False, indent=2) + "\n"

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
                    "Saved %s batch with %s record(s).", id, len(records)
                )

            return written

