"""Checkpoint persistence for recovery and auditing."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


@dataclass
class CheckpointState:
    last_job_id: str | None
    total_completed: int
    total_failed: int
    pending: int
    timestamp: float


class CheckpointManager:
    def __init__(
        self,
        checkpoint_dir: str | Path,
        *,
        interval_seconds: float,
        jobs_per_checkpoint: int,
        logger,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.interval_seconds = interval_seconds
        self.jobs_per_checkpoint = jobs_per_checkpoint
        self.logger = logger
        self._last_checkpoint = time.monotonic()
        self._processed_since_checkpoint = 0

    def maybe_checkpoint(
        self,
        state: CheckpointState,
    ) -> None:
        self._processed_since_checkpoint += 1
        elapsed = time.monotonic() - self._last_checkpoint
        if (
            self._processed_since_checkpoint >= self.jobs_per_checkpoint
            or elapsed >= self.interval_seconds
        ):
            self._write_checkpoint(state)
            self._processed_since_checkpoint = 0
            self._last_checkpoint = time.monotonic()

    def force_checkpoint(self, state: CheckpointState) -> None:
        self._write_checkpoint(state)
        self._processed_since_checkpoint = 0
        self._last_checkpoint = time.monotonic()

    def _write_checkpoint(self, state: CheckpointState) -> None:
        payload: Dict[str, Any] = {
            "last_job_id": state.last_job_id,
            "total_completed": state.total_completed,
            "total_failed": state.total_failed,
            "pending": state.pending,
            "timestamp": state.timestamp,
        }
        timestamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime(state.timestamp))
        path = self.checkpoint_dir / f"checkpoint-{timestamp}.json"
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        self.logger.debug("Checkpoint written to %s", path)


__all__ = ["CheckpointManager", "CheckpointState"]
