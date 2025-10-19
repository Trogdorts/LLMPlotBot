"""Aggregates runtime metrics and writes structured reports."""

from __future__ import annotations

import asyncio
import json
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping


@dataclass
class MetricSnapshot:
    total_jobs: int
    success: int
    failure: int
    retries: int
    avg_latency: float
    avg_tokens: float


class MetricsManager:
    def __init__(
        self,
        metrics_dir: str | Path,
        *,
        report_interval: float,
        include_system: bool,
        logger,
    ) -> None:
        self.metrics_dir = Path(metrics_dir)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.metrics_dir / "metrics.jsonl"
        self.report_interval = report_interval
        self.include_system = include_system
        self.logger = logger
        self._queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()
        self._stop = asyncio.Event()
        self._model_stats: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self._job_counts = defaultdict(float)
        self._last_flush = time.monotonic()

    # ------------------------------------------------------------------
    def record_success(self, model: str, *, elapsed: float, tokens: int) -> None:
        self._queue.put_nowait(
            {
                "type": "success",
                "model": model,
                "elapsed": float(elapsed),
                "tokens": int(tokens),
            }
        )

    def record_failure(self, model: str) -> None:
        self._queue.put_nowait({"type": "failure", "model": model})

    def record_retry(self, model: str) -> None:
        self._queue.put_nowait({"type": "retry", "model": model})

    def record_system_stats(self, stats: Mapping[str, Any]) -> None:
        if not self.include_system:
            return
        self._queue.put_nowait({"type": "system", "payload": dict(stats)})

    # ------------------------------------------------------------------
    async def run(self) -> None:
        while not self._stop.is_set():
            try:
                event = await asyncio.wait_for(self._queue.get(), timeout=self.report_interval)
            except asyncio.TimeoutError:
                self._flush()
                continue
            self._apply_event(event)
        while not self._queue.empty():
            self._apply_event(await self._queue.get())
        self._flush(final=True)

    def stop(self) -> None:
        self._stop.set()

    # ------------------------------------------------------------------
    def summary(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {"models": {}}
        for model, stats in self._model_stats.items():
            success = stats.get("success", 0)
            failure = stats.get("failure", 0)
            retries = stats.get("retries", 0)
            elapsed = stats.get("elapsed", 0.0)
            tokens = stats.get("tokens", 0.0)
            total = success + failure
            avg_elapsed = elapsed / success if success else 0.0
            avg_tokens = tokens / success if success else 0.0
            result["models"][model] = {
                "success": int(success),
                "failure": int(failure),
                "retries": int(retries),
                "avg_elapsed": avg_elapsed,
                "avg_tokens": avg_tokens,
                "total": int(total),
            }
        result["timestamp"] = time.time()
        return result

    # ------------------------------------------------------------------
    def _apply_event(self, event: Mapping[str, Any]) -> None:
        event_type = event.get("type")
        model = event.get("model")
        if event_type == "success" and model:
            stats = self._model_stats[model]
            stats["success"] += 1
            stats["elapsed"] += float(event.get("elapsed", 0.0))
            stats["tokens"] += float(event.get("tokens", 0))
        elif event_type == "failure" and model:
            self._model_stats[model]["failure"] += 1
        elif event_type == "retry" and model:
            self._model_stats[model]["retries"] += 1
        elif event_type == "system":
            payload = event.get("payload")
            if isinstance(payload, Mapping):
                self._append_json({"type": "system", **payload})
        self._maybe_flush()

    def _maybe_flush(self) -> None:
        if time.monotonic() - self._last_flush >= self.report_interval:
            self._flush()

    def _flush(self, *, final: bool = False) -> None:
        snapshot = self.summary()
        snapshot["final"] = final
        self._append_json(snapshot)
        self._last_flush = time.monotonic()
        self.logger.debug("Metrics snapshot written")

    def _append_json(self, payload: Mapping[str, Any]) -> None:
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


__all__ = ["MetricsManager", "MetricSnapshot"]
