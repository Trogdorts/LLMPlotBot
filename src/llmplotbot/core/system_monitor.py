"""Collects system metrics such as CPU, memory, and GPU utilisation."""

from __future__ import annotations

import asyncio
import shutil
import subprocess
from typing import Any, Dict

try:  # pragma: no cover - optional dependency
    import psutil  # type: ignore
except Exception:  # pragma: no cover - psutil may be unavailable
    psutil = None


class SystemMonitor:
    def __init__(self, *, interval: float, metrics, logger) -> None:
        self.interval = interval
        self.metrics = metrics
        self.logger = logger
        self._running = False
        self._has_nvidia_smi = shutil.which("nvidia-smi") is not None

    async def run(self, stop_event: asyncio.Event) -> None:
        self._running = True
        while not stop_event.is_set():
            try:
                stats = self._collect_stats()
                if stats:
                    self.metrics.record_system_stats(stats)
            except Exception as exc:  # pragma: no cover - defensive
                self.logger.debug("System monitor failed: %s", exc)
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=self.interval)
            except asyncio.TimeoutError:
                continue
        self._running = False

    def _collect_stats(self) -> Dict[str, Any]:
        stats: Dict[str, Any] = {}
        if psutil:
            stats["cpu_percent"] = psutil.cpu_percent(interval=None)
            memory = psutil.virtual_memory()
            stats["memory_percent"] = memory.percent
            stats["memory_used_mb"] = round(memory.used / (1024 * 1024), 2)
            gpu_stats = self._gpu_stats()
            if gpu_stats:
                stats.update(gpu_stats)
        return stats

    def _gpu_stats(self) -> Dict[str, Any]:
        if not self._has_nvidia_smi:
            return {}
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
        except Exception:  # pragma: no cover - GPU optional
            return {}
        line = result.stdout.strip().splitlines()[0]
        values = [value.strip() for value in line.split(",")]
        stats: Dict[str, Any] = {}
        if len(values) >= 2:
            stats["gpu_utilization_percent"] = float(values[0])
            stats["gpu_memory_used_mb"] = float(values[1])
        return stats


__all__ = ["SystemMonitor"]
