"""In-memory cache for per-task runtime statistics."""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

try:  # Optional dependency for richer CPU metrics
    import psutil  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    psutil = None

try:  # Optional dependency for GPU sampling
    import GPUtil  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    GPUtil = None


@dataclass(slots=True)
class MetricRecord:
    """Represents a single task execution metric entry."""

    task_id: str
    model: str
    latency: float
    success: bool
    attempts: int
    retries: int
    timestamp: float
    cpu_percent: Optional[float]
    gpu_utilization: Optional[float]

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


class MetricsCollector:
    """Thread-safe collector that buffers metrics until flushed."""

    def __init__(self) -> None:
        self._records: List[MetricRecord] = []
        self._lock = threading.Lock()
        self._start_time = time.time()
        self._total_recorded = 0
        self._finalized = False

    # ------------------------------------------------------------------
    @staticmethod
    def _sample_cpu_percent() -> Optional[float]:
        if psutil is not None:
            try:
                return float(psutil.cpu_percent(interval=None))
            except Exception:  # pragma: no cover - defensive
                return None
        # Fallback using load average if psutil is unavailable
        try:
            load1, _load5, _load15 = os.getloadavg()
            cpu_count = os.cpu_count() or 1
            return min(100.0, max(0.0, load1 / cpu_count * 100.0))
        except Exception:  # pragma: no cover - platform specific
            return None

    @staticmethod
    def _sample_gpu_utilization() -> Optional[float]:
        if GPUtil is None:  # pragma: no cover - optional dependency
            return None
        try:
            gpus = GPUtil.getGPUs()
        except Exception:  # pragma: no cover - GPU library may fail
            return None
        if not gpus:
            return None
        values = [gpu.load * 100.0 for gpu in gpus]
        return sum(values) / len(values)

    # ------------------------------------------------------------------
    def record_task(
        self,
        *,
        task_id: str,
        model: str,
        latency: float,
        success: bool,
        attempts: int,
    ) -> int:
        """Store a metric entry and return the cumulative count."""

        retries = max(0, attempts - 1)
        cpu_percent = self._sample_cpu_percent()
        gpu_utilization = self._sample_gpu_utilization()
        record = MetricRecord(
            task_id=task_id,
            model=model,
            latency=latency,
            success=success,
            attempts=attempts,
            retries=retries,
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            gpu_utilization=gpu_utilization,
        )

        with self._lock:
            if self._finalized:
                return self._total_recorded
            self._records.append(record)
            self._total_recorded += 1
            return self._total_recorded

    # ------------------------------------------------------------------
    def export(self, *, clear: bool = False) -> List[Dict[str, object]]:
        """Return a copy of cached metrics, optionally clearing the cache."""

        with self._lock:
            data = [record.to_dict() for record in self._records]
            if clear:
                self._records.clear()
            return data

    def flush(self) -> List[Dict[str, object]]:
        """Return and clear all cached metrics."""

        return self.export(clear=True)

    # ------------------------------------------------------------------
    @property
    def total_records(self) -> int:
        return self._total_recorded

    @property
    def start_time(self) -> float:
        return self._start_time

    # ------------------------------------------------------------------
    def finalize(self) -> None:
        with self._lock:
            self._finalized = True


__all__ = ["MetricRecord", "MetricsCollector"]
