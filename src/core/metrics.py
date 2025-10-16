"""Utility dataclasses for recording task runner metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict


@dataclass
class ModelMetrics:
    """Accumulates runtime information for a single model's queue."""

    queued: int = 0
    processed: int = 0
    success: int = 0
    failed: int = 0
    duration_sum: float = 0.0
    max_duration: float = 0.0
    total_attempts: int = 0
    max_attempts: int = 0
    retry_tasks: int = 0
    queue_duration: float = 0.0

    def record_task(self, *, success: bool, attempts: int, duration: float) -> None:
        """Update counters after a task completes."""

        self.processed += 1
        if success:
            self.success += 1
        else:
            self.failed += 1
        self.duration_sum += duration
        self.max_duration = max(self.max_duration, duration)
        self.total_attempts += attempts
        self.max_attempts = max(self.max_attempts, attempts)
        if attempts > 1:
            self.retry_tasks += 1

    def record_queue_duration(self, duration: float) -> None:
        """Persist the total time spent on the model queue."""

        self.queue_duration = duration

    def to_dict(self) -> Dict[str, Any]:
        """Return a serialisable snapshot for logging."""

        return asdict(self)


@dataclass
class RunnerMetrics:
    """Aggregate counters for a full task runner invocation."""

    total_tasks: int
    processed: int = 0
    success: int = 0
    failed: int = 0
    total_duration: float = 0.0
    max_duration: float = 0.0
    total_attempts: int = 0
    max_attempts: int = 0
    retry_tasks: int = 0
    start_time: float | None = None
    end_time: float | None = None
    per_model: Dict[str, ModelMetrics] = field(default_factory=dict)

    def ensure_model(self, model: str, queued: int = 0) -> ModelMetrics:
        """Return the metrics bucket for ``model`` creating it if required."""

        metrics = self.per_model.get(model)
        if metrics is None:
            metrics = ModelMetrics(queued=queued)
            self.per_model[model] = metrics
        else:
            metrics.queued = max(metrics.queued, queued)
        return metrics

    def record_task(
        self,
        *,
        model: str,
        success: bool,
        attempts: int,
        duration: float,
    ) -> None:
        """Update totals for the runner and the specific model."""

        self.processed += 1
        if success:
            self.success += 1
        else:
            self.failed += 1
        self.total_duration += duration
        self.max_duration = max(self.max_duration, duration)
        self.total_attempts += attempts
        self.max_attempts = max(self.max_attempts, attempts)
        if attempts > 1:
            self.retry_tasks += 1

        self.ensure_model(model).record_task(
            success=success, attempts=attempts, duration=duration
        )

    def record_queue_duration(self, *, model: str, duration: float) -> None:
        """Record the elapsed time for an individual model queue."""

        self.ensure_model(model).record_queue_duration(duration)

    def snapshot(self) -> Dict[str, Any]:
        """Return a copy of the metrics suitable for logging or testing."""

        return {
            "total_tasks": self.total_tasks,
            "processed": self.processed,
            "success": self.success,
            "failed": self.failed,
            "total_duration": self.total_duration,
            "max_duration": self.max_duration,
            "total_attempts": self.total_attempts,
            "max_attempts": self.max_attempts,
            "retry_tasks": self.retry_tasks,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "per_model": {model: metrics.to_dict() for model, metrics in self.per_model.items()},
        }


__all__ = ["ModelMetrics", "RunnerMetrics"]

