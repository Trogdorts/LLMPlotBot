"""Runtime metrics aggregation for LLMPlotBot."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Dict, List


@dataclass
class MetricRecord:
    model: str
    elapsed: float
    items: int
    status: str


class MetricsTracker:
    def __init__(self) -> None:
        self.records: List[MetricRecord] = []

    def record(self, model: str, elapsed: float, items: int, *, status: str) -> None:
        self.records.append(MetricRecord(model=model, elapsed=float(elapsed), items=int(items), status=status))

    def summary(self) -> Dict[str, object]:
        total = len(self.records)
        per_model: Dict[str, Dict[str, object]] = {}
        for record in self.records:
            model_bucket = per_model.setdefault(record.model, {"success": 0, "failure": 0, "elapsed": []})
            if record.status == "success":
                model_bucket["success"] += 1
            else:
                model_bucket["failure"] += 1
            model_bucket["elapsed"].append(record.elapsed)
        result = {"total_batches": total, "models": {}}
        for model, data in per_model.items():
            elapsed_values = data["elapsed"]
            result["models"][model] = {
                "success": data["success"],
                "failure": data["failure"],
                "avg_elapsed": mean(elapsed_values) if elapsed_values else 0.0,
            }
        return result


__all__ = ["MetricsTracker", "MetricRecord"]
