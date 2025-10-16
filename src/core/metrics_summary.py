"""Periodic and final summary reporting for collected metrics."""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional

from .metrics_collector import MetricsCollector


class MetricsSummaryReporter:
    """Aggregates metrics and persists summary snapshots."""

    def __init__(
        self,
        collector: MetricsCollector,
        logger,
        log_dir: str,
        *,
        summary_every_tasks: int = 0,
        summary_every_seconds: float = 0.0,
    ) -> None:
        self.collector = collector
        self.logger = logger
        self.summary_every_tasks = max(0, int(summary_every_tasks))
        self.summary_every_seconds = max(0.0, float(summary_every_seconds))
        self._summary_dir = Path(log_dir) / "metrics"
        self._summary_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_report_count = 0
        self._finalized = False
        self._signal_snapshot_emitted = False
        self._final_summary: Optional[Dict[str, object]] = None

    # ------------------------------------------------------------------
    def start(self) -> None:
        if self.summary_every_seconds <= 0.0 or self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run_periodic,
            name="MetricsSummary",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread and thread.is_alive():
            thread.join()
        self._thread = None

    # ------------------------------------------------------------------
    def _run_periodic(self) -> None:
        while not self._stop_event.wait(self.summary_every_seconds):
            self.report(trigger="time_interval", reason="interval")

    # ------------------------------------------------------------------
    def maybe_report(self, total_records: int) -> None:
        if self._finalized or self.summary_every_tasks <= 0:
            return
        if total_records - self._last_report_count < self.summary_every_tasks:
            return
        self.report(trigger="task_interval", reason="task_count")

    # ------------------------------------------------------------------
    def report(self, *, trigger: str, reason: str) -> Optional[Dict[str, object]]:
        if self._finalized:
            return self._final_summary
        with self._lock:
            records = self.collector.export(clear=False)
            summary = self._build_summary(
                records,
                trigger=trigger,
                reason=reason,
                end_time=time.time(),
                include_raw=False,
            )
            if summary is None:
                return None
            self._last_report_count = len(records)
            self._write_summary_snapshot(summary)
            self._log_summary(summary)
            return summary

    # ------------------------------------------------------------------
    def handle_shutdown_signal(self) -> None:
        if self._finalized:
            return
        with self._lock:
            if self._signal_snapshot_emitted:
                return
            self.stop()
            records = self.collector.export(clear=False)
            summary = self._build_summary(
                records,
                trigger="shutdown_signal",
                reason="signal",
                end_time=time.time(),
                include_raw=False,
            )
            if summary is None:
                return
            self._last_report_count = len(records)
            self._write_summary_snapshot(summary)
            self._log_summary(summary)
            self._signal_snapshot_emitted = True

    # ------------------------------------------------------------------
    def finalize(
        self,
        *,
        reason: str,
        session_end_time: Optional[float] = None,
    ) -> Optional[Dict[str, object]]:
        with self._lock:
            if self._finalized:
                return self._final_summary
            self._finalized = True
            self.stop()
            records = self.collector.flush()
            self.collector.finalize()
            summary = self._build_summary(
                records,
                trigger="final",
                reason=reason,
                end_time=session_end_time or time.time(),
                include_raw=True,
            )
            if summary is None:
                self._final_summary = None
                return None
            self._last_report_count = len(records)
            self._write_metrics_dump(records, summary)
            self._write_summary_snapshot(summary)
            self._log_summary(summary)
            self._final_summary = summary
            return summary

    # ------------------------------------------------------------------
    def _build_summary(
        self,
        records: List[Dict[str, object]],
        *,
        trigger: str,
        reason: str,
        end_time: float,
        include_raw: bool,
    ) -> Optional[Dict[str, object]]:
        start_time = self.collector.start_time
        session_duration = max(0.0, end_time - start_time)
        total = len(records)
        success = sum(1 for record in records if record.get("success"))
        failed = total - success
        latencies = [float(record.get("latency", 0.0)) for record in records]
        attempts = [int(record.get("attempts", 0)) for record in records]
        retries = [int(record.get("retries", 0)) for record in records]
        cpu_values = [
            float(record.get("cpu_percent"))
            for record in records
            if record.get("cpu_percent") is not None
        ]
        gpu_values = [
            float(record.get("gpu_utilization"))
            for record in records
            if record.get("gpu_utilization") is not None
        ]

        average_latency = mean(latencies) if latencies else 0.0
        max_latency = max(latencies) if latencies else 0.0
        failure_rate = (failed / total * 100.0) if total else 0.0
        average_retries = mean(retries) if retries else 0.0
        average_attempts = mean(attempts) if attempts else 0.0
        max_attempts = max(attempts) if attempts else 0
        throughput_per_minute = (total / session_duration * 60.0) if session_duration else 0.0
        cpu_average = mean(cpu_values) if cpu_values else None
        gpu_summary: Optional[Dict[str, float]]
        if gpu_values:
            overall = mean(gpu_values)
            window = gpu_values[-min(len(gpu_values), 10) :]
            recent = mean(window)
            gpu_summary = {
                "overall_avg": overall,
                "recent_avg": recent,
                "delta": recent - overall,
            }
        else:
            gpu_summary = None

        timestamp = datetime.fromtimestamp(end_time, tz=timezone.utc).isoformat()
        start_iso = datetime.fromtimestamp(start_time, tz=timezone.utc).isoformat()

        per_model: Dict[str, Dict[str, object]] = {}
        if records:
            grouped: Dict[str, List[Dict[str, object]]] = {}
            for record in records:
                model_name = str(record.get("model") or "unknown")
                grouped.setdefault(model_name, []).append(record)

            for model_name, model_records in grouped.items():
                model_total = len(model_records)
                model_success = sum(1 for entry in model_records if entry.get("success"))
                model_failed = model_total - model_success
                model_latencies = [float(entry.get("latency", 0.0)) for entry in model_records]
                model_attempts = [int(entry.get("attempts", 0)) for entry in model_records]
                model_retries = [int(entry.get("retries", 0)) for entry in model_records]

                per_model[model_name] = {
                    "processed": model_total,
                    "success": model_success,
                    "failed": model_failed,
                    "failure_rate": (model_failed / model_total * 100.0)
                    if model_total
                    else 0.0,
                    "average_latency": mean(model_latencies) if model_latencies else 0.0,
                    "max_latency": max(model_latencies) if model_latencies else 0.0,
                    "average_attempts": mean(model_attempts) if model_attempts else 0.0,
                    "total_attempts": sum(model_attempts),
                    "max_attempts": max(model_attempts) if model_attempts else 0,
                    "average_retries": mean(model_retries) if model_retries else 0.0,
                    "total_retries": sum(model_retries),
                    "max_retries": max(model_retries) if model_retries else 0,
                    "throughput_per_minute": (
                        model_total / session_duration * 60.0
                        if session_duration
                        else 0.0
                    ),
                    "last_updated": datetime.fromtimestamp(
                        max(
                            (float(entry.get("timestamp", 0.0)) for entry in model_records),
                            default=end_time,
                        ),
                        tz=timezone.utc,
                    ).isoformat(),
                }

        summary: Dict[str, object] = {
            "trigger": trigger,
            "reason": reason,
            "generated_at": timestamp,
            "session_start": start_iso,
            "session_end": timestamp,
            "duration_seconds": session_duration,
            "total_processed": total,
            "success": success,
            "failed": failed,
            "failure_rate": failure_rate,
            "average_latency": average_latency,
            "max_latency": max_latency,
            "throughput_per_minute": throughput_per_minute,
            "average_retries": average_retries,
            "total_retries": sum(retries),
            "average_attempts": average_attempts,
            "total_attempts": sum(attempts),
            "max_attempts": max_attempts,
            "max_retries": max(retries) if retries else 0,
            "total_models": len(per_model),
            "per_model": per_model,
            "total_requests": sum(attempts),
            "payloads_received": success,
        }

        if cpu_average is not None:
            summary["cpu_percent_avg"] = cpu_average
        if gpu_summary is not None:
            summary["gpu_utilization"] = gpu_summary
        if include_raw:
            summary["raw_metrics_dump"] = None  # placeholder replaced later
        return summary

    # ------------------------------------------------------------------
    def _log_summary(self, summary: Dict[str, object]) -> None:
        message = (
            "Metrics summary (%s): processed=%s success=%s failed=%s "
            "avg_latency=%.2fs failure_rate=%.2f%% throughput=%.2f/min "
            "requests=%s retries=%s"
        )
        self.logger.info(
            message,
            summary.get("trigger"),
            summary.get("total_processed"),
            summary.get("success"),
            summary.get("failed"),
            summary.get("average_latency", 0.0),
            summary.get("failure_rate", 0.0),
            summary.get("throughput_per_minute", 0.0),
            summary.get("total_requests", 0),
            summary.get("total_retries", 0),
        )
        gpu = summary.get("gpu_utilization")
        if isinstance(gpu, dict):
            self.logger.debug(
                "GPU utilization trend: overall=%.2f recent=%.2f delta=%.2f",
                gpu.get("overall_avg", 0.0),
                gpu.get("recent_avg", 0.0),
                gpu.get("delta", 0.0),
            )
        per_model = summary.get("per_model")
        if isinstance(per_model, dict):
            for model_name, stats in sorted(per_model.items()):
                self.logger.debug(
                    (
                        "Model %s metrics: processed=%s success=%s failed=%s "
                        "avg_latency=%.2fs failure_rate=%.2f%% requests=%s retries=%s"
                    ),
                    model_name,
                    stats.get("processed", 0),
                    stats.get("success", 0),
                    stats.get("failed", 0),
                    stats.get("average_latency", 0.0),
                    stats.get("failure_rate", 0.0),
                    stats.get("total_attempts", 0),
                    stats.get("total_retries", 0),
                )

    # ------------------------------------------------------------------
    def _write_summary_snapshot(self, summary: Dict[str, object]) -> Path:
        date_suffix = datetime.now(timezone.utc).strftime("%Y%m%d")
        summary_path = self._summary_dir / f"summary_{date_suffix}.json"
        existing: List[Dict[str, object]] = []
        if summary_path.exists():
            try:
                with summary_path.open("r", encoding="utf-8") as handle:
                    existing = json.load(handle)
                    if not isinstance(existing, list):
                        existing = []
            except Exception:  # pragma: no cover - defensive parsing
                existing = []
        existing.append(summary)
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(existing, handle, indent=2)
        summary["summary_path"] = summary_path.as_posix()
        return summary_path

    def _write_metrics_dump(
        self,
        records: List[Dict[str, object]],
        summary: Dict[str, object],
    ) -> None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        dump_path = self._summary_dir / f"metrics_dump_{timestamp}.json"
        with dump_path.open("w", encoding="utf-8") as handle:
            json.dump(records, handle, indent=2)
        summary["raw_metrics_dump"] = dump_path.as_posix()


__all__ = ["MetricsSummaryReporter"]
