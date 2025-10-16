"""Task runner that orchestrates per-model execution (now concurrent)."""

import logging
import threading
import time
from typing import Dict, List, Optional, Tuple

from .metrics import RunnerMetrics
from .model_connector import ModelConnector
from .task import Task
from .writer import ResultWriter


class TaskRunner:
    """Process tasks per model with retries using persistent connector sessions."""

    def __init__(
        self,
        tasks_by_model: Dict[str, List[Task]],
        connectors: Dict[str, ModelConnector],
        writer: ResultWriter,
        retry_limit: int,
        shutdown_event: threading.Event,
        logger: logging.Logger,
        *,
        summary_interval: float = 0.0,
        model_aliases: Optional[Dict[str, str]] = None,
    ) -> None:
        self.tasks_by_model = tasks_by_model
        self.connectors = connectors
        self.writer = writer
        self.retry_limit = max(0, retry_limit)
        self.shutdown_event = shutdown_event
        self.logger = logger
        self.model_aliases = model_aliases or {}
        self._summary_interval = max(0.0, float(summary_interval))
        self._summary_stop = threading.Event()
        self._summary_thread: threading.Thread | None = None

        total_tasks = sum(len(tasks) for tasks in tasks_by_model.values())
        self._metrics = RunnerMetrics(total_tasks=total_tasks)
        queued_counts: Dict[str, int] = {}
        for model in connectors:
            alias = self.model_aliases.get(model, model)
            count = len(tasks_by_model.get(model, []))
            queued_counts[alias] = queued_counts.get(alias, 0) + count

        for alias, queued in queued_counts.items():
            self._metrics.ensure_model(alias, queued=queued)

        self._stats_lock = threading.Lock()

        def _make_model_stats(queued: int) -> Dict[str, object]:
            return {
                "queued": queued,
                "processed": 0,
                "success": 0,
                "failed": 0,
                "duration_sum": 0.0,
                "max_duration": 0.0,
                "total_attempts": 0,
                "max_attempts": 0,
                "retry_tasks": 0,
                "queue_duration": 0.0,
            }

        per_model = {
            alias: _make_model_stats(queued) for alias, queued in queued_counts.items()
        }
        for model in connectors:
            alias = self.model_aliases.get(model, model)
            per_model.setdefault(alias, _make_model_stats(0))

        self._stats = {
            "total_tasks": total_tasks,
            "processed": 0,
            "success": 0,
            "failed": 0,
            "total_duration": 0.0,
            "max_duration": 0.0,
            "total_attempts": 0,
            "max_attempts": 0,
            "retry_tasks": 0,
            "start_time": None,
            "end_time": None,
            "per_model": per_model,
        }
        self._stats_lock = threading.Lock()

    # ------------------------------------------------------------------
    def run(self) -> None:
        """Run through all tasks for each model concurrently."""
        threads = []

        for model, tasks in self.tasks_by_model.items():
            if self.shutdown_event.is_set():
                break

            if not tasks:
                self.logger.debug("No tasks queued for model %s; skipping.", model)
                continue

            if model not in self.connectors:
                self.logger.error("No connector available for model %s", model)
                continue

            thread = threading.Thread(
                target=self._run_model_queue,
                name=f"TaskRunner-{model}",
                args=(model, tasks),
                daemon=False,
            )
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        if self.shutdown_event.is_set():
            self.logger.info("Task runner halted due to shutdown signal.")
        else:
            self.logger.info("Task runner completed all queued work.")

        self._log_summary()

    # ------------------------------------------------------------------
    def _run_model_queue(self, model: str, tasks: List[Task]) -> None:
        if self.shutdown_event.is_set():
            return

        if not tasks:
            self.logger.debug("No tasks queued for model %s; skipping.", model)
            return

        connector = self.connectors.get(model)
        if not connector:
            self.logger.error("No connector available for model %s", model)
            return

        queue_start = time.perf_counter()
        connector.start_session(
            tasks[0].prompt_dynamic,
            tasks[0].prompt_formatting,
        )

        try:
            self._process_model_tasks(connector, tasks)
        finally:
            connector.close_session()
            duration = time.perf_counter() - queue_start
            self._record_queue_duration(model, duration)

    # ------------------------------------------------------------------
    def _process_model_tasks(self, connector: ModelConnector, tasks: List[Task]) -> None:
        for task in tasks:
            if self.shutdown_event.is_set():
                self.logger.debug(
                    "Shutdown requested; abandoning remaining tasks for %s.",
                    connector.model,
                )
                return

            self.logger.info("Processing task id=%s model=%s", task.id, task.model)
            start = time.perf_counter()
            success, attempts = self._run_with_retries(connector, task)
            duration = time.perf_counter() - start
            self._record_task_metrics(connector.model, success, attempts, duration)

            if success:
                self.logger.info(
                    "Task id=%s processed successfully on attempt %s (%.2fs).",
                    task.id,
                    attempts,
                    duration,
                )
            else:
                if attempts == 0:
                    if self.shutdown_event.is_set():
                        self.logger.warning(
                            "Task id=%s was skipped due to shutdown before processing.",
                            task.id,
                        )
                    else:
                        self.logger.error(
                            "Task id=%s could not be processed because RETRY_LIMIT is 0.",
                            task.id,
                        )
                elif self.shutdown_event.is_set():
                    self.logger.warning(
                        "Task id=%s halted after %s attempt(s) due to shutdown.",
                        task.id,
                        attempts,
                    )
                else:
                    self.logger.error(
                        "Task id=%s failed after %s attempt(s); moving on.",
                        task.id,
                        attempts,
                    )

    # ------------------------------------------------------------------
    def _run_with_retries(self, connector: ModelConnector, task: Task) -> Tuple[bool, int]:
        attempt = 0
        while attempt < self.retry_limit and not self.shutdown_event.is_set():
            attempt += 1
            self.logger.debug(
                "Attempt %s for task id=%s model=%s",
                attempt,
                task.id,
                connector.model,
            )
            result = connector.send_headline(task.title)
            if result:
                result["title"] = task.title
                model_key = self.model_aliases.get(task.model, task.model)
                self.writer.write(task.id, model_key, task.prompt_hash, result)
                return True, attempt

            if attempt < self.retry_limit and not self.shutdown_event.is_set():
                connector.reinforce_compliance()
                self.logger.warning(
                    "Model %s returned invalid payload for task id=%s; retrying (%s/%s).",
                    connector.model,
                    task.id,
                    attempt,
                    self.retry_limit,
                )

        return False, attempt

    # ------------------------------------------------------------------
    def _record_task_metrics(self, model: str, success: bool, attempts: int, duration: float) -> None:
        with self._stats_lock:
            stats = self._stats
            stats["processed"] += 1
            if success:
                stats["success"] += 1
            else:
                stats["failed"] += 1
            stats["total_duration"] += duration
            stats["max_duration"] = max(stats["max_duration"], duration)
            stats["total_attempts"] += attempts
            stats["max_attempts"] = max(stats["max_attempts"], attempts)
            if attempts > 1:
                stats["retry_tasks"] += 1

            model_key = self.model_aliases.get(model, model)
            model_stats = stats["per_model"].setdefault(
                model_key,
                {
                    "queued": 0,
                    "processed": 0,
                    "success": 0,
                    "failed": 0,
                    "duration_sum": 0.0,
                    "max_duration": 0.0,
                    "total_attempts": 0,
                    "max_attempts": 0,
                    "retry_tasks": 0,
                    "queue_duration": 0.0,
                },
            )
            model_stats["processed"] += 1
            if success:
                model_stats["success"] += 1
            else:
                model_stats["failed"] += 1
            model_stats["duration_sum"] += duration
            model_stats["max_duration"] = max(model_stats["max_duration"], duration)
            model_stats["total_attempts"] += attempts
            model_stats["max_attempts"] = max(model_stats["max_attempts"], attempts)
            if attempts > 1:
                model_stats["retry_tasks"] += 1

    # ------------------------------------------------------------------
    def _record_queue_duration(self, model: str, duration: float) -> None:
        with self._stats_lock:
            model_key = self.model_aliases.get(model, model)
            model_stats = self._stats["per_model"].setdefault(
                model_key,
                {
                    "queued": 0,
                    "processed": 0,
                    "success": 0,
                    "failed": 0,
                    "duration_sum": 0.0,
                    "max_duration": 0.0,
                    "total_attempts": 0,
                    "max_attempts": 0,
                    "retry_tasks": 0,
                    "queue_duration": 0.0,
                },
            )
            model_stats["queue_duration"] = duration

    # ------------------------------------------------------------------
    def _log_summary(self) -> None:
        with self._stats_lock:
            stats = {
                **self._stats,
                "per_model": {model: dict(values) for model, values in self._stats["per_model"].items()},
            }

        total_processed = stats["processed"]
        total_time = None
        if stats["start_time"] is not None and stats["end_time"] is not None:
            total_time = max(0.0, stats["end_time"] - stats["start_time"])

        failure_rate = (stats["failed"] / total_processed * 100) if total_processed else 0.0

        self.logger.info(
            "Run summary: queued=%s processed=%s success=%s failed=%s failure_rate=%.2f%% total_time=%.2fs",
            stats["total_tasks"],
            total_processed,
            stats["success"],
            stats["failed"],
            failure_rate,
            total_time if total_time is not None else 0.0,
        )

        if total_processed:
            avg_duration = stats["total_duration"] / total_processed
            avg_attempts = stats["total_attempts"] / total_processed
            self.logger.info(
                "Task durations: avg=%.2fs max=%.2fs | Attempts: avg=%.2f max=%s retries=%s",
                avg_duration,
                stats["max_duration"],
                avg_attempts,
                stats["max_attempts"],
                stats["retry_tasks"],
            )

        for model, model_stats in sorted(stats["per_model"].items()):
            processed = model_stats.get("processed", 0)
            if not processed:
                continue
            model_failure_rate = (
                model_stats.get("failed", 0) / processed * 100 if processed else 0.0
            )
            avg_duration = model_stats["duration_sum"] / processed if processed else 0.0
            avg_attempts = model_stats["total_attempts"] / processed if processed else 0.0
            self.logger.info(
                "Model %s summary: queued=%s processed=%s success=%s failed=%s failure_rate=%.2f%% avg_duration=%.2fs max_duration=%.2fs avg_attempts=%.2f max_attempts=%s retries=%s queue_time=%.2fs",
                model,
                model_stats.get("queued", 0),
                processed,
                model_stats.get("success", 0),
                model_stats.get("failed", 0),
                model_failure_rate,
                avg_duration,
                model_stats.get("max_duration", 0.0),
                avg_attempts,
                model_stats.get("max_attempts", 0),
                model_stats.get("retry_tasks", 0),
                model_stats.get("queue_duration", 0.0),
            )

        auto_reminders = sum(
            connector.auto_compliance_reminders for connector in self.connectors.values()
        )
        manual_reminders = sum(
            connector.manual_compliance_reminders for connector in self.connectors.values()
        )
        array_warnings = sum(
            connector.array_warning_count for connector in self.connectors.values()
        )

        if auto_reminders or manual_reminders or array_warnings:
            self.logger.info(
                "Connector signals: auto_reminders=%s manual_reminders=%s multi_object_warnings=%s",
                auto_reminders,
                manual_reminders,
                array_warnings,
            )
