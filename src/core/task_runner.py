"""Concurrent task execution engine for the batch processing pipeline."""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from typing import Deque, Dict, List, Mapping, Optional, Sequence

from src.core.metrics_collector import MetricsCollector
from src.core.metrics_summary import MetricsSummaryReporter
from src.core.model_connector import ModelConnector
from src.core.task import Task
from src.core.writer import ResultWriter


class TaskRunner:
    """Coordinate per-model workers that pull tasks and persist results."""

    def __init__(
        self,
        tasks_by_model: Mapping[str, Sequence[Task]],
        connectors: Mapping[str, ModelConnector],
        writer: ResultWriter,
        retry_limit: int,
        shutdown_event: threading.Event,
        logger: logging.Logger,
        *,
        model_aliases: Optional[Mapping[str, str]] = None,
        metrics_collector: Optional[MetricsCollector] = None,
        summary_reporter: Optional[MetricsSummaryReporter] = None,
        batch_size: int | None = None,
    ) -> None:
        self._tasks_by_model = {
            model: deque(sequence) for model, sequence in tasks_by_model.items()
        }
        self._connectors = connectors
        self._writer = writer
        self._retry_limit = max(1, int(retry_limit or 1))
        self._shutdown = shutdown_event
        self.logger = logger
        self._model_aliases = dict(model_aliases or {})
        self._metrics = metrics_collector
        self._summary = summary_reporter
        self._batch_size = max(1, int(batch_size or 1))

        self._lock = threading.Lock()
        self._workers: List[threading.Thread] = []

    # ------------------------------------------------------------------
    def run(self) -> None:
        if not self._tasks_by_model:
            self.logger.warning("No tasks to execute.")
            return

        total_tasks = sum(len(queue) for queue in self._tasks_by_model.values())
        self.logger.info(
            "Starting TaskRunner for %s task(s) across %s model(s).",
            total_tasks,
            len(self._tasks_by_model),
        )

        for model, queue in self._tasks_by_model.items():
            thread = threading.Thread(
                target=self._worker,
                name=f"TaskWorker[{model}]",
                args=(model, queue),
                daemon=True,
            )
            self._workers.append(thread)
            thread.start()

        for thread in self._workers:
            thread.join()

        self.logger.info("TaskRunner complete. All workers shut down.")

    # ------------------------------------------------------------------
    def _worker(self, model: str, queue: Deque[Task]) -> None:
        connector = self._connectors[model]
        alias = self._model_aliases.get(model, model)
        while not self._shutdown.is_set():
            batch = self._get_next_batch(queue)
            if not batch:
                return

            task_ids = ", ".join(task.id for task in batch)
            self.logger.debug("[%s] Processing batch: %s", alias, task_ids)

            try:
                self._process_batch(model, batch, connector)
            except Exception as exc:  # pragma: no cover - defensive logging
                self.logger.exception("[%s] Unhandled error: %s", alias, exc)
                self._shutdown.set()
                return

    # ------------------------------------------------------------------
    def _get_next_batch(self, queue: Deque[Task]) -> List[Task]:
        with self._lock:
            if not queue:
                return []
            batch: List[Task] = []
            while queue and len(batch) < self._batch_size:
                batch.append(queue.popleft())
            return batch

    # ------------------------------------------------------------------
    def _process_batch(
        self,
        model: str,
        batch: Sequence[Task],
        connector: ModelConnector,
    ) -> None:
        alias = self._model_aliases.get(model, model)
        attempts = 0
        prompt = self._compose_prompt(batch)
        title_id = batch[0].id if len(batch) == 1 else f"{batch[0].id}+{len(batch)}"

        while attempts < self._retry_limit and not self._shutdown.is_set():
            attempts += 1
            try:
                response, latency = connector.send_to_model(prompt, title_id)
                payloads = self._extract_payloads(response, batch)
                self._persist_results(model, batch, payloads)
                self._record_metrics(batch, model, latency, attempts)
                self.logger.debug(
                    "[%s] Batch %s succeeded in %s attempt(s).",
                    alias,
                    title_id,
                    attempts,
                )
                return
            except Exception as exc:
                if attempts >= self._retry_limit:
                    self.logger.error(
                        "[%s] Failed batch %s after %s attempts: %s",
                        alias,
                        title_id,
                        attempts,
                        exc,
                    )
                    self._record_metrics(batch, model, 0.0, attempts, success=False)
                    break
                self.logger.warning(
                    "[%s] Attempt %s/%s failed for batch %s: %s",
                    alias,
                    attempts,
                    self._retry_limit,
                    title_id,
                    exc,
                )
                time.sleep(min(2.0, 0.5 * attempts))
        # Failures are terminal for the batch; no re-queueing takes place here.

    # ------------------------------------------------------------------
    def _compose_prompt(self, batch: Sequence[Task]) -> str:
        if not batch:
            return ""
        dynamic_section = batch[0].prompt_dynamic.strip()
        formatting_section = batch[0].prompt_formatting.strip()
        headlines: List[str] = ["### HEADLINES"]
        for index, task in enumerate(batch, start=1):
            headlines.append(f"{index}. {task.title.strip()}")
        sections = [dynamic_section, "\n".join(headlines), formatting_section]
        return "\n\n".join(section for section in sections if section).strip()

    # ------------------------------------------------------------------
    def _extract_payloads(
        self,
        response: Mapping[str, object],
        batch: Sequence[Task],
    ) -> List[Dict[str, object]]:
        content = ModelConnector.extract_content(response) if isinstance(response, dict) else ""
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON payload: {exc}") from exc

        if isinstance(parsed, dict):
            parsed_list: List[Dict[str, object]] = [parsed]
        elif isinstance(parsed, list):
            parsed_list = [item for item in parsed if isinstance(item, dict)]
        else:
            raise ValueError("Model response must be a JSON object or array of objects.")

        if len(parsed_list) < len(batch):
            raise ValueError(
                f"Model returned {len(parsed_list)} item(s) for batch of {len(batch)}"
            )

        payloads: List[Dict[str, object]] = []
        for task, item in zip(batch, parsed_list):
            payload = dict(item)
            payload.setdefault("title", task.title)
            payloads.append(payload)
        return payloads

    # ------------------------------------------------------------------
    def _persist_results(
        self,
        model: str,
        batch: Sequence[Task],
        payloads: Sequence[Mapping[str, object]],
    ) -> None:
        for task, payload in zip(batch, payloads):
            self._writer.write(task.id, model, task.prompt_hash, dict(payload))

    # ------------------------------------------------------------------
    def _record_metrics(
        self,
        batch: Sequence[Task],
        model: str,
        latency: float,
        attempts: int,
        *,
        success: bool = True,
    ) -> None:
        if self._metrics is None:
            return
        total_records = 0
        for task in batch:
            total_records = self._metrics.record_task(
                task_id=task.id,
                model=model,
                latency=latency,
                success=success,
                attempts=attempts,
            )
        if self._summary is not None and total_records:
            self._summary.maybe_report(total_records)


__all__ = ["TaskRunner"]
