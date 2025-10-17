"""Simplified TaskRunner stub using sequential execution."""

from __future__ import annotations

import logging
from typing import Dict, Iterable, Mapping


class TaskRunner:
    def __init__(
        self,
        tasks_by_model: Mapping[str, Iterable[object]],
        connectors: Mapping[str, object],
        writer,
        retry_limit: int,
        shutdown_event,
        logger: logging.Logger,
        *,
        model_aliases: Dict[str, str] | None = None,
        metrics_collector=None,
        summary_reporter=None,
        batch_size: int = 1,
    ) -> None:
        self.tasks_by_model = tasks_by_model
        self.connectors = connectors
        self.writer = writer
        self.retry_limit = retry_limit
        self.shutdown_event = shutdown_event
        self.logger = logger
        self.model_aliases = model_aliases or {}
        self.metrics_collector = metrics_collector
        self.summary_reporter = summary_reporter
        self.batch_size = max(int(batch_size or 1), 1)

    def run(self) -> None:
        total_tasks = sum(len(tasks) for tasks in self.tasks_by_model.values())
        model_list = ", ".join(sorted(self.tasks_by_model)) or "none"
        self.logger.warning(
            "TaskRunner is disabled in this simplified build; skipping %s task(s) for model(s): %s.",
            total_tasks,
            model_list,
        )
        self.logger.info("Use main.py in TEST_MODE for single-model sequential processing.")
