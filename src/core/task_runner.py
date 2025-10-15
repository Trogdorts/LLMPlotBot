"""Sequential task runner that orchestrates per-model execution."""

from typing import Dict, List

from .model_connector import ModelConnector
from .task import Task
from .writer import ResultWriter


class TaskRunner:
    """Process tasks sequentially with retries using persistent connector sessions."""

    def __init__(
        self,
        tasks_by_model: Dict[str, List[Task]],
        connectors: Dict[str, ModelConnector],
        writer: ResultWriter,
        retry_limit: int,
        shutdown_event,
        logger,
    ) -> None:
        self.tasks_by_model = tasks_by_model
        self.connectors = connectors
        self.writer = writer
        self.retry_limit = retry_limit
        self.shutdown_event = shutdown_event
        self.logger = logger

    # ------------------------------------------------------------------
    def run(self) -> None:
        """Run through all tasks for each model sequentially."""
        for model, tasks in self.tasks_by_model.items():
            if self.shutdown_event.is_set():
                break

            if not tasks:
                self.logger.debug("No tasks queued for model %s; skipping.", model)
                continue

            connector = self.connectors.get(model)
            if not connector:
                self.logger.error("No connector available for model %s", model)
                continue

            connector.start_session(tasks[0].prompt_text)
            try:
                self._process_model_tasks(connector, tasks)
            finally:
                connector.close_session()

        if self.shutdown_event.is_set():
            self.logger.info("Task runner halted due to shutdown signal.")
        else:
            self.logger.info("Task runner completed all queued work.")

    # ------------------------------------------------------------------
    def _process_model_tasks(self, connector: ModelConnector, tasks: List[Task]) -> None:
        for task in tasks:
            if self.shutdown_event.is_set():
                self.logger.debug("Shutdown requested; abandoning remaining tasks for %s.", connector.model)
                return

            self.logger.info("Processing task id=%s model=%s", task.id, task.model)
            if not self._run_with_retries(connector, task):
                self.logger.error(
                    "Task id=%s failed after %s attempt(s); moving on.",
                    task.id,
                    self.retry_limit,
                )

    def _run_with_retries(self, connector: ModelConnector, task: Task) -> bool:
        attempt = 0
        while attempt < self.retry_limit and not self.shutdown_event.is_set():
            attempt += 1
            self.logger.debug("Attempt %s for task id=%s", attempt, task.id)
            result = connector.send_headline(task.title)
            if result:
                result["title"] = task.title
                self.writer.write(task.id, task.model, task.prompt_hash, result)
                self.logger.info("Task id=%s processed successfully on attempt %s.", task.id, attempt)
                return True

            if attempt < self.retry_limit:
                connector.reinforce_compliance()
                self.logger.warning(
                    "Model %s returned invalid payload for task id=%s; retrying (%s/%s).",
                    connector.model,
                    task.id,
                    attempt,
                    self.retry_limit,
                )

        return False
