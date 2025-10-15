"""
Worker thread that consumes batches, calls the model connector, and persists results.
"""

import threading
import queue
from itertools import zip_longest
from typing import Dict
from .writer import ResultWriter
from .model_connector import ModelConnector


class Worker(threading.Thread):
    """Daemon thread that processes batches of tasks and writes results immediately."""

    def __init__(self, batch_q: "queue.Queue", writer: ResultWriter,
                 connectors: Dict[str, ModelConnector], shutdown_event: threading.Event, logger):
        super().__init__(daemon=True)
        self.batch_q = batch_q
        self.writer = writer
        self.connectors = connectors
        self.shutdown_event = shutdown_event
        self.logger = logger

    def run(self):
        """Process incoming batches until shutdown."""
        self.logger.debug("Worker thread started.")
        self.logger.debug(f"Active connectors: {list(self.connectors.keys())}")

        while not self.shutdown_event.is_set():
            try:
                batch = self.batch_q.get(timeout=0.5)
            except queue.Empty:
                continue

            if not batch:
                self.logger.debug("Received empty batch; skipping.")
                self.batch_q.task_done()
                continue

            model = batch[0].model
            self.logger.debug(f"Processing batch of {len(batch)} task(s) for model={model}.")

            connector = self.connectors.get(model)
            if not connector:
                self.logger.error(f"No connector found for model {model}")
                self.batch_q.task_done()
                continue

            responses = connector.send_batch(batch) or []
            received = len([r for r in responses if r]) if responses else 0
            if len(responses) != len(batch):
                self.logger.warning(
                    f"Model {model} returned {len(responses)} response(s) for {len(batch)} task(s); padding missing entries."
                )
            self.logger.debug(
                f"Worker got {received}/{len(batch)} valid responses for model={model}"
            )

            for task, response in zip_longest(batch, responses, fillvalue=None):
                if task is None:
                    continue
                if response:
                    if isinstance(response, dict):
                        response["title"] = task.title
                    self.writer.write(task.id, task.model, task.prompt_hash, response)
                else:
                    self.logger.warning(f"Empty/failed response id={task.id}")

            self.batch_q.task_done()

        self.logger.debug("Worker thread exiting.")
