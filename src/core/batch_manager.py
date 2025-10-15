"""
BatchManager groups tasks by (model, prompt_hash) and flushes by size or timeout.
"""

import threading
import time
import queue
from collections import defaultdict


class BatchManager(threading.Thread):
    """Groups incoming tasks into batches for efficient model calls."""

    def __init__(self, in_queue: "queue.Queue", out_queue: "queue.Queue",
                 shutdown_event: threading.Event, config: dict, logger):
        super().__init__(daemon=True)
        self.in_q = in_queue
        self.out_q = out_queue
        self.shutdown_event = shutdown_event
        self.batch_size = config["BATCH_SIZE"]
        self.timeout = config["BATCH_TIMEOUT"]
        self.logger = logger
        self.buffer = defaultdict(list)
        self.last_flush = defaultdict(lambda: time.time())

    def run(self):
        self.logger.info("BatchManager thread started.")
        while not self.shutdown_event.is_set():
            try:
                task = self.in_q.get(timeout=0.5)
            except queue.Empty:
                self._flush_expired()
                continue

            self.logger.debug(f"Received task id={task.id} model={task.model}")
            key = (task.model, task.prompt_hash)
            self.buffer[key].append(task)
            self.logger.info(
                f"Queued task {task.id} for model={task.model}; batch size now {len(self.buffer[key])}/{self.batch_size}"
            )

            if len(self.buffer[key]) >= self.batch_size:
                self.logger.info(f"Batch size threshold reached for {key}; flushing now.")
                self._flush_batch(key)

            self._flush_expired()

        # Flush remaining batches on shutdown
        for key in list(self.buffer.keys()):
            self._flush_batch(key)
        self.logger.info("BatchManager thread exiting.")

    def _flush_expired(self):
        now = time.time()
        for key, last in list(self.last_flush.items()):
            if now - last > self.timeout and self.buffer[key]:
                self.logger.info(f"Timeout reached for {key}; flushing {len(self.buffer[key])} task(s).")
                self._flush_batch(key)

    def _flush_batch(self, key):
        batch = self.buffer.pop(key)
        self.last_flush[key] = time.time()
        self.logger.info(f"Dispatching batch for model={key[0]} hash={key[1]} size={len(batch)}")
        self.out_q.put(batch)
        self.logger.debug(f"Flushing batch for {key[0]} (model) hash={key[1]} size={len(batch)}")

