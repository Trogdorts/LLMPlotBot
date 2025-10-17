"""Simplified TaskRunner stub using sequential execution."""
import logging


class TaskRunner:
    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def run(self):
        self.logger.info("TaskRunner is disabled in this simplified build.")
        self.logger.info("Use main.py for single-model sequential processing.")
