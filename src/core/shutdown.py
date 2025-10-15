
"""
ShutdownManager registers signal handlers and ensures writer flush on exit.
"""

import signal


class ShutdownManager:
    """Handles SIGINT/SIGTERM to trigger a graceful shutdown."""

    def __init__(self, shutdown_event, writer, logger):
        self.shutdown_event = shutdown_event
        self.writer = writer
        self.logger = logger

    def register(self):
        def handler(sig, frame):
            self.logger.warning("Shutdown signal received. Flushing buffers...")
            self.shutdown_event.set()
            self.writer.flush()
        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)
