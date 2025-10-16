
"""
ShutdownManager registers signal handlers and ensures writer flush on exit.
"""

import signal


class ShutdownManager:
    """Handles SIGINT/SIGTERM to trigger a graceful shutdown."""

    def __init__(self, shutdown_event, writer, logger, summary_reporter=None):
        self.shutdown_event = shutdown_event
        self.writer = writer
        self.logger = logger
        self.summary_reporter = summary_reporter

    def register(self):
        def handler(sig, frame):
            self.logger.warning("Shutdown signal received. Flushing buffers...")
            self.shutdown_event.set()
            self.writer.flush()
            if self.summary_reporter is not None:
                self.summary_reporter.handle_shutdown_signal()
        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)
