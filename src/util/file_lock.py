"""Cross-platform file lock utility using lock files."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass


class FileLockTimeout(TimeoutError):
    """Raised when a file lock cannot be acquired within the timeout."""


@dataclass
class FileLock:
    """Simple lock-file based context manager with stale lock cleanup."""

    path: str
    timeout: float = 10.0
    poll_interval: float = 0.1
    stale_seconds: float = 300.0

    def __post_init__(self) -> None:
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")
        if self.poll_interval <= 0:
            raise ValueError("poll_interval must be positive")
        if self.stale_seconds <= 0:
            raise ValueError("stale_seconds must be positive")
        self._fd = None

    # ------------------------------------------------------------------
    def acquire(self) -> None:
        deadline = time.time() + self.timeout

        while True:
            try:
                self._fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(self._fd, str(os.getpid()).encode("utf-8"))
                return
            except FileExistsError:
                if self._is_stale():
                    self._clear_stale()
                    continue

                if time.time() >= deadline:
                    raise FileLockTimeout(f"Timed out waiting for lock: {self.path}")
                time.sleep(self.poll_interval)

    # ------------------------------------------------------------------
    def release(self) -> None:
        if self._fd is not None:
            try:
                os.close(self._fd)
            finally:
                self._fd = None
        try:
            os.unlink(self.path)
        except FileNotFoundError:
            pass

    # ------------------------------------------------------------------
    def _is_stale(self) -> bool:
        try:
            stat = os.stat(self.path)
        except FileNotFoundError:
            return False
        age = time.time() - stat.st_mtime
        return age >= self.stale_seconds

    def _clear_stale(self) -> None:
        try:
            os.unlink(self.path)
        except FileNotFoundError:
            pass

    # ------------------------------------------------------------------
    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.release()
        return False
