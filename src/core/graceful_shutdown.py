"""Signal-aware shutdown helper."""

from __future__ import annotations

import asyncio
import signal
from typing import Iterable


class GracefulShutdown:
    def __init__(self) -> None:
        self._event = asyncio.Event()
        self._installed = False

    def install(self, signals: Iterable[int] | None = None) -> None:
        loop = asyncio.get_event_loop()
        if self._installed and loop.is_closed():  # pragma: no cover - defensive
            return
        self._installed = True
        targets = list(signals) if signals is not None else [signal.SIGINT, signal.SIGTERM]
        for sig in targets:
            try:
                loop.add_signal_handler(sig, self.trigger)
            except NotImplementedError:  # pragma: no cover - windows fallback
                signal.signal(sig, lambda *_: self.trigger())

    def trigger(self) -> None:
        if not self._event.is_set():
            self._event.set()

    @property
    def event(self) -> asyncio.Event:
        return self._event

    def is_triggered(self) -> bool:
        return self._event.is_set()


__all__ = ["GracefulShutdown"]
