"""Helpers for constructing and executing the batch testing workflow."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Mapping

from src.config import Settings, load_settings
from src.core.io import load_prompt, load_titles
from src.core.model_connector import ModelConnector
from src.core.testing import BatchTester
from src.core.writer import ResultWriter


@dataclass(slots=True)
class Application:
    """Coordinate the creation and execution of a batch test run."""

    settings: Settings
    tester: BatchTester
    prompt: str
    titles: Mapping[str, Any]
    logger: logging.Logger

    def run(self) -> int:
        """Execute the configured batch workflow and return an exit code."""

        try:
            self.tester.run(self.prompt, self.titles)
        except RuntimeError as exc:  # pragma: no cover - exercised via integration
            self.logger.error("Batch aborted: %s", exc)
            return 1
        return 0


def build_application(
    *, settings: Settings | None = None, logger: logging.Logger | None = None
) -> Application:
    """Construct the :class:`Application` with shared dependencies."""

    resolved_settings = settings or load_settings()
    if logger is None:
        logger = logging.getLogger("LLMPlotBot")

    prompt = load_prompt(resolved_settings.prompt_path, logger=logger)
    titles = load_titles(resolved_settings.titles_path, logger=logger)

    writer = ResultWriter(
        resolved_settings.generated_dir,
        strategy=resolved_settings.write_strategy,
        flush_interval=resolved_settings.write_batch_size,
        flush_seconds=resolved_settings.write_batch_seconds,
        flush_retry_limit=resolved_settings.write_batch_retry_limit,
        lock_timeout=resolved_settings.file_lock_timeout,
        lock_poll_interval=resolved_settings.file_lock_poll_interval,
        lock_stale_seconds=resolved_settings.file_lock_stale_seconds,
        logger=logger,
    )

    connector = ModelConnector(
        resolved_settings.model,
        resolved_settings.lm_studio_url,
        resolved_settings.request_timeout,
        logger,
    )

    tester = BatchTester(
        settings=resolved_settings,
        connector=connector,
        writer=writer,
        logger=logger,
    )

    return Application(
        settings=resolved_settings,
        tester=tester,
        prompt=prompt,
        titles=titles,
        logger=logger,
    )


__all__ = ["Application", "build_application"]
