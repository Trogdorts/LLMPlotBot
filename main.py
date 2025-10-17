"""CLI entry point for running the batch processing pipeline."""

from __future__ import annotations

import logging

from src.config import CONFIG, CONFIG_SOURCES
from src.core.pipeline import BatchProcessingPipeline
from src.util.logger_setup import resolve_log_level, setup_logger


def main() -> None:
    """Initialise dependencies and execute the batch pipeline."""

    console_level = resolve_log_level(CONFIG.get("LOG_LEVEL_CONSOLE"), default=logging.INFO)
    file_level = resolve_log_level(CONFIG.get("LOG_LEVEL_FILE"), default=console_level)

    logger = setup_logger(
        CONFIG["LOG_DIR"],
        console_level,
        file_level=file_level,
    )
    pipeline = BatchProcessingPipeline(
        CONFIG,
        logger=logger,
        config_sources=CONFIG_SOURCES,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
