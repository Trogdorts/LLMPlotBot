"""CLI entry point for running the batch processing pipeline."""

from __future__ import annotations

import logging

from src.config import CONFIG, CONFIG_SOURCES
from src.core.pipeline import BatchProcessingPipeline
from src.util.logger_setup import setup_logger


def main() -> None:
    """Initialise dependencies and execute the batch pipeline."""

    logger = setup_logger(CONFIG["LOG_DIR"], logging.DEBUG)
    pipeline = BatchProcessingPipeline(
        CONFIG,
        logger=logger,
        config_sources=CONFIG_SOURCES,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
