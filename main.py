"""Command-line entry point for LLMPlotBot."""

from __future__ import annotations

import sys

from llmplotbot.config import load_config
from llmplotbot.logging_utils import configure_logging
from llmplotbot.pipeline import ProcessingPipeline


def main() -> int:
    result = load_config(include_sources=True)
    config, sources = result.config, result.sources
    logger = configure_logging(config.get("LOG_LEVEL", "INFO"), log_dir=config.get("LOG_DIR"))
    logger.info("Loaded configuration from: %s", ", ".join(sources) or "<defaults>")

    pipeline = ProcessingPipeline(config, logger=logger, config_sources=sources)
    try:
        success = pipeline.run()
    except KeyboardInterrupt:  # pragma: no cover - manual interruption
        logger.warning("Interrupted by user.")
        return 1
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Pipeline terminated due to unexpected error: %s", exc)
        return 1
    return 0 if success else 2


if __name__ == "__main__":
    sys.exit(main())
