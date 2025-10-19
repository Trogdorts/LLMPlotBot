"""Command-line entry point for LLMPlotBot."""

from __future__ import annotations

import asyncio
import sys

from llmplotbot.config import load_config
from llmplotbot.logging_utils import configure_logging
from llmplotbot.runtime import LLMPlotBotRuntime


def main() -> int:
    result = load_config(include_sources=True)
    config, sources = result.config, result.sources
    logging_config = dict(config.get("logging", {}))
    paths = config.get("paths", {})
    logging_config.setdefault("log_dir", paths.get("logs"))
    logger = configure_logging(logging_config)
    logger.info("Loaded configuration from: %s", ", ".join(sources) or "<defaults>")

    runtime = LLMPlotBotRuntime(config, logger)
    try:
        success = asyncio.run(runtime.run())
    except KeyboardInterrupt:  # pragma: no cover - manual interruption
        logger.warning("Interrupted by user.")
        return 1
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.exception("Runtime terminated due to unexpected error: %s", exc)
        return 1
    return 0 if success else 2


if __name__ == "__main__":
    sys.exit(main())
