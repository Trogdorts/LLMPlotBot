"""Command-line entry point for exercising the batch tester."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence

from src.app.runtime import build_application
from src.config import load_settings


def configure_logging(*, level: int | str = logging.INFO) -> logging.Logger:
    """Initialise and return the shared application logger."""

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logger = logging.getLogger("LLMPlotBot")
    logger.propagate = False
    return logger


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments for the runner."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional path to an alternate configuration file.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=[
            "CRITICAL",
            "ERROR",
            "WARNING",
            "INFO",
            "DEBUG",
        ],
        help="Adjust the verbosity of runtime logging.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> int:
    """Entrypoint used by the ``if __name__ == '__main__'`` guard."""

    args = parse_args(argv)
    logger = configure_logging(level=args.log_level)
    settings = load_settings(args.config)

    application = build_application(settings=settings, logger=logger)
    return application.run()


if __name__ == "__main__":
    raise SystemExit(main())
