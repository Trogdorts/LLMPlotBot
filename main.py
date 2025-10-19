"""Command-line entry point for LLMPlotBot."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path


def _ensure_src_on_path() -> None:
    """Ensure the ``src`` directory is importable for local execution.

    When running ``python main.py`` directly the ``src`` directory that holds
    the project packages (``llmplotbot``, ``core``, etc.) is not automatically on
    ``sys.path``. Installing the project or executing via ``python -m`` would
    handle this for us, but keeping the helper means local runs work out of the
    box without requiring additional environment tweaks.
    """

    project_root = Path(__file__).resolve().parent
    src_path = project_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


_ensure_src_on_path()

from llmplotbot.runtime import LLMPlotBotRuntime


def main() -> int:
    runtime = LLMPlotBotRuntime.from_defaults()
    logger = runtime.logger
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
