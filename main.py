import logging

from src.config import load_settings
from src.core.io import load_prompt, load_titles
from src.core.model_connector import ModelConnector
from src.core.testing import BatchTester
from src.core.writer import ResultWriter


def configure_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    return logging.getLogger("LLMPlotBot")


def main() -> None:
    logger = configure_logging()
    settings = load_settings()

    prompt = load_prompt(settings.prompt_path, logger=logger)
    titles = load_titles(settings.titles_path, logger=logger)

    writer = ResultWriter(
        settings.generated_dir,
        strategy=settings.write_strategy,
        flush_interval=settings.write_batch_size,
        flush_seconds=settings.write_batch_seconds,
        flush_retry_limit=settings.write_batch_retry_limit,
        lock_timeout=settings.file_lock_timeout,
        lock_poll_interval=settings.file_lock_poll_interval,
        lock_stale_seconds=settings.file_lock_stale_seconds,
        logger=logger,
    )

    connector = ModelConnector(
        settings.model,
        settings.lm_studio_url,
        settings.request_timeout,
        logger,
    )

    tester = BatchTester(
        settings=settings,
        connector=connector,
        writer=writer,
        logger=logger,
    )

    try:
        tester.run(prompt, titles)
    except RuntimeError as exc:
        logger.error("Batch aborted: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
