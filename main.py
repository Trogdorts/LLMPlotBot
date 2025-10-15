"""Main entry point for sequential headline processing via persistent LLM sessions."""

import logging
import os
import threading

from src.config import CONFIG
from src.core.model_connector import ModelConnector
from src.core.shutdown import ShutdownManager
from src.core.task import Task
from src.core.task_runner import TaskRunner
from src.core.writer import ResultWriter
from src.util.backup_utils import create_backup
from src.util.lmstudio_models import get_model_keys
from src.util.logger_setup import setup_logger
from src.util.prompt_utils import load_and_archive_prompt
from src.util.utils_io import build_cache, load_cache


# ------------------------------------------------------------------
def resolve_model_endpoints(config, logger):
    """Return the endpoint map based on explicit config or running LM Studio models."""

    # Respect pre-configured endpoint mappings when provided.
    preconfigured = config.get("LLM_ENDPOINTS") or {}
    if preconfigured:
        logger.info(
            "Using pre-configured LLM endpoints for %s model(s).",
            len(preconfigured),
        )
        logger.debug("Pre-configured endpoints: %s", preconfigured)
        return dict(preconfigured)

    base_url = config.get("LLM_BASE_URL", "http://localhost:1234")

    configured_models = config.get("LLM_MODELS", [])
    if isinstance(configured_models, str):
        configured_models = [m.strip() for m in configured_models.split(",")]
    configured_models = [m.strip() for m in configured_models if m and m.strip()]

    if configured_models:
        logger.info("Using explicitly configured LLM models: %s", ", ".join(configured_models))
        models = configured_models
    else:
        models = get_model_keys(logger)
        if models:
            logger.info("Detected running LM Studio models: %s", ", ".join(models))
        else:
            logger.error("No running LM Studio models detected and no models configured.")
            return {}

    endpoints = {model: f"{base_url}/v1/chat/completions" for model in models}
    logger.debug("Resolved LLM endpoints: %s", endpoints)
    return endpoints


# ------------------------------------------------------------------
def main():
    # ---------- Initialize logging ----------
    logger = setup_logger(CONFIG["LOG_DIR"], logging.INFO)
    logger.info("=== LLM Sequential Processor Starting ===")

    # ---------- Auto-detect and register LLM models ----------
    CONFIG["LLM_ENDPOINTS"] = resolve_model_endpoints(CONFIG, logger)

    if not CONFIG["LLM_ENDPOINTS"]:
        logger.error("No LLM endpoints available. Exiting.")
        return

    # ---------- Ensure directories ----------
    os.makedirs(CONFIG["GENERATED_DIR"], exist_ok=True)
    create_backup(CONFIG["BACKUP_DIR"], CONFIG["IGNORE_FOLDERS"], logger)

    # ---------- Load or build title index ----------
    cache_file = os.path.join(CONFIG["BASE_DIR"], "titles_index.json")
    CONFIG["CACHE_PATH"] = cache_file

    titles = load_cache(CONFIG, logger)
    if titles is None:
        titles = build_cache(CONFIG, logger)

    logger.info("Loaded %s titles for processing.", len(titles))

    # ---------- Load prompt ----------
    prompt_bundle = load_and_archive_prompt(CONFIG["BASE_DIR"], logger)
    prompt_hash = prompt_bundle.prompt_hash
    logger.info("Active prompt hash: %s", prompt_hash)

    # ---------- Setup global objects ----------
    shutdown_event = threading.Event()
    writer = ResultWriter(
        CONFIG["GENERATED_DIR"],
        strategy=CONFIG.get("WRITE_STRATEGY", "immediate"),
        flush_interval=CONFIG.get("WRITE_BATCH_SIZE", 1),
        flush_seconds=CONFIG.get("WRITE_BATCH_SECONDS", 5.0),
        lock_timeout=CONFIG.get("FILE_LOCK_TIMEOUT", 10.0),
        lock_poll_interval=CONFIG.get("FILE_LOCK_POLL_INTERVAL", 0.1),
        lock_stale_seconds=CONFIG.get("FILE_LOCK_STALE_SECONDS", 300.0),
        logger=logger,
    )
    ShutdownManager(shutdown_event, writer, logger).register()

    # ---------- Prepare task lists ----------
    title_items = list(titles.items())
    if CONFIG["TEST_MODE"]:
        limit = CONFIG.get("TEST_LIMIT_PER_MODEL")
        if limit:
            title_items = title_items[:limit]
            logger.info(
                "TEST_MODE enabled: limiting to %s headline(s) per model.",
                limit,
            )

    connectors = {
        model: ModelConnector(model, url, CONFIG["REQUEST_TIMEOUT"], logger)
        for model, url in CONFIG["LLM_ENDPOINTS"].items()
    }
    logger.info("Initialized %s connector(s).", len(connectors))

    tasks_by_model = {model: [] for model in connectors}
    for model in connectors:
        for tid, info in title_items:
            tasks_by_model[model].append(
                Task(
                    tid,
                    info["title"],
                    model,
                    prompt_hash,
                    prompt_bundle.dynamic_section,
                    prompt_bundle.formatting_section,
                )
            )

    total_tasks = sum(len(items) for items in tasks_by_model.values())
    logger.info(
        "Prepared %s task(s) spanning %s model(s).",
        total_tasks,
        len(tasks_by_model),
    )

    runner = TaskRunner(
        tasks_by_model,
        connectors,
        writer,
        CONFIG["RETRY_LIMIT"],
        shutdown_event,
        logger,
    )

    try:
        runner.run()
    except KeyboardInterrupt:
        logger.warning("KeyboardInterrupt detected. Initiating shutdown...")
        shutdown_event.set()
    finally:
        for connector in connectors.values():
            connector.close_session()
        writer.flush()
        logger.info("=== Shutdown complete ===")


# ------------------------------------------------------------------
if __name__ == "__main__":
    main()
