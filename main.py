"""
Main orchestrator for the LLM batch processor.
Handles backups, prompt loading, task generation, batching, workers, and graceful shutdown.
Uses utils_io.py for title indexing and caching.
"""

import os
import time
import queue
import threading
import logging
from src.config import CONFIG
from src.util.logger_setup import setup_logger
from src.util.backup_utils import create_backup
from src.util.prompt_utils import load_and_archive_prompt
from src.util.utils_io import load_cache, build_cache
from src.util.lmstudio_models import get_model_keys
from src.core.task import Task
from src.core.batch_manager import BatchManager
from src.core.worker import Worker
from src.core.writer import ResultWriter
from src.core.model_connector import ModelConnector
from src.core.shutdown import ShutdownManager


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
    logger.info("=== LLM Batch Processor Starting ===")

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
    prompt = load_and_archive_prompt(CONFIG["BASE_DIR"], logger)
    prompt_text, prompt_hash = prompt["prompt"], prompt["hash"]
    logger.info("Active prompt hash: %s", prompt_hash)

    # ---------- Setup global objects ----------
    shutdown_event = threading.Event()
    writer = ResultWriter(CONFIG["GENERATED_DIR"], logger=logger)
    ShutdownManager(shutdown_event, writer, logger).register()

    # ---------- Generate tasks ----------
    logger.info("Preparing task queues.")
    task_q, batch_q = queue.Queue(), queue.Queue()

    title_items = list(titles.items())
    if CONFIG["TEST_MODE"]:
        per_model_limit = CONFIG["BATCH_SIZE"] * CONFIG["TEST_BATCHES"]
        title_items = title_items[:per_model_limit]
        logger.info(
            "TEST_MODE enabled: limiting to %s task(s) per model across %s model(s).",
            per_model_limit,
            len(CONFIG["LLM_ENDPOINTS"]),
        )

    total_tasks = 0
    for model in CONFIG["LLM_ENDPOINTS"]:
        for tid, info in title_items:
            task_q.put(Task(tid, info["title"], model, prompt_hash, prompt_text))
            total_tasks += 1

    logger.info(
        "Queued %s task(s) spanning %s model(s).",
        total_tasks,
        len(CONFIG["LLM_ENDPOINTS"]),
    )

    # ---------- Initialize model connectors ----------
    connectors = {
        m: ModelConnector(m, u, CONFIG["REQUEST_TIMEOUT"], logger)
        for m, u in CONFIG["LLM_ENDPOINTS"].items()
    }

    logger.info("Initialized %s connector(s).", len(connectors))

    # ---------- Start batch manager & worker threads ----------
    bm = BatchManager(task_q, batch_q, shutdown_event, CONFIG, logger)
    bm.start()

    workers = [
        Worker(batch_q, writer, connectors, shutdown_event, logger)
        for _ in range(CONFIG["NUM_WORKERS"])
    ]
    for w in workers:
        w.start()

    # ---------- Main control loop ----------
    try:
        if CONFIG["TEST_MODE"]:
            logger.info("Running in TEST_MODE: processing one batch then exit...")
            while not shutdown_event.is_set():
                logger.debug(f"task_q={task_q.qsize()} batch_q={batch_q.qsize()}")
                time.sleep(2)
                if task_q.empty() and batch_q.empty():
                    logger.debug("Queues empty; ending test run.")
                    shutdown_event.set()
        else:
            while not shutdown_event.is_set():
                logger.debug(f"task_q={task_q.qsize()} batch_q={batch_q.qsize()}")
                time.sleep(2)
    except KeyboardInterrupt:
        logger.warning("KeyboardInterrupt detected. Initiating shutdown...")
        shutdown_event.set()

    # ---------- Graceful shutdown ----------
    bm.join()
    for w in workers:
        w.join()
    writer.flush()
    logger.info("=== Shutdown complete ===")


# ------------------------------------------------------------------
if __name__ == "__main__":
    main()
