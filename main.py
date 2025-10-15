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
import requests
from src.config import CONFIG
from src.util.logger_setup import setup_logger
from src.util.backup_utils import create_backup
from src.util.prompt_utils import load_and_archive_prompt
from src.util.utils_io import load_cache, build_cache
from src.core.task import Task
from src.core.batch_manager import BatchManager
from src.core.worker import Worker
from src.core.writer import ResultWriter
from src.core.model_connector import ModelConnector
from src.core.shutdown import ShutdownManager


# ------------------------------------------------------------------
def auto_discover_models(config, logger):
    """
    Query the LM Studio or OpenAI-compatible API for available models.
    Dynamically updates CONFIG["LLM_ENDPOINTS"] with all detected model names.
    """
    base_url = config.get("LLM_BASE_URL", "http://localhost:1234")
    url = f"{base_url}/v1/models"
    endpoints = {}

    try:
        r = requests.get(url, timeout=5)
        r.raise_for_status()
        payload = r.json()
        models = [m.get("id") or m.get("name") for m in payload.get("data", []) if m.get("id") or m.get("name")]

        if not models:
            msg = "No models returned by /v1/models; using manual configuration."
            logger.warning(msg)
            return config

        for m in models:
            endpoints[m] = f"{base_url}/v1/chat/completions"

        config["LLM_ENDPOINTS"] = endpoints

        msg = f"Detected LLM models ({len(models)}): {', '.join(models)}"
        logger.info(msg)


    except Exception as e:
        msg = f"Automatic model discovery failed: {e}"
        logger.warning(msg)

    return config


# ------------------------------------------------------------------
def main():
    # ---------- Initialize logging ----------
    logger = setup_logger(CONFIG["LOG_DIR"], logging.INFO)
    logger.info("=== LLM Batch Processor Starting ===")

    # ---------- Auto-detect and register LLM models ----------
    CONFIG.update(auto_discover_models(CONFIG, logger))

    # ---------- Ensure directories ----------
    os.makedirs(CONFIG["GENERATED_DIR"], exist_ok=True)
    create_backup(CONFIG["BACKUP_DIR"], CONFIG["IGNORE_FOLDERS"], logger)

    # ---------- Load or build title index ----------
    cache_file = os.path.join(CONFIG["BASE_DIR"], "titles_index.json")
    CONFIG["CACHE_PATH"] = cache_file

    titles = load_cache(CONFIG, logger)
    if titles is None:
        titles = build_cache(CONFIG, logger)

    logger.info(f"Loaded {len(titles)} titles for processing.")

    # ---------- Load prompt ----------
    prompt = load_and_archive_prompt(CONFIG["BASE_DIR"], logger)
    prompt_text, prompt_hash = prompt["prompt"], prompt["hash"]
    logger.info(f"Active prompt hash: {prompt_hash}")

    # ---------- Setup global objects ----------
    shutdown_event = threading.Event()
    writer = ResultWriter(CONFIG["GENERATED_DIR"], logger=logger)
    ShutdownManager(shutdown_event, writer, logger).register()

    # ---------- Generate tasks ----------
    logger.debug("Generating tasks for queue...")
    task_q, batch_q = queue.Queue(), queue.Queue()

    count = 0
    for model in CONFIG["LLM_ENDPOINTS"]:
        for tid, info in titles.items():
            task_q.put(Task(tid, info["title"], model, prompt_hash, prompt_text))
            count += 1
            if CONFIG["TEST_MODE"] and count >= CONFIG["BATCH_SIZE"]:
                break
        if CONFIG["TEST_MODE"]:
            break
    logger.info(f"Total tasks queued: {task_q.qsize()}")

    # ---------- Initialize model connectors ----------
    connectors = {
        m: ModelConnector(m, u, CONFIG["REQUEST_TIMEOUT"], logger)
        for m, u in CONFIG["LLM_ENDPOINTS"].items()
    }

    logger.info(f"Initialized connectors for models: {', '.join(connectors.keys())}")

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
