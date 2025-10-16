"""Main entry point for sequential headline processing via persistent LLM sessions."""

import logging
import os
import threading

from src.config import CONFIG, CONFIG_SOURCES
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

    def _normalise_names(raw):
        if isinstance(raw, str):
            items = [part.strip() for part in raw.split(",")]
        elif isinstance(raw, (list, tuple, set)):
            items = [str(part).strip() for part in raw]
        else:
            return []
        return [item for item in items if item]

    blocklist = {name.lower() for name in _normalise_names(config.get("LLM_BLOCKLIST", []))}

    preconfigured = config.get("LLM_ENDPOINTS") or {}
    if preconfigured:
        filtered = {
            model: url
            for model, url in preconfigured.items()
            if model and model.lower() not in blocklist
        }
        if not filtered:
            logger.error("All configured LLM endpoints were filtered by LLM_BLOCKLIST.")
            return {}
        if len(filtered) != len(preconfigured):
            removed = sorted(set(preconfigured) - set(filtered))
            logger.warning(
                "Excluding %s blocklisted model(s): %s",
                len(removed),
                ", ".join(removed),
            )
        logger.info("Using pre-configured LLM endpoints for %s model(s).", len(filtered))
        logger.debug("Pre-configured endpoints: %s", filtered)
        return filtered

    base_url = config.get("LLM_BASE_URL", "http://localhost:1234")

    configured_models = _normalise_names(config.get("LLM_MODELS", []))

    if configured_models:
        logger.info("Using explicitly configured LLM models: %s", ", ".join(configured_models))
        candidate_models = configured_models
    else:
        detected = get_model_keys(logger)
        if not detected:
            logger.error("No running LM Studio models detected and no models configured.")
            return {}
        logger.info("Detected running LM Studio models: %s", ", ".join(detected))
        candidate_models = detected

    filtered_models = [model for model in candidate_models if model.lower() not in blocklist]
    if not filtered_models:
        logger.error("No models available after applying LLM_BLOCKLIST filters.")
        return {}

    removed = sorted(set(candidate_models) - set(filtered_models))
    if removed:
        logger.warning(
            "Excluding %s blocklisted model(s): %s",
            len(removed),
            ", ".join(removed),
        )

    endpoints = {model: f"{base_url}/v1/chat/completions" for model in filtered_models}
    logger.debug("Resolved LLM endpoints: %s", endpoints)
    return endpoints


# ------------------------------------------------------------------
def main():
    # ---------- Initialize logging ----------
    logger = setup_logger(CONFIG["LOG_DIR"], logging.INFO)
    logger.info("=== LLM Sequential Processor Starting ===")
    if CONFIG_SOURCES:
        logger.info("Loaded config overrides from: %s", ", ".join(CONFIG_SOURCES))
    else:
        logger.debug("No config override files found; using defaults.")

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

    compliance_interval = int(CONFIG.get("COMPLIANCE_REMINDER_INTERVAL", 0) or 0)
    if compliance_interval > 0:
        logger.info(
            "Automatic JSON compliance reminders every %s headline(s).",
            compliance_interval,
        )
    summary_interval = float(CONFIG.get("SUMMARY_LOG_INTERVAL_SECONDS", 0) or 0)
    if summary_interval > 0:
        logger.info(
            "Periodic summary logging every %.1f second(s).",
            summary_interval,
        )
    connectors = {
        model: ModelConnector(
            model,
            url,
            CONFIG["REQUEST_TIMEOUT"],
            compliance_interval,
            logger,
        )
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
        summary_interval=summary_interval,
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
