"""Main entry point for sequential headline processing via persistent LLM sessions."""

import logging
import os
import threading
import time
from collections import OrderedDict

from src.config import CONFIG, CONFIG_SOURCES
from src.core.metrics_collector import MetricsCollector
from src.core.metrics_summary import MetricsSummaryReporter
from src.core.model_connector import ModelConnector
from src.core.shutdown import ShutdownManager
from src.core.task import Task
from src.core.task_runner import TaskRunner
from src.core.writer import ResultWriter
from src.util.backup_utils import create_backup
from src.util.lmstudio_models import (
    get_model_keys,
    group_model_keys,
    normalize_model_key,
)
from src.util.logger_setup import setup_logger
from src.util.prompt_utils import load_and_archive_prompt
from src.util.result_utils import ExistingResultChecker
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

        grouped = group_model_keys(detected)
        duplicates = {base: keys for base, keys in grouped.items() if len(keys) > 1}
        if duplicates:
            duplicate_summary = ", ".join(
                f"{base} ×{len(keys)}" for base, keys in duplicates.items()
            )
            logger.info("Detected multi-instance models: %s", duplicate_summary)

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
        flush_retry_limit=CONFIG.get("WRITE_BATCH_RETRY_LIMIT", 3),
        lock_timeout=CONFIG.get("FILE_LOCK_TIMEOUT", 10.0),
        lock_poll_interval=CONFIG.get("FILE_LOCK_POLL_INTERVAL", 0.1),
        lock_stale_seconds=CONFIG.get("FILE_LOCK_STALE_SECONDS", 300.0),
        logger=logger,
    )
    metrics_collector = MetricsCollector()
    summary_task_interval = int(CONFIG.get("METRICS_SUMMARY_TASK_INTERVAL", 0) or 0)
    summary_time_seconds = float(
        CONFIG.get(
            "METRICS_SUMMARY_TIME_SECONDS",
            CONFIG.get("SUMMARY_INTERVAL", 0.0),
        )
        or 0.0
    )
    summary_reporter = MetricsSummaryReporter(
        metrics_collector,
        logger,
        CONFIG["LOG_DIR"],
        summary_every_tasks=summary_task_interval,
        summary_every_seconds=summary_time_seconds,
    )
    summary_reporter.start()
    ShutdownManager(shutdown_event, writer, logger, summary_reporter=summary_reporter).register()

    # ---------- Prepare task lists ----------
    title_items = list(titles.items())
    test_limit_per_model = None
    if CONFIG["TEST_MODE"]:
        limit = CONFIG.get("TEST_LIMIT_PER_MODEL")
        if limit:
            try:
                test_limit_per_model = int(limit)
            except (TypeError, ValueError):
                logger.warning(
                    "Invalid TEST_LIMIT_PER_MODEL value %r; ignoring per-model limit.",
                    limit,
                )
                test_limit_per_model = None
            else:
                logger.info(
                    "TEST_MODE enabled: limiting to %s headline(s) per model.",
                    test_limit_per_model,
                )

    compliance_interval = int(CONFIG.get("COMPLIANCE_REMINDER_INTERVAL", 0) or 0)
    if compliance_interval > 0:
        logger.info(
            "Automatic JSON compliance reminders every %s headline(s).",
            compliance_interval,
        )
    connectors = {
        model: ModelConnector(
            model,
            url,
            CONFIG["REQUEST_TIMEOUT"],
            compliance_interval,
            logger,
            CONFIG.get("EXPECTED_LANGUAGE"),
        )
        for model, url in CONFIG["LLM_ENDPOINTS"].items()
    }
    logger.info("Initialized %s connector(s).", len(connectors))

    model_groups = OrderedDict()
    model_aliases = {}
    for model in connectors:
        base = normalize_model_key(model)
        model_groups.setdefault(base, []).append(model)
        model_aliases[model] = base

    def _distribute_titles(items, slots):
        if slots <= 1:
            return [list(items)]
        buckets = [[] for _ in range(slots)]
        for idx, item in enumerate(items):
            buckets[idx % slots].append(item)
        return buckets

    result_checker = ExistingResultChecker(CONFIG["GENERATED_DIR"], logger)
    skipped_by_model = {}
    tasks_by_model = {}
    for base, models in model_groups.items():
        subsets = _distribute_titles(title_items, len(models))
        if len(models) > 1:
            logger.info(
                "Distributing %s headline(s) across %s instances of %s.",
                len(title_items),
                len(models),
                base,
            )
        for model, subset in zip(models, subsets):
            filtered_subset = []
            for tid, info in subset:
                if result_checker.has_entry(tid, model, prompt_hash):
                    skipped_by_model[model] = skipped_by_model.get(model, 0) + 1
                    continue
                filtered_subset.append((tid, info))

            model_tasks = [
                Task(
                    tid,
                    info["title"],
                    model,
                    prompt_hash,
                    prompt_bundle.dynamic_section,
                    prompt_bundle.formatting_section,
                )
                for tid, info in filtered_subset
            ]
            if test_limit_per_model is not None and len(model_tasks) > test_limit_per_model:
                model_tasks = model_tasks[:test_limit_per_model]

            tasks_by_model[model] = model_tasks

    total_skipped = sum(skipped_by_model.values())
    if total_skipped:
        logger.info(
            "Skipping %s existing task(s) across %s model(s).",
            total_skipped,
            len(skipped_by_model),
        )
        for model, count in sorted(skipped_by_model.items()):
            logger.debug(
                "Model %s already has %s completed task(s) for prompt %s.",
                model,
                count,
                prompt_hash,
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
        model_aliases=model_aliases,
        metrics_collector=metrics_collector,
        summary_reporter=summary_reporter,
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
        summary_reporter.finalize(
            reason="normal_exit", session_end_time=time.time()
        )
        logger.info("=== Shutdown complete ===")


# ------------------------------------------------------------------
if __name__ == "__main__":
    main()
