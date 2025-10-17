"""High-level orchestration for the LLM batch processing workflow."""

from __future__ import annotations
import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Dict, Iterable, MutableMapping

from src.core.batch_planner import BatchPlan, BatchPlanner
from src.core.metrics_collector import MetricsCollector
from src.core.metrics_summary import MetricsSummaryReporter
from src.core.model_connector import ModelConnector
from src.core.shutdown import ShutdownManager
from src.core.task_runner import TaskRunner
from src.core.writer import ResultWriter
from src.util.backup_utils import create_backup
from src.util.lmstudio_models import get_model_keys
from src.util.prompt_utils import PromptBundle, load_and_archive_prompt
from src.util.result_utils import ExistingResultChecker
from src.util.utils_io import build_cache, load_cache


@dataclass(slots=True)
class PipelineDependencies:
    """Bundle of initialized helper objects."""

    writer: ResultWriter
    metrics_collector: MetricsCollector
    summary_reporter: MetricsSummaryReporter
    shutdown_event: threading.Event


class BatchProcessingPipeline:
    """Execute the full fetch → plan → run → persist workflow."""

    def __init__(
        self,
        config: MutableMapping[str, object],
        *,
        logger: logging.Logger,
        config_sources: Iterable[str] | None = None,
    ) -> None:
        self.config = config
        self.logger = logger
        self.config_sources = tuple(config_sources or ())
        self._current_prompt_hash: str | None = None

    # ------------------------------------------------------------------
    def run(self) -> bool:
        """Main entry point for full pipeline execution."""
        self._log_startup()

        endpoints = self._resolve_model_endpoints()
        if not endpoints:
            self.logger.error("No LLM endpoints available. Exiting.")
            return False
        self.config["LLM_ENDPOINTS"] = endpoints

        self._prepare_directories()
        titles = self._load_titles()
        if not titles:
            self.logger.error("No titles available for processing.")
            return False
        original_title_count = len(titles)

        prompt_bundle = self._load_prompt()
        prompt_hash = prompt_bundle.prompt_hash
        self._current_prompt_hash = prompt_hash
        self.logger.info("Active prompt hash: %s", prompt_hash)

        result_checker = ExistingResultChecker(self.config["GENERATED_DIR"], self.logger)
        titles = self._filter_titles_for_run(titles, result_checker, endpoints.keys())
        if not titles:
            self.logger.warning(
                "All %s loaded titles already have generated output for the active models.",
                original_title_count,
            )
            return False

        filtered_count = len(titles)
        if filtered_count != original_title_count:
            self.logger.info(
                "Loaded %s titles for processing after skipping %s existing result(s).",
                filtered_count,
                original_title_count - filtered_count,
            )
        else:
            self.logger.info("Loaded %s titles for processing.", filtered_count)

        planner = BatchPlanner(
            titles,
            prompt_hash,
            prompt_dynamic=prompt_bundle.dynamic_section,
            prompt_formatting=prompt_bundle.formatting_section,
            result_checker=result_checker,
            logger=self.logger,
            test_limit_per_model=self._test_limit_per_model(),
        )
        plan = planner.build(endpoints)
        self._log_plan(plan)
        if not plan.total_tasks:
            self.logger.warning("No work to process after skipping existing results.")
            return False

        tasks_by_model = plan.tasks_by_model
        active_endpoints = {
            model: endpoints[model]
            for model in tasks_by_model
            if model in endpoints
        }

        missing_endpoints = set(tasks_by_model) - set(active_endpoints)
        if missing_endpoints:
            self.logger.error(
                "Missing endpoint configuration for model(s): %s",
                ", ".join(sorted(missing_endpoints)),
            )
            return False

        deps = self._build_dependencies()
        connectors = self._create_connectors(active_endpoints)

        runner = TaskRunner(
            tasks_by_model,
            connectors,
            deps.writer,
            int(self.config.get("RETRY_LIMIT", 3)),
            deps.shutdown_event,
            self.logger,
            model_aliases=dict(plan.model_aliases),
            metrics_collector=deps.metrics_collector,
            summary_reporter=deps.summary_reporter,
            batch_size=self.config.get("TASK_BATCH_SIZE", 1),
        )

        try:
            runner.run()
        except KeyboardInterrupt:
            self.logger.warning("KeyboardInterrupt detected. Initiating shutdown...")
            deps.shutdown_event.set()
        finally:
            for connector in connectors.values():
                connector.shutdown()
            deps.writer.flush()
            deps.summary_reporter.finalize(
                reason="normal_exit", session_end_time=time.time()
            )
            self.logger.info("=== Shutdown complete ===")

        return True

    # ------------------------------------------------------------------
    def _filter_titles_for_run(
        self,
        titles,
        result_checker: ExistingResultChecker,
        models: Iterable[str],
    ):
        """Remove titles that already have generated output when not testing."""
        if self.config.get("TEST_MODE"):
            return titles

        active_models = tuple(models)
        filtered = {}
        skipped_existing = 0
        skipped_by_model = 0

        for identifier, info in titles.items():
            if active_models and any(
                result_checker.has_model_entry(identifier, model)
                for model in active_models
            ):
                skipped_by_model += 1
                continue

            if result_checker.record_exists(identifier):
                skipped_existing += 1
                continue

            filtered[identifier] = info

        skipped_total = skipped_existing + skipped_by_model
        if skipped_total:
            details = []
            if skipped_existing:
                details.append(f"{skipped_existing} with existing JSON files")
            if skipped_by_model:
                details.append(f"{skipped_by_model} with entries for active model(s)")
            detail_msg = "; ".join(details)
            self.logger.info(
                "Skipping %s title(s) that already have generated output (%s).",
                skipped_total,
                detail_msg,
            )

        return filtered

    # ------------------------------------------------------------------
    def _build_dependencies(self) -> PipelineDependencies:
        """Construct shared helper objects and dependencies."""
        writer = ResultWriter(
            self.config["GENERATED_DIR"],
            strategy=self.config.get("WRITE_STRATEGY", "immediate"),
            flush_interval=self.config.get("WRITE_BATCH_SIZE", 1),
            flush_seconds=self.config.get("WRITE_BATCH_SECONDS", 5.0),
            flush_retry_limit=self.config.get("WRITE_BATCH_RETRY_LIMIT", 3),
            lock_timeout=self.config.get("FILE_LOCK_TIMEOUT", 10.0),
            lock_poll_interval=self.config.get("FILE_LOCK_POLL_INTERVAL", 0.1),
            lock_stale_seconds=self.config.get("FILE_LOCK_STALE_SECONDS", 300.0),
            logger=self.logger,
        )

        metrics_collector = MetricsCollector()
        summary_task_interval = int(self.config.get("METRICS_SUMMARY_TASK_INTERVAL", 0) or 0)
        summary_time_seconds = float(
            self.config.get(
                "METRICS_SUMMARY_TIME_SECONDS",
                self.config.get("SUMMARY_INTERVAL", 0.0),
            )
            or 0.0
        )

        summary_reporter = MetricsSummaryReporter(
            metrics_collector,
            self.logger,
            self.config["LOG_DIR"],
            summary_every_tasks=summary_task_interval,
            summary_every_seconds=summary_time_seconds,
        )
        summary_reporter.start()

        shutdown_event = threading.Event()
        ShutdownManager(
            shutdown_event,
            writer,
            self.logger,
            summary_reporter=summary_reporter,
        ).register()

        return PipelineDependencies(
            writer=writer,
            metrics_collector=metrics_collector,
            summary_reporter=summary_reporter,
            shutdown_event=shutdown_event,
        )

    # ------------------------------------------------------------------
    def _log_startup(self) -> None:
        self.logger.info("=== LLM Sequential Processor Starting ===")
        if self.config_sources:
            self.logger.info("Loaded config overrides from: %s", ", ".join(self.config_sources))
        else:
            self.logger.debug("No config override files found; using defaults.")

    # ------------------------------------------------------------------
    def _resolve_model_endpoints(self) -> Dict[str, str]:
        """Resolve LM Studio model endpoints."""
        base_url = str(self.config.get("LLM_BASE_URL", "http://localhost:1234")).rstrip("/")
        configured_models = self.config.get("LLM_MODELS") or []

        if isinstance(configured_models, str):
            configured_models = [m.strip() for m in configured_models.split(",") if m.strip()]

        if configured_models:
            self.logger.info("Using explicitly configured models: %s", ", ".join(configured_models))
            models = configured_models
        else:
            detected = get_model_keys(self.logger)
            if not detected:
                self.logger.error("No running LM Studio models detected.")
                return {}
            self.logger.info("Detected running LM Studio models: %s", ", ".join(detected))
            models = detected

        endpoints = {model: f"{base_url}/v1/chat/completions" for model in models}
        self.logger.debug("Resolved endpoints: %s", endpoints)
        return endpoints

    # ------------------------------------------------------------------
    def _prepare_directories(self) -> None:
        os.makedirs(self.config["GENERATED_DIR"], exist_ok=True)
        create_backup(self.config["BACKUP_DIR"], self.config["IGNORE_FOLDERS"], self.logger)

    # ------------------------------------------------------------------
    def _load_titles(self):
        cache_path = os.path.join(self.config["BASE_DIR"], "titles_index.json")
        self.config["CACHE_PATH"] = cache_path
        titles = load_cache(self.config, self.logger)
        if titles is None:
            titles = build_cache(self.config, self.logger)
        return titles

    # ------------------------------------------------------------------
    def _load_prompt(self) -> PromptBundle:
        return load_and_archive_prompt(self.config["BASE_DIR"], self.logger)

    # ------------------------------------------------------------------
    def _test_limit_per_model(self) -> int | None:
        limit = self.config.get("TEST_LIMIT_PER_MODEL")
        try:
            return int(limit)
        except (TypeError, ValueError):
            return None

    # ------------------------------------------------------------------
    def _log_plan(self, plan: BatchPlan) -> None:
        self.logger.info(
            "Prepared %s task(s) spanning %s model(s).",
            plan.total_tasks,
            len(plan.tasks_by_model),
        )

    # ------------------------------------------------------------------
    def _create_connectors(self, endpoints: Dict[str, str]):
        """Create ModelConnector instances for each active endpoint."""
        compliance_interval = int(self.config.get("COMPLIANCE_REMINDER_INTERVAL", 0) or 0)
        expected_language = self.config.get("EXPECTED_LANGUAGE") or "en"

        if compliance_interval > 0:
            self.logger.info(
                "Automatic JSON compliance reminders every %s headline(s).",
                compliance_interval,
            )

        connectors = {
            model: ModelConnector(
                model,
                url,
                int(self.config.get("REQUEST_TIMEOUT", 90)),
                self.logger,
                compliance_interval=compliance_interval,
                expected_language=expected_language,
            )
            for model, url in endpoints.items()
        }
        self.logger.info("Initialized %s connector(s).", len(connectors))
        return connectors
