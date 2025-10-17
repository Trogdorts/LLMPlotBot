"""High-level orchestration for the LLM processing workflow."""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Dict, Iterable, MutableMapping

from src.core.batch_planner import BatchPlan, BatchPlanner
from src.core.metrics_collector import MetricsCollector
from src.core.metrics_summary import MetricsSummaryReporter
from src.core.model_connector import ModelConnector
from src.core.prompt_spec import PromptSpecification
from src.core.shutdown import ShutdownManager
from src.core.task_runner import TaskRunner
from src.core.writer import ResultWriter
from src.util.backup_utils import create_backup
from src.util.lmstudio_models import get_model_keys, group_model_keys
from src.util.prompt_utils import PromptBundle, load_and_archive_prompt
from src.util.result_utils import ExistingResultChecker
from src.util.utils_io import build_cache, load_cache


@dataclass(slots=True)
class PipelineDependencies:
    """Bundle of already-configured helper objects."""

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
        self.logger.info("Loaded %s titles for processing.", len(titles))

        prompt_bundle = self._load_prompt()
        prompt_hash = prompt_bundle.prompt_hash
        self._current_prompt_hash = prompt_hash
        self.logger.info("Active prompt hash: %s", prompt_hash)

        result_checker = ExistingResultChecker(
            self.config["GENERATED_DIR"], self.logger
        )

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
        connectors = self._create_connectors(active_endpoints, prompt_bundle.specification)

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
    def _log_startup(self) -> None:
        self.logger.info("=== LLM Sequential Processor Starting ===")
        if self.config_sources:
            self.logger.info("Loaded config overrides from: %s", ", ".join(self.config_sources))
        else:
            self.logger.debug("No config override files found; using defaults.")

    # ------------------------------------------------------------------
    def _resolve_model_endpoints(self) -> Dict[str, str]:
        config = self.config

        def _normalise_names(raw: object) -> list[str]:
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
                for model, url in dict(preconfigured).items()
                if model and model.lower() not in blocklist
            }
            if not filtered:
                self.logger.error(
                    "All configured LLM endpoints were filtered by LLM_BLOCKLIST."
                )
                return {}
            if len(filtered) != len(preconfigured):
                removed = sorted(set(preconfigured) - set(filtered))
                self.logger.warning(
                    "Excluding %s blocklisted model(s): %s",
                    len(removed),
                    ", ".join(removed),
                )
            self.logger.info(
                "Using pre-configured LLM endpoints for %s model(s).", len(filtered)
            )
            self.logger.debug("Pre-configured endpoints: %s", filtered)
            return filtered

        base_url = str(config.get("LLM_BASE_URL", "http://localhost:1234"))
        configured_models = _normalise_names(config.get("LLM_MODELS", []))

        if configured_models:
            self.logger.info(
                "Using explicitly configured LLM models: %s",
                ", ".join(configured_models),
            )
            candidate_models = configured_models
        else:
            detected = get_model_keys(self.logger)
            if not detected:
                self.logger.error(
                    "No running LM Studio models detected and no models configured."
                )
                return {}
            self.logger.info(
                "Detected running LM Studio models: %s", ", ".join(detected)
            )

            grouped = group_model_keys(detected)
            duplicates = {base: keys for base, keys in grouped.items() if len(keys) > 1}
            if duplicates:
                duplicate_summary = ", ".join(
                    f"{base} ×{len(keys)}" for base, keys in duplicates.items()
                )
                self.logger.info("Detected multi-instance models: %s", duplicate_summary)
            candidate_models = detected

        filtered_models = [
            model for model in candidate_models if model.lower() not in blocklist
        ]
        if not filtered_models:
            self.logger.error("No models available after applying LLM_BLOCKLIST filters.")
            return {}

        removed = sorted(set(candidate_models) - set(filtered_models))
        if removed:
            self.logger.warning(
                "Excluding %s blocklisted model(s): %s",
                len(removed),
                ", ".join(removed),
            )

        return {
            model: f"{base_url}/v1/chat/completions" for model in filtered_models
        }

    # ------------------------------------------------------------------
    def _prepare_directories(self) -> None:
        os.makedirs(self.config["GENERATED_DIR"], exist_ok=True)
        create_backup(
            self.config["BACKUP_DIR"],
            self.config.get("IGNORE_FOLDERS", []),
            self.logger,
        )

    # ------------------------------------------------------------------
    def _load_titles(self) -> Mapping[str, Mapping[str, object]]:
        cache_file = os.path.join(self.config["BASE_DIR"], "titles_index.json")
        self.config["CACHE_PATH"] = cache_file
        titles = load_cache(self.config, self.logger)
        if titles is None:
            titles = build_cache(self.config, self.logger)
        return titles or {}

    # ------------------------------------------------------------------
    def _load_prompt(self) -> PromptBundle:
        return load_and_archive_prompt(self.config["BASE_DIR"], self.logger)

    # ------------------------------------------------------------------
    def _build_dependencies(self) -> PipelineDependencies:
        shutdown_event = threading.Event()
        writer = ResultWriter(
            self.config["GENERATED_DIR"],
            retry_limit=int(self.config.get("WRITE_RETRY_LIMIT", 3) or 3),
            lock_timeout=float(self.config.get("FILE_LOCK_TIMEOUT", 10.0) or 10.0),
            lock_poll_interval=float(
                self.config.get("FILE_LOCK_POLL_INTERVAL", 0.1) or 0.1
            ),
            lock_stale_seconds=float(
                self.config.get("FILE_LOCK_STALE_SECONDS", 300.0) or 300.0
            ),
            logger=self.logger,
        )
        metrics_collector = MetricsCollector()
        summary_reporter = MetricsSummaryReporter(
            metrics_collector,
            self.logger,
            self.config["LOG_DIR"],
            summary_every_tasks=int(
                self.config.get("METRICS_SUMMARY_TASK_INTERVAL", 0) or 0
            ),
            summary_every_seconds=float(
                self.config.get("METRICS_SUMMARY_TIME_SECONDS",
                                self.config.get("SUMMARY_INTERVAL", 0.0))
                or 0.0
            ),
        )
        summary_reporter.start()
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
    def _test_limit_per_model(self) -> int | None:
        if not self.config.get("TEST_MODE"):
            return None
        limit = self.config.get("TEST_LIMIT_PER_MODEL")
        if limit is None:
            return None
        try:
            parsed = int(limit)
        except (TypeError, ValueError):
            self.logger.warning(
                "Invalid TEST_LIMIT_PER_MODEL value %r; ignoring per-model limit.",
                limit,
            )
            return None
        self.logger.info(
            "TEST_MODE enabled: limiting to %s headline(s) per model.", parsed
        )
        return parsed

    # ------------------------------------------------------------------
    def _log_plan(self, plan: BatchPlan) -> None:
        if plan.total_skipped:
            self.logger.info(
                "Skipping %s existing task(s) across %s model(s).",
                plan.total_skipped,
                len(plan.skipped_by_model),
            )
            for model, count in sorted(plan.skipped_by_model.items()):
                self.logger.debug(
                    "Model %s already has %s completed task(s) for prompt %s.",
                    model,
                    count,
                    self._current_prompt_hash or "<current>",
                )

        self.logger.info(
            "Prepared %s task(s) spanning %s model(s).",
            plan.total_tasks,
            plan.total_models,
        )

    # ------------------------------------------------------------------
    def _create_connectors(
        self, endpoints: Mapping[str, str], spec: PromptSpecification
    ) -> Dict[str, ModelConnector]:
        compliance_interval = int(
            self.config.get("COMPLIANCE_REMINDER_INTERVAL", 0) or 0
        )
        if compliance_interval > 0:
            self.logger.info(
                "Automatic JSON compliance reminders every %s headline(s).",
                compliance_interval,
            )

        connectors: Dict[str, ModelConnector] = {}
        for model, url in endpoints.items():
            connectors[model] = ModelConnector(
                model,
                url,
                int(self.config.get("REQUEST_TIMEOUT", 60) or 60),
                compliance_interval,
                self.logger,
                self.config.get("EXPECTED_LANGUAGE"),
                prompt_spec=spec,
            )
        self.logger.info("Initialized %s connector(s).", len(connectors))
        return connectors

