"""Runtime orchestration for LLMPlotBot."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any, Dict

from .config import load_config
from core import (
    CheckpointManager,
    GracefulShutdown,
    JobManager,
    MetricsManager,
    OutputWriter,
    RetryConfig,
    SystemMonitor,
    WorkerPool,
)
from .logging_utils import configure_logging
from utils.prompts import PromptManager
from utils.titles import load_titles


class LLMPlotBotRuntime:
    def __init__(self, config: Dict[str, Any], logger) -> None:
        self.config = config
        self.logger = logger

    @classmethod
    def from_defaults(cls) -> "LLMPlotBotRuntime":
        result = load_config(include_sources=True)
        logging_config = dict(result.config.get("logging", {}))
        paths = result.config.get("paths", {})
        logging_config.setdefault("log_dir", paths.get("logs"))
        logger = configure_logging(logging_config)
        sources = ", ".join(result.sources) or "<defaults>"
        logger.info("Loaded configuration from: %s", sources)
        return cls(result.config, logger)

    async def run(self) -> bool:
        testing_config = self.config.get("testing", {})
        if testing_config.get("dry_run"):
            self.logger.info("Dry-run mode enabled; configuration validated successfully.")
            return True

        paths = self.config.get("paths", {})
        prompt_filename = self.config.get("prompt", {}).get("filename", "prompt.txt")
        prompt_manager = PromptManager(
            paths.get("prompts"),
            filename=prompt_filename,
            archive_dir=paths.get("prompt_archive"),
        )
        try:
            prompt_bundle = prompt_manager.load()
        except FileNotFoundError as exc:
            self.logger.error("Prompt file missing: %s", exc)
            return False
        self.logger.info("Loaded prompt hash %s", prompt_bundle.prompt_hash)

        try:
            titles = load_titles(paths.get("titles_index"))
        except FileNotFoundError as exc:
            self.logger.error("Titles index not found: %s", exc)
            return False
        except ValueError as exc:
            self.logger.error("Invalid titles index: %s", exc)
            return False
        if testing_config.get("enabled"):
            max_jobs = int(testing_config.get("max_jobs", 0))
            if max_jobs > 0:
                titles = titles[:max_jobs]
                self.logger.info("Test mode active: limiting to %d headline(s)", len(titles))

        job_manager = JobManager(paths.get("jobs_db"), logger=self.logger)
        job_manager.initialize()
        job_manager.seed_jobs(titles)
        if job_manager.pending_jobs() == 0:
            self.logger.warning("No pending jobs to process.")
            return False

        metrics = MetricsManager(
            paths.get("metrics"),
            report_interval=float(self.config.get("metrics", {}).get("report_interval", 10.0)),
            include_system=bool(self.config.get("metrics", {}).get("include_system", True)),
            logger=self.logger,
        )
        checkpoint_manager = CheckpointManager(
            paths.get("checkpoints"),
            interval_seconds=float(self.config.get("checkpoints", {}).get("interval_seconds", 30)),
            jobs_per_checkpoint=int(self.config.get("checkpoints", {}).get("jobs_per_checkpoint", 25)),
            logger=self.logger,
        )
        output_writer = OutputWriter(paths.get("outputs"), paths.get("failed"), logger=self.logger)
        shutdown = GracefulShutdown()
        retry_cfg = self.config.get("model", {}).get("retry", {})
        testing_limit = None
        if testing_config.get("enabled"):
            max_jobs = testing_config.get("max_jobs")
            if max_jobs:
                testing_limit = int(max_jobs)

        worker_pool = WorkerPool(
            job_manager=job_manager,
            output_writer=output_writer,
            metrics=metrics,
            checkpoint_manager=checkpoint_manager,
            shutdown=shutdown,
            retry_config=RetryConfig(
                max_attempts=int(retry_cfg.get("max_attempts", 3)),
                backoff_seconds=float(retry_cfg.get("backoff_seconds", 2.0)),
                max_backoff_seconds=float(retry_cfg.get("max_backoff_seconds", 30.0)),
            ),
            logger=self.logger,
            testing_limit=testing_limit,
        )

        monitor_stop = asyncio.Event()
        if self.config.get("metrics", {}).get("include_system", True):
            system_monitor = SystemMonitor(
                interval=5.0,
                metrics=metrics,
                logger=self.logger,
            )

            async def _mirror_shutdown() -> None:
                await shutdown.event.wait()
                monitor_stop.set()

            mirror_task = asyncio.create_task(_mirror_shutdown())
            monitor_task = asyncio.create_task(system_monitor.run(monitor_stop))
        else:
            monitor_stop.set()

            async def _noop() -> None:
                return None

            monitor_task = asyncio.create_task(_noop())
            mirror_task = asyncio.create_task(_noop())
        metrics_task = asyncio.create_task(metrics.run())

        try:
            await worker_pool.run(
                prompt=prompt_bundle,
                models=self.config.get("model", {}).get("models", []),
                base_url=str(self.config.get("model", {}).get("base_url")),
                timeout=float(self.config.get("model", {}).get("timeout", 120)),
                max_concurrency=int(self.config.get("model", {}).get("max_concurrency", 1)),
            )
            shutdown.trigger()
            monitor_stop.set()
        except Exception:
            shutdown.trigger()
            monitor_stop.set()
            raise
        finally:
            metrics.stop()
            await asyncio.gather(metrics_task, return_exceptions=True)
            await asyncio.gather(monitor_task, mirror_task, return_exceptions=True)
        summary = metrics.summary()
        self._write_summary(paths.get("summaries"), summary)
        return True

    def _write_summary(self, directory: str, summary: Dict[str, Any]) -> None:
        if not directory:
            return
        target_dir = Path(directory)
        target_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
        summary_path = target_dir / f"run-summary-{timestamp}.json"
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        self.logger.info("Run summary written to %s", summary_path)


__all__ = ["LLMPlotBotRuntime"]
