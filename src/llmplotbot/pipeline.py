"""High-level orchestration for the LLMPlotBot processing workflow."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Iterable, Mapping, MutableMapping

from .config import DEFAULT_CONFIG
from .connectors import ModelConnector, ModelRequest
from .endpoints import resolve_endpoints
from .metrics import MetricsTracker
from .planner import HeadlineRecord, TaskPlanner, load_titles
from .prompting import PromptManager
from .results import ResultStore, ResultWriter


class ProcessingPipeline:
    def __init__(
        self,
        config: MutableMapping[str, object],
        *,
        logger: logging.Logger,
        config_sources: Iterable[str] = (),
    ) -> None:
        self.config = dict(DEFAULT_CONFIG)
        self.config.update(dict(config))
        self.logger = logger
        self.config_sources = tuple(config_sources)

    # ------------------------------------------------------------------
    def run(self) -> bool:
        self._log_startup()

        endpoints = resolve_endpoints(self.config, self.logger)
        if not endpoints:
            self.logger.error("No active LLM endpoints resolved; aborting run.")
            return False

        prompt_manager = PromptManager(
            self.config.get("PROMPT_DIR", "data"),
            archive_dir=self.config.get("PROMPT_ARCHIVE_DIR"),
            prompt_filename=self.config.get("PROMPT_FILE"),
        )
        prompt_bundle = prompt_manager.load_prompt()
        self._create_prompt_backup(prompt_bundle.prompt)

        titles = load_titles(self.config.get("TITLES_INDEX", "data/titles_index.json"))
        if not titles:
            self.logger.error("No titles available for processing.")
            return False

        store = ResultStore(self.config.get("GENERATED_DIR", "data/generated_data"))
        planner = TaskPlanner(
            prompt_bundle,
            store,
            batch_size=int(self.config.get("TASK_BATCH_SIZE", 1)),
            test_limit_per_model=self.config.get("TEST_LIMIT_PER_MODEL") if self.config.get("TEST_MODE") else None,
        )
        plan = planner.build_plan(titles, endpoints)
        if not plan.total_tasks:
            self.logger.warning("No pending tasks after filtering existing results.")
            return False

        writer = ResultWriter(
            self.config.get("GENERATED_DIR", "data/generated_data"),
            strategy=str(self.config.get("WRITE_STRATEGY", "immediate")),
            flush_interval=int(self.config.get("WRITE_BATCH_SIZE", 25)),
            flush_seconds=float(self.config.get("WRITE_BATCH_SECONDS", 5.0)),
            retry_limit=int(self.config.get("WRITE_BATCH_RETRY_LIMIT", 3)),
            lock_timeout=float(self.config.get("FILE_LOCK_TIMEOUT", 10.0)),
            lock_poll_interval=float(self.config.get("FILE_LOCK_POLL_INTERVAL", 0.1)),
            lock_stale_seconds=float(self.config.get("FILE_LOCK_STALE_SECONDS", 300.0)),
            logger=self.logger,
        )
        metrics = MetricsTracker()

        connectors = {
            model: ModelConnector(
                model,
                endpoint,
                timeout=int(self.config.get("REQUEST_TIMEOUT", 90)),
                logger=self.logger,
            )
            for model, endpoint in endpoints.items()
        }

        for batch in plan.batches:
            connector = connectors[batch.model]
            request = ModelRequest(
                prompt=batch.prompt_text,
                items=[{"identifier": item.identifier, "title": item.title} for item in batch.items],
            )
            self.logger.info(
                "Dispatching %d headline(s) to %s.",
                len(batch.items),
                batch.model,
            )
            try:
                response = connector.invoke(request)
                raw = response["raw"]
                elapsed = float(response.get("elapsed", 0.0))
                content = ModelConnector.extract_text(raw)
                parsed = json.loads(content)
                records = self._coerce_records(parsed, batch.items)
            except Exception as exc:  # pragma: no cover - defensive logging
                self.logger.error("%s failed: %s", batch.model, exc, exc_info=True)
                metrics.record(batch.model, 0.0, len(batch.items), status="failure")
                continue

            if not records:
                self.logger.warning("%s returned no usable records.", batch.model)
                metrics.record(batch.model, 0.0, len(batch.items), status="failure")
                continue

            for item, record in zip(batch.items, records):
                record.setdefault("title", item.title)
                writer.write(item.identifier, batch.model, prompt_bundle.prompt_hash, record)
            metrics.record(batch.model, elapsed, len(records), status="success")

        writer.flush()
        summary = metrics.summary()
        self._log_summary(summary)
        for connector in connectors.values():
            connector.shutdown()
        return True

    # ------------------------------------------------------------------
    def _coerce_records(
        self,
        payload,
        items: Iterable[HeadlineRecord],
    ) -> list[dict]:
        records: list[dict] = []
        items_list = list(items)
        if isinstance(payload, Mapping):
            payload = [payload]
        if not isinstance(payload, list):
            raise ValueError("Model response was not a JSON object or list")
        if len(payload) < len(items_list):
            self.logger.warning(
                "Model returned %d record(s) for %d headline(s).",
                len(payload),
                len(items_list),
            )
        for raw, item in zip(payload, items_list):
            if isinstance(raw, Mapping):
                record = dict(raw)
            else:
                record = {"raw": raw}
            record.setdefault("title", item.title)
            records.append(record)
        return records

    def _create_prompt_backup(self, prompt_text: str) -> None:
        backup_dir = Path(self.config.get("BACKUP_DIR", "backups"))
        backup_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        backup_path = backup_dir / f"prompt-{timestamp}.txt"
        backup_path.write_text(prompt_text + "\n", encoding="utf-8")

    def _log_startup(self) -> None:
        info = {
            "config_sources": list(self.config_sources),
            "generated_dir": self.config.get("GENERATED_DIR"),
            "prompt_dir": self.config.get("PROMPT_DIR"),
        }
        self.logger.info("Starting LLMPlotBot with configuration: %s", json.dumps(info, indent=2))

    def _log_summary(self, summary: Mapping[str, object]) -> None:
        self.logger.info("=== PIPELINE SUMMARY ===")
        self.logger.info("Total batches: %s", summary.get("total_batches"))
        models = summary.get("models", {})
        if isinstance(models, Mapping):
            for model, stats in models.items():
                if isinstance(stats, Mapping):
                    self.logger.info(
                        "%s -> success=%s failure=%s avg_elapsed=%.2fs",
                        model,
                        stats.get("success"),
                        stats.get("failure"),
                        float(stats.get("avg_elapsed", 0.0)),
                    )
        self.logger.info("========================")


__all__ = ["ProcessingPipeline"]
