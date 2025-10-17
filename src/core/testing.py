"""Batch testing workflow for exercising the language model."""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from statistics import mean, stdev
from typing import List, Mapping

from src.config import Settings
from src.core.prompts import hash_prompt, make_structured_prompt
from src.core.prompting import (
    format_debug_payload,
    iter_first_entry,
    try_parse_json,
    validate_entry,
)


@dataclass
class BatchStats:
    """Lightweight summary of a batch run."""

    processed: int = 0
    valid_json: int = 0
    timings: List[float] = field(default_factory=list)

    def record(self, *, elapsed: float, is_valid: bool) -> None:
        self.timings.append(elapsed)
        self.processed += 1
        if is_valid:
            self.valid_json += 1

    @property
    def success_rate(self) -> float:
        if not self.processed:
            return 0.0
        return (self.valid_json / self.processed) * 100

    @property
    def average_time(self) -> float:
        return mean(self.timings) if self.timings else 0.0

    @property
    def std_time(self) -> float:
        return stdev(self.timings) if len(self.timings) > 1 else 0.0

    def log_summary(self, logger: logging.Logger) -> None:
        summarize_batch(self, logger=logger)


def summarize_batch(stats: BatchStats, *, logger: logging.Logger) -> None:
    """Log a concise summary of ``stats`` using ``logger``."""

    logger.info("=== BATCH SUMMARY ===")
    logger.info("Processed: %s", stats.processed)
    logger.info("Valid JSON: %s (%.1f%%)", stats.valid_json, stats.success_rate)
    logger.info(
        "Avg response time: %.2fs (±%.2f)", stats.average_time, stats.std_time
    )
    logger.info("=====================\n")


class BatchTester:
    """Coordinate prompt generation, validation, and persistence."""

    def __init__(
        self,
        *,
        settings: Settings,
        connector,
        writer,
        logger: logging.Logger,
        random_source: random.Random | None = None,
    ) -> None:
        self.settings = settings
        self.connector = connector
        self.writer = writer
        self.logger = logger
        self.random = random_source or random.Random()

    def run(self, prompt: str, titles: Mapping[str, Mapping[str, str]]) -> BatchStats:
        """Execute a single test batch and return its summary statistics."""

        stats = BatchStats()
        self._initialise_model(prompt)

        sample_ids = self._select_titles(titles)
        if not sample_ids:
            self.logger.warning("No titles available for testing.")
            self.writer.flush()
            return stats

        for index, title_id in enumerate(sample_ids, start=1):
            title = self._extract_title(titles[title_id])
            self.logger.info(
                "[%s/%s] %s: %s",
                index,
                len(sample_ids),
                title_id,
                title[:80],
            )

            structured_prompt = make_structured_prompt(title)
            response, elapsed = self.connector.send_to_model(structured_prompt, title_id)
            content = self.connector.extract_content(response)

            parsed = try_parse_json(content)
            if parsed is not None:
                self.logger.debug(
                    "Parsed JSON for %s:\n%s",
                    title_id,
                    format_debug_payload(parsed),
                )

            valid_entry = next(iter_first_entry(parsed), None)
            is_valid = bool(valid_entry and validate_entry(valid_entry))

            if is_valid:
                prompt_hash = hash_prompt(structured_prompt)
                self.writer.write(title_id, self.settings.model, prompt_hash, valid_entry)
                self.logger.info("%s: ✅ Valid JSON saved.", title_id)
            else:
                self.logger.warning(
                    "%s: Invalid JSON returned after %.2fs.", title_id, elapsed
                )
                self.logger.debug("Invalid JSON payload for %s:\n%s", title_id, content)

            stats.record(elapsed=elapsed, is_valid=is_valid)

        self.writer.flush()
        stats.log_summary(self.logger)
        return stats

    # ------------------------------------------------------------------
    def _initialise_model(self, prompt: str) -> None:
        self.logger.info("Sending initialization prompt to LM Studio.")
        response, _ = self.connector.send_to_model(prompt, "INIT")
        content = self.connector.extract_content(response)

        if not content.strip():
            self.logger.warning("Empty response from model. Aborting run.")
            raise RuntimeError("Model failed to acknowledge initial prompt.")

        if "CONFIRM" in content.upper() or content.strip().startswith("["):
            self.logger.info("CONFIRM received; proceeding.")
        else:
            self.logger.warning(
                "Initialization message not a confirmation; continuing anyway."
            )
            self.logger.debug("Initialization response:\n%s", content)

    def _select_titles(self, titles: Mapping[str, Mapping[str, str]]) -> List[str]:
        keys = list(titles.keys())
        sample_size = min(self.settings.test_sample_size, len(keys))
        return self.random.sample(keys, sample_size) if sample_size else []

    @staticmethod
    def _extract_title(data: Mapping[str, str] | str) -> str:
        if isinstance(data, Mapping):
            return str(data.get("title", ""))
        return str(data)


__all__ = ["BatchStats", "BatchTester", "summarize_batch"]
