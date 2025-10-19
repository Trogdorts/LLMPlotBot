"""Concurrent worker orchestration for processing jobs."""

from __future__ import annotations

import asyncio
import json
import math
import time
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, MutableMapping

import httpx

from .checkpoint_manager import CheckpointManager, CheckpointState
from .job_manager import Job, JobManager
from .ollama import OllamaConnector
from .output_writer import OutputWriter
from ..utils.prompts import PromptBundle


@dataclass(frozen=True)
class RetryConfig:
    max_attempts: int
    backoff_seconds: float
    max_backoff_seconds: float


class WorkerPool:
    def __init__(
        self,
        *,
        job_manager: JobManager,
        output_writer: OutputWriter,
        metrics,
        checkpoint_manager: CheckpointManager,
        shutdown,
        retry_config: RetryConfig,
        logger,
        testing_limit: int | None = None,
    ) -> None:
        self.job_manager = job_manager
        self.output_writer = output_writer
        self.metrics = metrics
        self.checkpoint_manager = checkpoint_manager
        self.shutdown = shutdown
        self.retry_config = retry_config
        self.logger = logger
        self.testing_limit = testing_limit
        self._counter_lock = asyncio.Lock()
        self._completed = 0
        self._failed = 0
        self._processed = 0

    async def run(
        self,
        *,
        prompt: PromptBundle,
        models: Iterable[str],
        base_url: str,
        timeout: float,
        max_concurrency: int,
    ) -> None:
        self.shutdown.install()
        model_list = list(models)
        tasks = [
            asyncio.create_task(
                self._worker(idx, prompt, model_list, base_url, timeout),
                name=f"worker-{idx}",
            )
            for idx in range(max(1, max_concurrency))
        ]
        await asyncio.gather(*tasks)

    async def _worker(
        self,
        worker_id: int,
        prompt: PromptBundle,
        models: list[str],
        base_url: str,
        timeout: float,
    ) -> None:
        connectors = {
            model: OllamaConnector(base_url=base_url, model=model, timeout=timeout, logger=self.logger)
            for model in models
        }
        try:
            while not self.shutdown.is_triggered():
                if not await self._can_process_next():
                    break
                job = await asyncio.to_thread(self.job_manager.fetch_job)
                if not job:
                    pending = await asyncio.to_thread(self.job_manager.pending_jobs)
                    if pending == 0:
                        break
                    await asyncio.sleep(0.5)
                    continue
                await self._process_job(job, prompt, connectors)
        finally:
            for connector in connectors.values():
                try:
                    await connector.aclose()
                except Exception:  # pragma: no cover - cleanup best effort
                    pass

    async def _can_process_next(self) -> bool:
        if self.testing_limit is None:
            return True
        async with self._counter_lock:
            return self._processed < self.testing_limit

    async def _process_job(
        self,
        job: Job,
        prompt: PromptBundle,
        connectors: Mapping[str, OllamaConnector],
    ) -> None:
        successes: Dict[str, MutableMapping[str, object]] = {}
        failures: Dict[str, MutableMapping[str, object]] = {}
        for model, connector in connectors.items():
            if self.shutdown.is_triggered():
                break
            result = await self._invoke_model(job, model, connector, prompt)
            if result["status"] == "success":
                successes[model] = result["payload"]
            else:
                failures[model] = {"error": result["error"]}
        path = self.output_writer.write(
            job,
            prompt_hash=prompt.prompt_hash,
            successes=successes,
            failures=failures,
        )
        if successes:
            self.job_manager.mark_success(job.identifier, result_path=str(path), prompt_hash=prompt.prompt_hash)
            await self._update_counters(completed=1, failed=0, last_job_id=job.identifier)
        else:
            error_messages = "; ".join(payload.get("error", "") for payload in failures.values())
            self.job_manager.mark_failure(
                job.identifier,
                error=error_messages or "Model failure",
                retry=False,
                retry_limit=self.retry_config.max_attempts,
            )
            await self._update_counters(completed=0, failed=1, last_job_id=job.identifier)

    async def _invoke_model(
        self,
        job: Job,
        model: str,
        connector: OllamaConnector,
        prompt: PromptBundle,
    ) -> Dict[str, object]:
        attempt = 0
        parse_attempts = 0
        last_error: str | None = None
        while attempt < self.retry_config.max_attempts and not self.shutdown.is_triggered():
            try:
                response = await connector.generate(prompt.prompt, job.title)
                payload = self._parse_response(response["text"])
                self.metrics.record_success(model, elapsed=response.get("elapsed", 0.0), tokens=response.get("tokens", 0))
                return {"status": "success", "payload": payload}
            except json.JSONDecodeError as exc:
                parse_attempts += 1
                last_error = f"ParseError: {exc}".strip()
                if parse_attempts >= 2:
                    self.metrics.record_failure(model)
                    break
                self.metrics.record_retry(model)
                await asyncio.sleep(self._backoff(parse_attempts))
            except httpx.HTTPError as exc:
                attempt += 1
                last_error = f"HTTPError: {exc}"
                self.metrics.record_retry(model)
                if attempt >= self.retry_config.max_attempts:
                    self.metrics.record_failure(model)
                    break
                await asyncio.sleep(self._backoff(attempt))
            except Exception as exc:  # pragma: no cover - defensive
                attempt += 1
                last_error = f"Error: {exc}"
                self.metrics.record_retry(model)
                if attempt >= self.retry_config.max_attempts:
                    self.metrics.record_failure(model)
                    break
                await asyncio.sleep(self._backoff(attempt))
        return {"status": "failed", "error": last_error or "Unknown error"}

    def _parse_response(self, text: str) -> MutableMapping[str, object]:
        if not text:
            raise json.JSONDecodeError("Empty response", text, 0)
        parsed = json.loads(text)
        if isinstance(parsed, MutableMapping):
            return dict(parsed)
        if isinstance(parsed, list):
            return {"items": parsed}
        return {"value": parsed}

    async def _update_counters(self, *, completed: int, failed: int, last_job_id: str) -> None:
        pending = await asyncio.to_thread(self.job_manager.pending_jobs)
        timestamp = time.time()
        async with self._counter_lock:
            self._completed += completed
            self._failed += failed
            self._processed += completed + failed
            state = CheckpointState(
                last_job_id=last_job_id,
                total_completed=self._completed,
                total_failed=self._failed,
                pending=pending,
                timestamp=timestamp,
            )
        self.checkpoint_manager.maybe_checkpoint(state)

    def _backoff(self, attempt: int) -> float:
        delay = self.retry_config.backoff_seconds * math.pow(2, max(0, attempt - 1))
        return float(min(self.retry_config.max_backoff_seconds, delay))


__all__ = ["WorkerPool", "RetryConfig"]
