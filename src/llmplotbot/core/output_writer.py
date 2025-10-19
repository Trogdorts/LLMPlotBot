"""Utilities for writing per-job JSON results atomically."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping

from .job_manager import Job


class OutputWriter:
    def __init__(self, outputs_dir: str | Path, failed_dir: str | Path, *, logger) -> None:
        self.outputs_dir = Path(outputs_dir)
        self.failed_dir = Path(failed_dir)
        self.logger = logger
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.failed_dir.mkdir(parents=True, exist_ok=True)

    def write(
        self,
        job: Job,
        *,
        prompt_hash: str,
        successes: Mapping[str, Mapping[str, Any]],
        failures: Mapping[str, Mapping[str, Any]],
    ) -> Path:
        record = self._build_record(job, prompt_hash, successes, failures)
        directory = self.outputs_dir if successes else self.failed_dir
        path = directory / f"{job.identifier}.json"
        self._atomic_dump(path, record)
        self.logger.debug("Wrote result for %s to %s", job.identifier, path)
        return path

    def _build_record(
        self,
        job: Job,
        prompt_hash: str,
        successes: Mapping[str, Mapping[str, Any]],
        failures: Mapping[str, Mapping[str, Any]],
    ) -> Dict[str, Any]:
        models: Dict[str, Dict[str, Any]] = {}
        for model, payload in successes.items():
            models.setdefault(model, {})[prompt_hash] = {
                **payload,
                "meta": {
                    "prompt_hash": prompt_hash,
                    "status": "success",
                },
            }
        for model, payload in failures.items():
            models.setdefault(model, {})[prompt_hash] = {
                **payload,
                "meta": {
                    "prompt_hash": prompt_hash,
                    "status": "failed",
                },
            }
        status = "success" if successes else "failed"
        return {
            "title": job.title,
            "id": job.identifier,
            "file_path": job.file_path,
            "status": status,
            "llm_models": models,
        }

    def _atomic_dump(self, path: Path, payload: Dict[str, Any]) -> None:
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        tmp_path.replace(path)


__all__ = ["OutputWriter"]
