"""Persistent job queue backed by SQLite."""

from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

from llmplotbot.utils.titles import Headline

ISO_FORMAT = "%Y-%m-%dT%H:%M:%S"


@dataclass
class Job:
    identifier: str
    title: str
    file_path: str | None
    status: str
    retries: int
    last_error: str | None
    result_path: str | None
    prompt_hash: str | None
    updated_at: str


class JobManager:
    """Manages durable job state and provides workers with pending tasks."""

    def __init__(self, db_path: str | Path, *, logger) -> None:
        self.db_path = Path(db_path)
        self.logger = logger
        self._lock = threading.Lock()
        self._connect_kwargs = {"detect_types": sqlite3.PARSE_DECLTYPES}

    # ------------------------------------------------------------------
    def initialize(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    file_path TEXT,
                    status TEXT NOT NULL,
                    retries INTEGER NOT NULL DEFAULT 0,
                    last_error TEXT,
                    result_path TEXT,
                    prompt_hash TEXT,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.commit()
        self.recover_incomplete_jobs()

    # ------------------------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30, check_same_thread=False, **self._connect_kwargs)
        conn.row_factory = sqlite3.Row
        return conn

    # ------------------------------------------------------------------
    def recover_incomplete_jobs(self) -> None:
        now = self._timestamp()
        with self._connect() as conn:
            conn.execute(
                "UPDATE jobs SET status='pending', updated_at=? WHERE status='processing'",
                (now,),
            )
            conn.commit()

    # ------------------------------------------------------------------
    def seed_jobs(self, headlines: Iterable[Headline]) -> int:
        inserted = 0
        now = self._timestamp()
        with self._connect() as conn:
            conn.execute("BEGIN")
            for headline in headlines:
                result = conn.execute(
                    """
                    INSERT OR IGNORE INTO jobs (id, title, file_path, status, updated_at)
                    VALUES (?, ?, ?, 'pending', ?)
                    """,
                    (headline.identifier, headline.title, headline.source_path, now),
                )
                inserted += result.rowcount
            conn.commit()
        if inserted:
            self.logger.info("Seeded %d new job(s) into queue", inserted)
        return inserted

    # ------------------------------------------------------------------
    def fetch_job(self) -> Optional[Job]:
        with self._lock:
            with self._connect() as conn:
                conn.execute("BEGIN IMMEDIATE")
                row = conn.execute(
                    "SELECT * FROM jobs WHERE status='pending' ORDER BY updated_at LIMIT 1"
                ).fetchone()
                if not row:
                    conn.commit()
                    return None
                now = self._timestamp()
                conn.execute(
                    "UPDATE jobs SET status='processing', updated_at=? WHERE id=?",
                    (now, row["id"]),
                )
                conn.commit()
        return Job(
            identifier=row["id"],
            title=row["title"],
            file_path=row["file_path"],
            status="processing",
            retries=row["retries"],
            last_error=row["last_error"],
            result_path=row["result_path"],
            prompt_hash=row["prompt_hash"],
            updated_at=row["updated_at"],
        )

    # ------------------------------------------------------------------
    def mark_success(self, job_id: str, *, result_path: str, prompt_hash: str) -> None:
        now = self._timestamp()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE jobs
                SET status='completed', result_path=?, prompt_hash=?, last_error=NULL, updated_at=?
                WHERE id=?
                """,
                (result_path, prompt_hash, now, job_id),
            )
            conn.commit()

    # ------------------------------------------------------------------
    def mark_failure(self, job_id: str, *, error: str, retry: bool, retry_limit: int) -> None:
        now = self._timestamp()
        with self._connect() as conn:
            if retry:
                row = conn.execute("SELECT retries FROM jobs WHERE id=?", (job_id,)).fetchone()
                retries = int(row["retries"] if row else 0) + 1
                status = "pending" if retries < retry_limit else "failed"
                conn.execute(
                    """
                    UPDATE jobs
                    SET status=?, retries=?, last_error=?, updated_at=?
                    WHERE id=?
                    """,
                    (status, retries, error, now, job_id),
                )
            else:
                conn.execute(
                    """
                    UPDATE jobs
                    SET status='failed', last_error=?, updated_at=?
                    WHERE id=?
                    """,
                    (error, now, job_id),
                )
            conn.commit()

    # ------------------------------------------------------------------
    def pending_jobs(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) FROM jobs WHERE status='pending'").fetchone()
            return int(row[0]) if row else 0

    # ------------------------------------------------------------------
    def total_jobs(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()
            return int(row[0]) if row else 0

    # ------------------------------------------------------------------
    def _timestamp(self) -> str:
        return datetime.utcnow().strftime(ISO_FORMAT)


__all__ = ["Job", "JobManager"]
