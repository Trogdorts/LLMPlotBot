"""Utility helpers for loading and caching title metadata."""

from __future__ import annotations

import gc
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Optional, Tuple

from .path_utils import normalize_for_logging

from .path_utils import normalize_for_logging

LOCK = threading.Lock()


def load_cache(config, logger):
    """Load titles_index.json if it exists in BASE_DIR."""
    cache_path = config["CACHE_PATH"]
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(
            "Loaded %s cached entries from %s.",
            len(data),
            normalize_for_logging(cache_path),
        )
        return data
    logger.warning("Cache file not found.")
    return None


def load_cache(config: Mapping[str, Any], logger) -> Optional[CacheIndex]:
    """Return the cached title index if it exists on disk."""

    cache_path = Path(config["CACHE_PATH"])
    if not cache_path.exists():
        logger.warning("Cache file not found.")
        return None

    with cache_path.open("r", encoding="utf-8") as handle:
        data: CacheIndex = json.load(handle)

    logger.info(
        "Loaded %s cached entries from %s.",
        len(data),
        normalize_for_logging(str(cache_path)),
    )
    return data


def save_final(data: CacheIndex, config: Mapping[str, Any], logger) -> None:
    """Persist the title index back to ``CACHE_PATH`` with basic locking."""

    cache_path = Path(config["CACHE_PATH"])
    with LOCK:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        logger.info(
            "Saved %s entries to %s",
            len(data),
            normalize_for_logging(cache_path),
        )
    gc.collect()


def _iter_json_files(root: Path) -> Iterator[Path]:
    for dirpath, _, filenames in os.walk(root):
        base = Path(dirpath)
        for name in filenames:
            if name.endswith(".json"):
                yield base / name


def _read_json_file(path: Path) -> Optional[Tuple[str, str, Path]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            item = json.load(handle)
    except Exception:
        return None

    identifier = item.get("id")
    title = item.get("title")
    if identifier and title:
        return str(identifier), str(title), path
    return None


def build_cache(config: Mapping[str, Any], logger) -> CacheIndex:
    """Scan ``JSON_DIR`` for headline files and build an in-memory index."""

    json_dir = Path(config.get("JSON_DIR", config["BASE_DIR"]))
    max_workers = config.get("MAX_WORKERS", 4)
    index: CacheIndex = {}
    start = datetime.now()
    logger.info("Scanning %s", normalize_for_logging(json_dir))

    logger.info("Scanning %s", normalize_for_logging(str(json_dir)))

    json_files = list(_iter_json_files(json_dir))
    total = len(json_files)
    logger.info("Found %s JSON files.", total)
    malformed = 0
    processed = 0

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_read_json_file, path) for path in json_files]
            for future in as_completed(futures):
                result = future.result()
                if result is None:
                    malformed += 1
                    continue

                identifier, title, path = result
                index[identifier] = {"title": title, "path": str(path)}
                processed += 1
                if processed % 20000 == 0:
                    logger.info("Processed %s/%s", processed, total)
                if processed % 50000 == 0:
                    gc.collect()
    except KeyboardInterrupt:
        logger.warning("KeyboardInterrupt — saving partial index.")
        save_final(index, config, logger)
        raise SystemExit(1)
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error("Error while building cache: %s", exc)

    save_final(index, config, logger)
    elapsed = (datetime.now() - start).total_seconds()
    logger.info(
        "Completed indexing %s entries. Skipped %s malformed files. Time: %.2fs",
        len(index),
        malformed,
        elapsed,
    )
    gc.collect()
    return index
