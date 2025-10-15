"""
utils_io.py
===============
Handles high-performance JSON scanning, cache creation, and safe I/O.
"""

import os
import json
import gc
import threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

LOCK = threading.Lock()


def load_cache(config, logger):
    """Load titles_index.json if it exists in BASE_DIR."""
    cache_path = config["CACHE_PATH"]
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} cached entries from {cache_path}.")
        return data
    logger.warning("Cache file not found.")
    return None


def save_final(data, config, logger):
    """Save index data to BASE_DIR."""
    cache_path = config["CACHE_PATH"]
    with LOCK:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        logger.info(f"Saved {len(data)} entries to {cache_path}")
    gc.collect()


def read_json_file(path):
    """Safely read a JSON file and extract 'id' and 'title'."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            item = json.load(f)
        pid = item.get("id")
        title = item.get("title")
        if pid and title:
            return pid, title, path
    except Exception:
        return None
    return None


def build_cache(config, logger):
    """Walk JSON_DIR, extract IDs and titles using threads."""
    json_dir = config["JSON_DIR"]
    max_workers = config["MAX_WORKERS"]
    index = {}
    start = datetime.now()
    logger.info(f"Scanning {json_dir}")

    json_files = [
        os.path.join(root, name)
        for root, _, files in os.walk(json_dir)
        for name in files if name.endswith(".json")
    ]

    total = len(json_files)
    logger.info(f"Found {total} JSON files.")
    malformed = 0
    processed = 0

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for future in as_completed(executor.submit(read_json_file, p) for p in json_files):
                result = future.result()
                if result:
                    pid, title, path = result
                    index[pid] = {"title": title, "path": path}
                    processed += 1
                    if processed % 20000 == 0:
                        logger.info(f"Processed {processed}/{total}")
                else:
                    malformed += 1
                if processed % 50000 == 0:
                    gc.collect()
    except KeyboardInterrupt:
        logger.warning("KeyboardInterrupt — saving partial index.")
        save_final(index, config, logger)
        raise SystemExit(1)
    except Exception as e:
        logger.error(f"Error: {e}")

    save_final(index, config, logger)
    elapsed = (datetime.now() - start).total_seconds()
    logger.info(f"Completed indexing {len(index)} entries. Skipped {malformed} malformed files. Time: {elapsed:.2f}s")
    gc.collect()
    return index
