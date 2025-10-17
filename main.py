# main.py

import json
import logging
import random
import signal
import time
from pathlib import Path
from statistics import mean, stdev
from threading import Event

from src.core.pipeline import BatchProcessingPipeline
from src.core.model_connector import ModelConnector
from src.core.writer import ResultWriter
from src.util.config_manager import CONFIG, CONFIG_SOURCES, DEFAULT_CONFIG
from src.util.logger_setup import resolve_log_level
from src.util.prompt_utils import (
    load_prompt,
    make_structured_prompt,
    try_parse_json,
    validate_entry,
)

# ---------- Static paths ----------
TITLES_PATH = Path("data/titles_index.json")
PROMPT_PATH = Path("data/prompt.txt")

# ---------- Derived config ----------
LOG_LEVEL = resolve_log_level(CONFIG.get("LOG_LEVEL"), default=logging.INFO)
GENERATED_DIR = Path(CONFIG.get("GENERATED_DIR", DEFAULT_CONFIG["GENERATED_DIR"]))
REQUEST_TIMEOUT = int(CONFIG.get("REQUEST_TIMEOUT", 90))
TEST_LIMIT = int(CONFIG.get("TEST_LIMIT_PER_MODEL", 10))

# Prefer configured model; fallback to explicit name
_cfg_models = CONFIG.get("LLM_MODELS") or []
if isinstance(_cfg_models, str):
    _cfg_models = [m.strip() for m in _cfg_models.split(",") if m.strip()]
MODEL = (_cfg_models[0] if _cfg_models else "creative-writing-model")

# Prefer configured base URL; fallback to localhost
LM_STUDIO_URL = f"{CONFIG.get('LLM_BASE_URL', 'http://localhost:1234')}/v1/chat/completions"


logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("LLMPlotBot")


def _install_signal_handlers(stop_event: Event):
    handled = []
    for name in ("SIGINT", "SIGTERM", "SIGHUP"):
        signum = getattr(signal, name, None)
        if signum is None:
            continue
        try:
            prev = signal.getsignal(signum)
            signal.signal(signum, lambda s, f, _prev=prev: _handle_signal(s, f, stop_event, _prev))
        except (ValueError, OSError):
            continue
        handled.append((signum, prev))
    def restore():
        for signum, prev in handled:
            try:
                signal.signal(signum, prev)
            except (ValueError, OSError):
                pass
    return restore


def _handle_signal(signum, frame, stop_event: Event, previous_handler):
    if not stop_event.is_set():
        try:
            name = signal.Signals(signum).name
        except ValueError:
            name = str(signum)
        logger.info("Received %s; requesting graceful shutdown.", name)
    stop_event.set()
    if signum == getattr(signal, "SIGINT", None):
        raise KeyboardInterrupt
    if callable(previous_handler):
        previous_handler(signum, frame)


def load_titles(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Title index not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    logger.info("Loaded %,d total title–ID pairs.", len(data))
    return data


def summarize_batch(times, valid_json_count, total_count):
    avg_time = mean(times) if times else 0.0
    sd_time = stdev(times) if len(times) > 1 else 0.0
    success_rate = (valid_json_count / total_count) * 100.0 if total_count else 0.0
    logger.info("=== BATCH SUMMARY ===")
    logger.info("Processed: %d", total_count)
    logger.info("Valid JSON: %d (%.1f%%)", valid_json_count, success_rate)
    logger.info("Avg response time: %.2fs (±%.2f)", avg_time, sd_time)
    logger.info("=====================\n")


def main():
    # Pipeline path when not testing
    if not bool(CONFIG.get("TEST_MODE")):
        logger.info("TEST_MODE is false. Running full pipeline.")
        logger.info("Loaded config sources: %s", ", ".join(CONFIG_SOURCES))
        pipeline = BatchProcessingPipeline(dict(CONFIG), logger=logger, config_sources=CONFIG_SOURCES)
        pipeline.run()
        return

    # Test harness path
    stop_event = Event()
    restore_signals = _install_signal_handlers(stop_event)

    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    writer = ResultWriter(
        str(GENERATED_DIR),
        strategy=str(CONFIG.get("WRITE_STRATEGY", "immediate")),
        flush_interval=int(CONFIG.get("WRITE_BATCH_SIZE", 25)),
        flush_seconds=float(CONFIG.get("WRITE_BATCH_SECONDS", 5.0)),
        flush_retry_limit=int(CONFIG.get("WRITE_BATCH_RETRY_LIMIT", 3)),
        lock_timeout=float(CONFIG.get("FILE_LOCK_TIMEOUT", 10.0)),
        lock_poll_interval=float(CONFIG.get("FILE_LOCK_POLL_INTERVAL", 0.1)),
        lock_stale_seconds=float(CONFIG.get("FILE_LOCK_STALE_SECONDS", 300.0)),
        logger=logger,
    )

    prompt = load_prompt(PROMPT_PATH)
    titles = load_titles(TITLES_PATH)

    connector = ModelConnector(MODEL, LM_STUDIO_URL, REQUEST_TIMEOUT, logger)

    logger.info("Sending initialization prompt to LM Studio. [INIT]")
    init_resp, _ = connector.send_to_model(prompt, "INIT")
    init_content = connector.extract_content(init_resp)
    logger.info("Instructions acknowledged on INIT.")

    if not init_content or not init_content.strip():
        logger.warning("Empty response from model. Aborting.")
        return
    if not ("CONFIRM" in init_content.upper() or init_content.strip().startswith("[")):
        logger.warning("Initialization message not a confirmation; continuing anyway.")
        print(init_content)

    keys = list(titles.keys())
    sample_n = min(TEST_LIMIT, len(keys))
    sample_keys = random.sample(keys, sample_n)
    times = []
    valid_json_count = 0

    for i, key in enumerate(sample_keys, start=1):
        if stop_event.is_set():
            logger.info("Shutdown requested. Stopping test harness.")
            break

        title = titles[key]["title"]
        logger.info("[%d/%d] %s: %s", i, sample_n, key, title[:80])

        structured_prompt = make_structured_prompt(title)
        resp, elapsed = connector.send_to_model(structured_prompt, key)
        msg = connector.extract_content(resp)
        parsed = try_parse_json(msg)

        print("\n--- RAW RESPONSE ---")
        print(msg)
        print("--------------------")

        if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict) and validate_entry(parsed[0]):
            writer.write(key, MODEL, "prompt_hash_placeholder", parsed[0])
            valid_json_count += 1
            logger.info("%s: ✅ Valid JSON saved.", key)
        else:
            logger.warning("%s: Invalid or incomplete JSON after %.2fs.", key, elapsed)
            print("❌ Invalid JSON returned.")
        print("--------------------\n")

        times.append(elapsed)

        if i % 10 == 0 or i == sample_n:
            summarize_batch(times, valid_json_count, i)

        time.sleep(0.5)

    writer.flush()
    restore_signals()
    logger.info("=== TEST RUN COMPLETED ===")


if __name__ == "__main__":
    main()
