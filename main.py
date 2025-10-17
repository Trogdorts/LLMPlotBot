import json
import logging
import math
import random
import signal
import time
from collections import Counter
from pathlib import Path
from statistics import mean, median, stdev

from threading import Event

from src.util.config_manager import (
    CONFIG,
    DEFAULT_CONFIG,
)  # reuse paths if available
from src.core.model_connector import ModelConnector
from src.core.writer import ResultWriter
from src.util.logger_setup import resolve_log_level
from src.util.prompt_utils import (
    load_prompt,
    make_structured_prompt,
    try_parse_json,
    validate_entry,
)

# ===== CONFIG =====
LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"
TITLES_PATH = Path("data/titles_index.json")
PROMPT_PATH = Path("data/prompt.txt")
MODEL = "creative-writing-model"
TEST_SAMPLE_SIZE = 10
CONSECUTIVE_FAILURE_LIMIT = 3
GENERATED_DIR = Path(DEFAULT_CONFIG["GENERATED_DIR"])
# ==================

logging.basicConfig(
    level=resolve_log_level(CONFIG.get("LOG_LEVEL"), default=logging.INFO),
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("LLMPlotBot")


def _install_signal_handlers(stop_event: Event):
    """Install signal handlers that request a graceful shutdown."""

    handled_signals = []
    for name in ("SIGINT", "SIGTERM", "SIGHUP"):
        signum = getattr(signal, name, None)
        if signum is None:
            continue
        try:
            previous = signal.getsignal(signum)
            signal.signal(
                signum,
                lambda s, f, *, _prev=previous: _handle_signal(s, f, stop_event, _prev),
            )
        except (ValueError, OSError):  # pragma: no cover - platform differences
            continue
        else:
            handled_signals.append((signum, previous))

    def restore():
        for signum, previous in handled_signals:
            try:
                signal.signal(signum, previous)
            except (ValueError, OSError):  # pragma: no cover - platform differences
                continue

    return restore


def _handle_signal(signum, frame, stop_event: Event, previous_handler):
    """Shared logic for installed signal handlers."""

    if not stop_event.is_set():
        try:
            name = signal.Signals(signum).name
        except ValueError:  # pragma: no cover - defensive
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
    logger.info(f"Loaded {len(data):,} total title–ID pairs.")
    return data


def _percentile(values, pct):
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * (pct / 100.0)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return float(ordered[int(rank)])
    lower_val = float(ordered[lower])
    upper_val = float(ordered[upper])
    return lower_val + (upper_val - lower_val) * (rank - lower)


def _streaks(records):
    max_success = max_failure = 0
    current_success = current_failure = 0
    for record in records:
        if record["success"]:
            current_success += 1
            current_failure = 0
            max_success = max(max_success, current_success)
        else:
            current_failure += 1
            current_success = 0
            max_failure = max(max_failure, current_failure)
    return max_success, max_failure, current_failure


def summarize_batch(records):
    total = len(records)
    latencies = [entry["elapsed"] for entry in records]
    successes = [entry for entry in records if entry["success"]]
    failures = [entry for entry in records if not entry["success"]]
    success_count = len(successes)
    failure_count = len(failures)
    success_rate = (success_count / total * 100.0) if total else 0.0
    avg_time = mean(latencies) if latencies else 0.0
    sd_time = stdev(latencies) if len(latencies) > 1 else 0.0
    median_time = median(latencies) if latencies else 0.0
    min_time = min(latencies) if latencies else 0.0
    max_time = max(latencies) if latencies else 0.0
    p90_time = _percentile(latencies, 90)
    p10_time = _percentile(latencies, 10)
    recent_window = records[-5:]
    recent_latencies = [entry["elapsed"] for entry in recent_window]
    recent_success_rate = (
        sum(1 for entry in recent_window if entry["success"]) / len(recent_window) * 100.0
        if recent_window
        else 0.0
    )
    recent_avg_latency = mean(recent_latencies) if recent_latencies else 0.0
    max_success_streak, max_failure_streak, current_failure_streak = _streaks(records)

    slowest = sorted(records, key=lambda entry: entry["elapsed"], reverse=True)[:3]
    fastest = sorted(records, key=lambda entry: entry["elapsed"])[:3]

    failure_reasons = Counter(
        entry["failure_reason"] for entry in failures if entry.get("failure_reason")
    )
    title_lengths = [
        entry["title_length"]
        for entry in records
        if entry.get("title_length") is not None
    ]
    avg_title_len = mean(title_lengths) if title_lengths else 0.0
    min_title_len = min(title_lengths) if title_lengths else 0
    max_title_len = max(title_lengths) if title_lengths else 0

    logger.info("=== BATCH SUMMARY ===")
    logger.info(
        "Totals    : processed=%d success=%d failures=%d success_rate=%.1f%%",
        total,
        success_count,
        failure_count,
        success_rate,
    )
    logger.info(
        "Latency   : avg=%.2fs median=%.2fs min=%.2fs max=%.2fs p10=%.2fs p90=%.2fs std=%.2fs",
        avg_time,
        median_time,
        min_time,
        max_time,
        p10_time,
        p90_time,
        sd_time,
    )
    logger.info(
        "Recent    : window=%d avg=%.2fs success_rate=%.1f%%",
        len(recent_window),
        recent_avg_latency,
        recent_success_rate,
    )
    logger.info(
        "Streaks   : longest_success=%d longest_failure=%d current_failure=%d",
        max_success_streak,
        max_failure_streak,
        current_failure_streak,
    )
    if title_lengths:
        logger.info(
            "Title len : avg=%.1f chars range=%d-%d",
            avg_title_len,
            min_title_len,
            max_title_len,
        )
    if slowest:
        logger.info(
            "Slowest   : %s",
            ", ".join(f"{entry['task_id']} ({entry['elapsed']:.2f}s)" for entry in slowest),
        )
    if fastest:
        logger.info(
            "Fastest   : %s",
            ", ".join(f"{entry['task_id']} ({entry['elapsed']:.2f}s)" for entry in fastest),
        )
    if failure_reasons:
        logger.info(
            "Failures  : %s",
            ", ".join(
                f"{reason} x{count}" for reason, count in failure_reasons.most_common()
            ),
        )
    logger.info("=====================\n")


def main():
    prompt = load_prompt(PROMPT_PATH)
    titles = load_titles(TITLES_PATH)
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    writer = ResultWriter(GENERATED_DIR, logger=logger)

    connector = ModelConnector(MODEL, LM_STUDIO_URL, 90, logger=logger)

    stop_event = Event()
    restore_signals = _install_signal_handlers(stop_event)

    def resend_instructions(tag: str) -> bool:
        if stop_event.is_set():
            logger.info("Skipping %s instruction resend due to shutdown request.", tag)
            return False
        logger.info("Sending initialization prompt to LM Studio. [%s]", tag)
        try:
            response, _ = connector.send_to_model(prompt, tag)
        except Exception:
            logger.exception("Failed to deliver instructions on %s.", tag)
            return False

        content = connector.extract_content(response)

        if not content or not content.strip():
            logger.warning("Empty response from model when sending %s instructions.", tag)
            return False

        if not ("CONFIRM" in content.upper() or content.strip().startswith("[")):
            logger.warning(
                "Instruction acknowledgement unexpected on %s; continuing anyway. Raw Content: %s",
                tag,
                content,
            )
        else:
            logger.info("Instructions acknowledged on %s.", tag)
        return True

    try:
        if not resend_instructions("INIT"):
            logger.warning("Initial instruction handshake failed; aborting run.")
            return
        sample_keys = random.sample(list(titles.keys()), TEST_SAMPLE_SIZE)

        batch_records = []
        last_summary_size = 0

        consecutive_failures = 0

        for i, key in enumerate(sample_keys, start=1):
            if stop_event.is_set():
                logger.info("Shutdown requested; stopping before processing remaining tasks.")
                break

            title = titles[key]["title"]
            logger.info(f"[{i}/{TEST_SAMPLE_SIZE}] {key}: {title[:80]}")
            structured_prompt = make_structured_prompt(title)

            success = False
            failure_reason = None
            elapsed = 0.0

            try:
                resp, elapsed = connector.send_to_model(structured_prompt, key)
                msg = connector.extract_content(resp)
                parsed = try_parse_json(msg)
                logger.debug(f"RECEIVED RAW: {msg}")
                logger.debug(f"RECEIVED PARSED: {parsed}")
            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received; finalizing run.")
                stop_event.set()
                break
            except Exception:
                logger.exception("Unexpected error while processing %s.", key)
                failure_reason = "unexpected_exception"
                consecutive_failures += 1
            else:
                if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
                    first_entry = parsed[0]
                    if validate_entry(first_entry):
                        writer.write(key, MODEL, "prompt_hash_placeholder", first_entry)
                        success = True
                        logger.info(f"{key}: ✅ Valid JSON saved.")
                        consecutive_failures = 0
                    else:
                        logger.warning(f"{key}: JSON missing required fields.")
                        failure_reason = "missing_required_fields"
                        consecutive_failures += 1
                else:
                    logger.warning(f"{key}: Invalid JSON returned after {elapsed:.2f}s.")
                    failure_reason = "invalid_json"
                    consecutive_failures += 1

            if failure_reason and not success:
                logger.debug("%s marked as failure due to %s.", key, failure_reason)

            batch_records.append(
                {
                    "task_id": key,
                    "elapsed": elapsed,
                    "success": success,
                    "timestamp": time.time(),
                    "title_length": len(title),
                    "failure_reason": failure_reason,
                }
            )

            if consecutive_failures >= CONSECUTIVE_FAILURE_LIMIT:
                logger.warning(
                    "Detected %s consecutive JSON failures. Resending instructions...",
                    consecutive_failures,
                )
                handshake_success = resend_instructions("REINSTRUCT")
                consecutive_failures = 0 if handshake_success else consecutive_failures

            if i % 10 == 0 or i == TEST_SAMPLE_SIZE:
                summarize_batch(batch_records)
                last_summary_size = len(batch_records)

            if stop_event.is_set():
                logger.info("Shutdown requested; ending loop before delay.")
                break
            time.sleep(0.5)

        if batch_records and last_summary_size != len(batch_records):
            summarize_batch(batch_records)
        logger.info("=== TEST RUN COMPLETED ===")
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received outside main loop; shutting down.")
    finally:
        stop_event.set()
        try:
            restore_signals()
        finally:
            try:
                writer.flush()
            finally:
                connector.shutdown()
        logger.info("Resources cleaned up. Exiting.")


if __name__ == "__main__":
    main()
