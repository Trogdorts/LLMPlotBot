import json
import logging
import random
import time
from pathlib import Path
from statistics import mean, stdev

from src.core.model_connector import ModelConnector
from src.core.writer import ResultWriter  # <-- reattached
from src.config import DEFAULT_CONFIG     # reuse paths if available
from src.util.prompt_utils import load_prompt, make_structured_prompt, try_parse_json, validate_entry

# ===== CONFIG =====
LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"
TITLES_PATH = Path("data/titles_index.json")
PROMPT_PATH = Path("data/prompt.txt")
MODEL = "creative-writing-model"
TEST_SAMPLE_SIZE = 10
GENERATED_DIR = Path(DEFAULT_CONFIG["GENERATED_DIR"])
# ==================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("LLMPlotBot")





def load_titles(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Title index not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    logger.info(f"Loaded {len(data):,} total title–ID pairs.")
    return data





def summarize_batch(times, valid_json_count, total_count):
    avg_time = mean(times) if times else 0
    sd_time = stdev(times) if len(times) > 1 else 0
    success_rate = (valid_json_count / total_count) * 100 if total_count else 0
    logger.info("=== BATCH SUMMARY ===")
    logger.info(f"Processed: {total_count}")
    logger.info(f"Valid JSON: {valid_json_count} ({success_rate:.1f}%)")
    logger.info(f"Avg response time: {avg_time:.2f}s (±{sd_time:.2f})")
    logger.info("=====================\n")


def main():
    prompt = load_prompt(PROMPT_PATH)
    titles = load_titles(TITLES_PATH)
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    writer = ResultWriter(GENERATED_DIR, logger=logger)

    connector = ModelConnector(MODEL, LM_STUDIO_URL, 90, logger)

    logger.info("Sending initialization prompt to LM Studio.")
    response, _ = connector.send_to_model(prompt, "INIT")
    content = connector.extract_content(response)

    if not content or not content.strip():
        logger.warning("Empty response from model. Aborting.")
        return

    if not ("CONFIRM" in content.upper() or content.strip().startswith("[")):
        logger.warning(f"Initialization message not a confirmation; continuing anyway. Raw Content: {content}")


    logger.info("CONFIRM received or skipped; proceeding.")
    sample_keys = random.sample(list(titles.keys()), TEST_SAMPLE_SIZE)

    times, valid_json_count = [], 0

    for i, key in enumerate(sample_keys, start=1):
        title = titles[key]["title"]
        logger.info(f"[{i}/{TEST_SAMPLE_SIZE}] {key}: {title[:80]}")
        structured_prompt = make_structured_prompt(title)

        resp, elapsed = connector.send_to_model(structured_prompt, key)
        msg = connector.extract_content(resp)
        parsed = try_parse_json(msg)
        logger.debug(f"RECEIVED RAW: {msg}")
        logger.debug(f"RECEIVED PARSED: {parsed}")

        if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
            first_entry = parsed[0]
            if validate_entry(first_entry):
                writer.write(key, MODEL, "prompt_hash_placeholder", first_entry)
                valid_json_count += 1
                logger.info(f"{key}: ✅ Valid JSON saved.")
            else:
                logger.warning(f"{key}: JSON missing required fields.")
        else:
            logger.warning(f"{key}: Invalid JSON returned after {elapsed:.2f}s.")

        times.append(elapsed)

        if i % 10 == 0 or i == TEST_SAMPLE_SIZE:
            summarize_batch(times, valid_json_count, i)
        time.sleep(0.5)

    logger.info("=== TEST RUN COMPLETED ===")


if __name__ == "__main__":
    main()
