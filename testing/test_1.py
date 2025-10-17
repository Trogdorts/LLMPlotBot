import json
import logging
import requests
import time
import random
from pprint import pformat
from pathlib import Path
from statistics import mean, stdev

# ===== CONFIG =====
LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"
TITLES_PATH = r"C:\Users\criss\GIT\LLMPlotBot\data\titles_index.json"
PROMPT_PATH = "prompt.txt"
MODEL = "creative-writing-model"
TEST_SAMPLE_SIZE = 10
# ==================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def load_prompt(path: str) -> str:
    path = Path(path)
    if not path.exists():
        logger.error(f"Prompt file not found: {path}")
        raise FileNotFoundError(path)
    content = path.read_text(encoding="utf-8").strip()
    logger.debug(f"Prompt loaded ({len(content)} chars)")
    return content


def load_titles(path: str) -> dict:
    path = Path(path)
    if not path.exists():
        logger.error(f"Title index not found: {path}")
        raise FileNotFoundError(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    logger.info(f"Loaded {len(data):,} total title–ID pairs.")
    return data


def send_to_lm_studio(prompt: str, title_id: str) -> dict:
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "stream": False,
    }

    start = time.perf_counter()
    try:
        r = requests.post(LM_STUDIO_URL, json=payload)
    except requests.RequestException as e:
        logger.error(f"Connection error for {title_id}: {e}")
        raise

    elapsed = time.perf_counter() - start
    size = len(r.content)
    if not r.ok:
        logger.error(f"HTTP {r.status_code} for {title_id}: {r.text[:200]}")
        r.raise_for_status()

    try:
        data = r.json()
        logger.info(f"{title_id}: {elapsed:.2f}s | {size:,} bytes | HTTP {r.status_code}")
        return data, elapsed
    except Exception as e:
        logger.error(f"JSON decode error for {title_id}: {e}")
        logger.debug("Response text:\n%s", r.text)
        raise


def extract_content(response: dict) -> str:
    return response.get("choices", [{}])[0].get("message", {}).get("content", "").strip()


def try_parse_json(text: str):
    try:
        return json.loads(text)
    except Exception:
        return None


def make_structured_prompt(title: str) -> str:
    return f"""
You are a story-idea abstraction engine. 
Fill in the following JSON structure completely based on the given title.
Write natural, complete, realistic content for every field.
Return ONLY valid JSON. Do not add commentary, markdown, or explanation.

Title:
"{title}"

Required output schema:
[
  {{
    "title": "{title}",
    "core_event": "<one complete rewritten sentence under 50 words>",
    "themes": ["concept1", "concept2"],
    "tone": "<stylistic tone label>",
    "conflict_type": "<short phrase for the central tension>",
    "stakes": "<one concise sentence of what’s at risk>",
    "setting_hint": "<short location or situational hint>",
    "characters": ["role1", "role2"],
    "potential_story_hooks": ["hook1", "hook2"]
  }}
]
Output must start with [ and end with ] and be valid JSON.
"""


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

    logger.info("Sending initialization prompt to LM Studio.")
    response, _ = send_to_lm_studio(prompt, "INIT")
    content = extract_content(response)

    if "CONFIRM" not in content.upper():
        logger.warning("CONFIRM not detected. Printing model output:")
        logger.info("Model output:\n%s", content)
        return

    logger.info("CONFIRM received. Selecting random sample of titles for testing.")
    all_keys = list(titles.keys())
    sample_keys = random.sample(all_keys, TEST_SAMPLE_SIZE)
    logger.info(f"Selected {len(sample_keys)} random entries for test batch.")

    times = []
    valid_json_count = 0
    processed = 0

    for key in sample_keys:
        processed += 1
        title = titles[key]["title"]
        logger.info(f"[{processed}/{TEST_SAMPLE_SIZE}] {key}: {title[:80]}")

        structured_prompt = make_structured_prompt(title)
        resp, elapsed = send_to_lm_studio(structured_prompt, key)
        msg = extract_content(resp)
        parsed = try_parse_json(msg)

        logger.info("\n--- RAW RESPONSE ---")
        logger.info("%s", msg)
        logger.info("--------------------")

        logger.info("\n--- PARSED JSON ---")
        if parsed:
            logger.info("%s", pformat(parsed, sort_dicts=False))
            valid_json_count += 1
        else:
            logger.warning(f"{key}: Invalid JSON returned after {elapsed:.2f}s.")
            logger.info("❌ Invalid JSON returned.")
        logger.info("--------------------\n")

        times.append(elapsed)

        # Summary after every 10 processed (or at end)
        if processed % 10 == 0 or processed == TEST_SAMPLE_SIZE:
            summarize_batch(times, valid_json_count, processed)

        time.sleep(0.5)

    logger.info("=== TEST RUN COMPLETED ===")


if __name__ == "__main__":
    main()
