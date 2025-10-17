# LLMPlotBot

LLMPlotBot is a lightweight script for exercising locally hosted language models
through [LM Studio](https://lmstudio.ai/). It sends a structured prompt for a
handful of sample titles, validates that the model returns JSON, and stores each
valid response on disk.

## Project layout

```
main.py           # Entry point for running ad-hoc batches
src/core/         # Minimal connector and result-writer helpers
src/util/         # File locking used by the writer
src/config.py     # Default configuration values and simple overrides
config/config.json# Optional user overrides (created manually)
data/             # Prompt text, title index, and generated outputs
```

## Prerequisites

* Python 3.11+
* An LM Studio instance listening on `http://localhost:1234`
* `requests` (installed via `pip install -r requirements.txt`)

## Running a batch

1. Start LM Studio and make sure the desired model is loaded.
2. Update `data/prompt.txt` with the instruction template you want to send.
3. Ensure `data/titles_index.json` contains the titles to sample from.
4. Execute the runner:

```bash
python main.py
```

The script sends an initial confirmation prompt. After the model acknowledges
(or if you choose to continue without confirmation) a random sample of
`TEST_SAMPLE_SIZE` titles is selected and sent for generation. Raw responses are
printed to the console for inspection.

Valid JSON responses are persisted to `data/generated_data/<title_id>.json`.
Each file contains the most recent response grouped by model and prompt hash.

## Configuration

Default values live in `src/config.py`. To override them without editing source
code, create `config/config.json` and provide any subset of keys from
`DEFAULT_CONFIG`. For example:

```json
{
  "GENERATED_DIR": "./data/generated_data",
  "WRITE_STRATEGY": "batch",
  "WRITE_BATCH_SIZE": 10,
  "WRITE_BATCH_SECONDS": 2.0
}
```

Configuration values currently in use:

| Key | Description |
| --- | --- |
| `GENERATED_DIR` | Directory for persisted model responses. |
| `REQUEST_TIMEOUT` | Seconds to wait for LM Studio responses. |
| `WRITE_STRATEGY` | Either `"immediate"` (default) or `"batch"`. |
| `WRITE_BATCH_SIZE` | Number of queued responses before a batch flush. |
| `WRITE_BATCH_SECONDS` | Maximum time between batch flushes. |
| `WRITE_BATCH_RETRY_LIMIT` | Retry attempts when file locks fail. |
| `FILE_LOCK_TIMEOUT` | Seconds to wait for an existing lock to clear. |
| `FILE_LOCK_POLL_INTERVAL` | Sleep interval while waiting on a lock. |
| `FILE_LOCK_STALE_SECONDS` | Age at which a lock file is considered stale. |
| `LM_STUDIO_URL` | Endpoint for LM Studio's chat completions API. |
| `MODEL` | Model identifier passed to LM Studio. |
| `TITLES_PATH` | Path to the JSON file containing title metadata. |
| `PROMPT_PATH` | Path to the prompt template file. |
| `TEST_SAMPLE_SIZE` | Number of titles sampled per batch run. |

## Customising the prompt workflow

The helper `make_structured_prompt` in `main.py` defines the JSON schema the
model must fill out. Adjust the template to request different fields or to
change instructions. The script expects a JSON array containing at least one
object; `validate_entry` can be expanded with additional checks if you want more
rigorous validation.

## Development

* Run `pytest` to execute the lightweight unit tests.
* Use `python -m compileall main.py src` to perform a quick syntax check.
* Generated files are stored under `data/generated_data/`. Remove individual
  files to re-run prompts for specific titles.

This trimmed-down codebase is intentionally small so new features can be added
incrementally.
