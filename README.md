# LLMPlotBot

LLMPlotBot orchestrates batch processing jobs for language models. It loads
headlines, prepares a structured prompt, dispatches work to one or more LLM
endpoints, and stores the responses together with basic runtime metrics.

The project targets Python 3.11+.

## Project layout

```
src/
├── core/              # Durable job queue, worker pool, metrics, and connectors
├── llmplotbot/        # Configuration loading, logging helpers, and runtime entry
└── utils/             # Prompt and headline helpers shared across components
```

`main.py` is the CLI entry point. Runtime artefacts (generated JSON, prompt
archives, and backups) are stored under `data/` and `backups/` by default.

## Configuration

Defaults live in `src/llmplotbot/config.py`. The first time the application
runs it materialises `config/config.json`. Additional overrides can be provided
via, in order of precedence:

1. `config/config.local.json`
2. `config.local.json` at the repository root
3. A file pointed to by the `LLMPLOTBOT_CONFIG` environment variable

Configuration files are deep-merged, so you only need to override the keys you
care about. Useful keys include:

- `LLM_ENDPOINTS`: Mapping of model name → HTTP endpoint. If omitted, the
  pipeline queries LM Studio (`LLM_BASE_URL`) or falls back to composing URLs
  from `LLM_MODELS`.
- `TASK_BATCH_SIZE`: Number of headlines bundled into a single request.
- `WRITE_STRATEGY`, `WRITE_BATCH_SIZE`, `WRITE_BATCH_SECONDS`: Result write
  buffering controls.
- `FILE_LOCK_TIMEOUT`, `FILE_LOCK_POLL_INTERVAL`, `FILE_LOCK_STALE_SECONDS`:
  Lock behaviour for generated JSON files.
- `TEST_MODE` / `TEST_LIMIT_PER_MODEL`: Limit processing to a subset of
  headlines.

## Prompt workflow

Prompts are stored in `data/prompt.txt`. Editing this file updates the dynamic
instructions; the formatting section is regenerated automatically from the
prompt specification defined in `src/utils/prompts.py`. A helper snippet
(`prompt_user_snippet.txt`) is created for quick iteration and every generated
prompt is archived under `data/prompts/` with a hash-based filename. The active
prompt contents are also backed up to `backups/prompt-<timestamp>.txt` at the
start of each run.

## Running the pipeline

Install dependencies into a virtual environment and execute `main.py`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

The CLI loads configuration, resolves endpoints, builds the plan, and iterates
over each batch sequentially. Results are written to `data/generated_data/` with
lock-protected atomic updates to avoid clobbering concurrent runs.

## Runtime summary

At the end of a run the pipeline logs a compact summary showing the number of
batches processed per model, successes vs failures, and the average response
time observed for each model. JSON output is saved per headline with a
`llm_models` section keyed by model name and prompt hash so repeated runs skip
already generated content.
