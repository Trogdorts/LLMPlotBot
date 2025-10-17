# LLMPlotBot

LLMPlotBot orchestrates end-to-end processing jobs for language models. It
loads prompts, distributes headline tasks across available model endpoints,
collects the responses, and persists the structured results alongside rich
runtime metrics.

## Prompt authoring workflow

Prompts live in a single `data/prompt.txt` file. The authoring convention keeps
the human-editable instructions at the top of the file and the non-negotiable
formatting requirements at the bottom. Separate the sections with a blank line
to keep the split clear.

The runtime keeps `prompt.txt` normalised in this order and archives a hashed
copy in `data/prompts/`. Update the instruction section directly to change
behaviour while preserving the formatting rules appended underneath.

## Result persistence

Model responses are saved immediately to `data/generated_data/`. Writes are
protected with lock files so that multiple connectors or processes can safely
write to the same result set. Configure the behaviour via the following keys in
`src/config.py`:

- `WRITE_RETRY_LIMIT`: number of per-file retry attempts before giving up.
- `FILE_LOCK_TIMEOUT`, `FILE_LOCK_POLL_INTERVAL`, and
  `FILE_LOCK_STALE_SECONDS`: control file-lock acquisition and stale lock
  cleanup.

Each result file is updated atomically and keeps a per-model, per-prompt hash of
the structured data returned by the LLM.

## Configuration and overrides

Defaults live in `src/config.py` and are mirrored into `config/default.json` the
first time the application starts. The loader keeps that file in sync with new
defaults so you can tweak values without losing upstream changes. Override any
value by creating a `config/config.local.json` file (preferred), by adding
`config.local.json` at the project root, or by pointing the `LLMPLOTBOT_CONFIG`
environment variable at another JSON file. Values are merged deeply, so you can
override just the keys you care about.

- `LLM_BLOCKLIST` removes unwanted models from consideration even if they are
  running or explicitly listed.
- `COMPLIANCE_REMINDER_INTERVAL` (0 disables) automatically replays the
  JSON-compliance reminder after every _N_ headlines to keep long sessions on
  track.
- `TASK_BATCH_SIZE` controls how many headlines are sent to each connector per
  LLM request. Increase it to process more titles per round-trip; decrease it
  if a model struggles with large payloads.

Active override sources are logged on start-up.

## Runtime metrics

Each run logs a summary with total runtime, success and failure rates, retry
counts, and per-model averages. Connector-level reminders (manual, automatic,
and multi-object response warnings) are aggregated in the summary so you can
spot models that drift off spec.

## Installation

The project targets Python 3.11+. Create a virtual environment and install the
dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running the processor

The default entry point lives in `main.py`. After activating your virtual
environment, execute:

```bash
python main.py
```

The CLI bootstraps logging, loads configuration overrides, resolves active LLM
endpoints, and then executes the processing pipeline. Logs, generated outputs, and
archives are stored under the directories referenced in the configuration.

## Configuring LLMPlotBot

Configuration defaults are defined in `src/config.py` and materialised to
`config/default.json` the first time you run the application. To customise
behaviour without modifying tracked files, create a
`config/config.local.json` file that overrides only the keys you need. The
loader performs a deep merge, so nested dictionaries are combined instead of
replaced.

For example, to point at a remote model endpoint:

```json
{
  "LLM_ENDPOINTS": {
    "gpt4": "http://example.com:8000/v1/completions"
  },
}
```

You can also place a `config.local.json` at the project root or provide an
absolute path via the `LLMPLOTBOT_CONFIG` environment variable if you want to
store overrides elsewhere:

```bash
export LLMPLOTBOT_CONFIG=/path/to/custom-config.json
python main.py
```

Key configuration options include:

- `LLM_ENDPOINTS`: Explicit mapping of model names to HTTP endpoints. If
  omitted, the pipeline queries LM Studio for available models and falls back
  to `LLM_BASE_URL` and `LLM_MODELS`.
- `TASK_BATCH_SIZE`: Number of tasks sent to each connector per request.
- `TEST_MODE` and `TEST_LIMIT_PER_MODEL`: Limit processing to a subset of
  titles for dry runs.
- `WRITE_RETRY_LIMIT`: Number of per-file retry attempts before giving up.
- `COMPLIANCE_REMINDER_INTERVAL`: Frequency (0 disables) of JSON-compliance
  reminders injected into long sessions.
- `LOG_DIR`, `GENERATED_DIR`, `BACKUP_DIR`: File system locations for runtime
  artefacts.
- `LOG_LEVEL`: Console log verbosity (e.g. `"DEBUG"`, `"INFO"`).

All active override sources are logged at startup, making it easy to confirm
which configuration files were applied.
