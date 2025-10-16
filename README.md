# LLMPlotBot

## Prompt authoring workflow

Prompts live in a single `data/prompt.txt` file. The authoring convention keeps
the human-editable instructions at the top of the file and the non-negotiable
formatting requirements at the bottom. Separate the sections with a blank line
to keep the split clear.

The runtime keeps `prompt.txt` normalised in this order and archives a hashed
copy in `data/prompts/`. Update the instruction section directly to change
behaviour while preserving the formatting rules appended underneath.

## Result persistence

Model responses are saved incrementally to `data/generated_data/` as soon as
they are available. Writes are protected with lock files so that multiple
connectors or processes can safely write to the same result set. Configure the
behaviour via the following keys in `src/config.py`:

- `WRITE_STRATEGY`: `"immediate"` (default) writes every result as soon as it
  arrives; set to `"batch"` to buffer results.
- `WRITE_BATCH_SIZE` and `WRITE_BATCH_SECONDS`: thresholds for batch flushing.
- `FILE_LOCK_TIMEOUT`, `FILE_LOCK_POLL_INTERVAL`, and
  `FILE_LOCK_STALE_SECONDS`: control file-lock acquisition and stale lock
  cleanup.

Each result file is updated atomically and keeps a per-model, per-prompt hash of
the structured data returned by the LLM.

## Configuration and overrides

Defaults live in `src/config.py`. Override any value by creating a
`config.local.json` file at the project root or by pointing the
`LLMPLOTBOT_CONFIG` environment variable at another JSON file. Values are merged
deeply, so you can override just the keys you care about.

- `LLM_BLOCKLIST` removes unwanted models from consideration even if they are
  running or explicitly listed.
- `COMPLIANCE_REMINDER_INTERVAL` (0 disables) automatically replays the
  JSON-compliance reminder after every _N_ headlines to keep long sessions on
  track.

Active override sources are logged on start-up.

## Runtime metrics

Each run logs a summary with total runtime, success and failure rates, retry
counts, and per-model averages. Connector-level reminders (manual, automatic,
and multi-object response warnings) are aggregated in the summary so you can
spot models that drift off spec.
