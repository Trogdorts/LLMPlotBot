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
