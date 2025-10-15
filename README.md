# LLMPlotBot

## Prompt authoring workflow

Prompts are now split into two files under `data/`:

- `prompt_instructions.txt` contains the editable guidance for how headlines
  should be interpreted and which data should be returned.
- `prompt_formatting.txt` contains the fixed formatting and output compliance
  requirements that keep responses machine-readable.

The combined prompt is automatically written to `prompt.txt` at runtime and a
hashed copy is archived in `data/prompts/`. Edit the instructions file to tweak
behaviour without risking accidental changes to the required formatting rules.

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
