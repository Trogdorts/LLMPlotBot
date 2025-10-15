
"""
Central configuration for the LLM batch processor.
All paths are relative to the project root when running main.py.
"""

CONFIG = {
    "BASE_DIR": "./data",
    "BACKUP_DIR": "./backups",
    "LOG_DIR": "./logs",
    "GENERATED_DIR": "./data/generated_data",
    "IGNORE_FOLDERS": ["backups", ".venv", "__pycache__", "logs"],
    "TEST_MODE": True,                # True = process one batch then exit
    "TEST_BATCHES": 5,
    "BATCH_SIZE": 8,                  # headlines per request to an LLM
    "JSON_DIR": "./data/json",        # source directory scanned when building the cache
    "MAX_WORKERS": 8,                 # threads used while indexing JSON files
    "NUM_WORKERS": 8,                 # worker threads consuming batches
    "BATCH_TIMEOUT": 2.0,             # seconds to flush partial batch
    "RETRY_LIMIT": 3,                 # per-task retry cap
    "REQUEST_TIMEOUT": 90,            # seconds for LLM HTTP request
    "LLM_BASE_URL": "http://localhost:1234",
    "LLM_MODELS": [],                # optional explicit list of model keys to use
    "LLM_ENDPOINTS": {},
}
