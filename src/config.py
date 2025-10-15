
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
    "TEST_MODE": True,                # True = limit processing for quick smoke tests
    "TEST_LIMIT_PER_MODEL": 10,       # number of headlines per model when TEST_MODE is enabled
    "JSON_DIR": "./data/json",        # source directory scanned when building the cache
    "MAX_WORKERS": 8,                 # threads used while indexing JSON files
    "RETRY_LIMIT": 3,                 # per-task retry cap
    "REQUEST_TIMEOUT": 90,            # seconds for LLM HTTP request
    "LLM_BASE_URL": "http://localhost:1234",
    "LLM_MODELS": [],                # optional explicit list of model keys to use
    "LLM_ENDPOINTS": {},
}
