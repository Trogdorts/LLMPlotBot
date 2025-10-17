from __future__ import annotations

import json

from src.core.prompting import (
    is_valid_entry,
    parse_json_payload,
    try_parse_json,
    validate_entry,
)


def test_try_parse_json_alias() -> None:
    payload = json.dumps({"key": "value"})
    assert try_parse_json(payload) == {"key": "value"}
    assert parse_json_payload(payload) == {"key": "value"}


def test_validate_entry_alias() -> None:
    entry = {"core_event": "", "themes": [], "tone": ""}
    assert validate_entry(entry) is True
    assert is_valid_entry(entry) is True
