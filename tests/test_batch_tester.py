from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List

import pytest

from src.config import Settings
from src.core.testing import BatchTester


class StubConnector:
    def __init__(self, responses: Dict[str, Dict[str, Any]]):
        self.responses = responses
        self.calls: List[str] = []

    def send_to_model(self, prompt: str, title_id: str):
        self.calls.append(title_id)
        response = self.responses.get(title_id)
        if response is None:
            raise AssertionError(f"Unexpected prompt for {title_id}")
        return response, response.get("elapsed", 0.1)

    @staticmethod
    def extract_content(response: Dict[str, Any]) -> str:
        return response["choices"][0]["message"]["content"]


class StubWriter:
    def __init__(self):
        self.records: List[Any] = []

    def write(self, *record: Any) -> None:
        self.records.append(record)

    def flush(self) -> None:  # pragma: no cover - interface requirement
        pass


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings.from_mapping(
        {
            "GENERATED_DIR": tmp_path,
            "REQUEST_TIMEOUT": 30,
            "WRITE_STRATEGY": "immediate",
            "WRITE_BATCH_SIZE": 1,
            "WRITE_BATCH_SECONDS": 1.0,
            "WRITE_BATCH_RETRY_LIMIT": 1,
            "FILE_LOCK_TIMEOUT": 1.0,
            "FILE_LOCK_POLL_INTERVAL": 0.1,
            "FILE_LOCK_STALE_SECONDS": 5.0,
            "LM_STUDIO_URL": "http://test",
            "MODEL": "test-model",
            "TITLES_PATH": tmp_path / "titles.json",
            "PROMPT_PATH": tmp_path / "prompt.txt",
            "TEST_SAMPLE_SIZE": 2,
        },
        base_dir=tmp_path,
    )


def test_batch_tester_persists_valid_entries(settings: Settings) -> None:
    responses = {
        "INIT": {
            "choices": [{"message": {"content": "CONFIRM"}}],
            "elapsed": 0.1,
        },
        "story-1": {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            [
                                {
                                    "title": "Story",
                                    "core_event": "Something happens.",
                                    "themes": ["a", "b"],
                                    "tone": "dramatic",
                                }
                            ]
                        )
                    }
                }
            ],
            "elapsed": 0.2,
        },
    }
    connector = StubConnector(responses)
    writer = StubWriter()
    tester = BatchTester(
        settings=settings,
        connector=connector,
        writer=writer,
        logger=logging.getLogger("test"),
        random_source=random.Random(0),
    )

    titles = {"story-1": {"title": "Story"}}
    stats = tester.run("prompt", titles)

    assert stats.processed == 1
    assert stats.valid_json == 1
    assert len(writer.records) == 1
    assert connector.calls == ["INIT", "story-1"]


def test_batch_tester_raises_on_empty_initial_response(settings: Settings) -> None:
    responses = {
        "INIT": {
            "choices": [{"message": {"content": ""}}],
            "elapsed": 0.1,
        }
    }
    connector = StubConnector(responses)
    writer = StubWriter()
    tester = BatchTester(
        settings=settings,
        connector=connector,
        writer=writer,
        logger=logging.getLogger("test"),
        random_source=random.Random(0),
    )

    with pytest.raises(RuntimeError):
        tester.run("prompt", {"story-1": {"title": "Story"}})
