import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.writer import ResultWriter


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_write_persists_single_record(tmp_path: Path):
    writer = ResultWriter(tmp_path)
    writer.write(
        "sample-id",
        "model-alpha",
        "hash-1",
        {
            "title": "Original Title",
            "core_event": "A concise summary.",
        },
    )

    output_path = tmp_path / "sample-id.json"
    assert output_path.exists()

    data = read_json(output_path)
    assert data["title"] == "Original Title"
    assert data["llm_models"]["model-alpha"]["hash-1"] == {
        "core_event": "A concise summary."
    }


def test_write_many_merges_records_without_title_clobber(tmp_path: Path):
    writer = ResultWriter(tmp_path)

    writer.write(
        "shared-id",
        "model-alpha",
        "hash-1",
        {
            "title": "Canonical Title",
            "core_event": "Initial summary.",
        },
    )

    writer.write_many(
        "shared-id",
        [
            (
                "model-beta",
                "hash-2",
                {
                    "title": "",
                    "core_event": "Secondary summary.",
                },
            ),
            (
                "model-alpha",
                "hash-3",
                {
                    "title": "Conflicting Title",
                    "core_event": "Updated summary.",
                },
            ),
        ],
    )

    data = read_json(tmp_path / "shared-id.json")
    assert data["title"] == "Canonical Title"

    model_alpha = data["llm_models"]["model-alpha"]
    assert model_alpha["hash-1"] == {"core_event": "Initial summary."}
    assert model_alpha["hash-3"] == {"core_event": "Updated summary."}

    model_beta = data["llm_models"]["model-beta"]
    assert model_beta["hash-2"] == {"core_event": "Secondary summary."}
