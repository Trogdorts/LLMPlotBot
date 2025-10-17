from __future__ import annotations

from src.core.prompts import build_structured_prompt, make_structured_prompt


def test_make_structured_prompt_alias() -> None:
    title = "Test Title"
    assert make_structured_prompt(title) == build_structured_prompt(title)
    assert title in make_structured_prompt(title)
