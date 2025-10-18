"""Task planning logic for distributing headlines across LLM models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Mapping, Sequence, Tuple

from .prompting import PromptBundle
from .results import ResultStore


@dataclass(frozen=True)
class HeadlineRecord:
    identifier: str
    title: str
    metadata: Mapping[str, object] | None = None


@dataclass(frozen=True)
class TaskBatch:
    model: str
    prompt_text: str
    items: Tuple[HeadlineRecord, ...]


@dataclass(frozen=True)
class TaskPlan:
    batches: Tuple[TaskBatch, ...]

    @property
    def total_tasks(self) -> int:
        return sum(len(batch.items) for batch in self.batches)

    @property
    def models(self) -> Tuple[str, ...]:
        return tuple(sorted({batch.model for batch in self.batches}))

    def batches_for_model(self, model: str) -> Tuple[TaskBatch, ...]:
        return tuple(batch for batch in self.batches if batch.model == model)


class TaskPlanner:
    """Plan work items for each model while skipping completed outputs."""

    def __init__(
        self,
        prompt: PromptBundle,
        store: ResultStore,
        *,
        batch_size: int,
        test_limit_per_model: int | None = None,
    ) -> None:
        self.prompt = prompt
        self.store = store
        self.batch_size = max(1, int(batch_size))
        self.test_limit = max(0, int(test_limit_per_model or 0)) or None

    def build_plan(self, titles: Sequence[HeadlineRecord], endpoints: Mapping[str, str]) -> TaskPlan:
        composer = _PromptComposer(self.prompt)
        batches: List[TaskBatch] = []

        for model in sorted(endpoints):
            eligible = self._filter_titles(titles, model)
            if self.test_limit:
                eligible = eligible[: self.test_limit]
            for chunk in _chunked(eligible, self.batch_size):
                prompt_text = composer.compose(chunk)
                batches.append(TaskBatch(model=model, prompt_text=prompt_text, items=tuple(chunk)))
        return TaskPlan(tuple(batches))

    def _filter_titles(self, titles: Sequence[HeadlineRecord], model: str) -> List[HeadlineRecord]:
        filtered: List[HeadlineRecord] = []
        for record in titles:
            if not self.store.has_entry(record.identifier, model, self.prompt.prompt_hash):
                filtered.append(record)
        return filtered


class _PromptComposer:
    def __init__(self, bundle: PromptBundle) -> None:
        self.bundle = bundle

    def compose(self, items: Sequence[HeadlineRecord]) -> str:
        lines: List[str] = [self.bundle.prompt.strip(), "", "### HEADLINES"]
        for index, item in enumerate(items, start=1):
            lines.append(f"{index}. {item.title}")
        return "\n".join(lines).strip() + "\n"


def load_titles(path) -> List[HeadlineRecord]:
    import json
    from pathlib import Path

    location = Path(path)
    if not location.exists():
        raise FileNotFoundError(f"Title index not found: {path}")
    payload = json.loads(location.read_text(encoding="utf-8"))
    if isinstance(payload, Mapping):
        records = []
        for identifier, data in payload.items():
            title = data["title"] if isinstance(data, Mapping) else data
            if isinstance(title, str) and title.strip():
                records.append(HeadlineRecord(identifier=str(identifier), title=title.strip()))
        return sorted(records, key=lambda r: r.identifier)
    if isinstance(payload, Sequence):
        records = []
        for entry in payload:
            if isinstance(entry, Mapping):
                identifier = str(entry.get("id") or entry.get("identifier") or entry.get("slug"))
                title = entry.get("title")
            else:
                identifier = str(entry)
                title = entry
            if isinstance(title, str) and title.strip():
                records.append(HeadlineRecord(identifier=identifier, title=title.strip()))
        return records
    raise ValueError("Unsupported titles index structure")


def _chunked(items: Sequence[HeadlineRecord], size: int) -> Iterator[Tuple[HeadlineRecord, ...]]:
    size = max(1, int(size))
    chunk: List[HeadlineRecord] = []
    for item in items:
        chunk.append(item)
        if len(chunk) == size:
            yield tuple(chunk)
            chunk.clear()
    if chunk:
        yield tuple(chunk)


__all__ = ["HeadlineRecord", "TaskBatch", "TaskPlan", "TaskPlanner", "load_titles"]
