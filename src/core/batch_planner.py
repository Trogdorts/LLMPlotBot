"""Utilities for converting cached titles into per-model task queues."""

from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

from src.core.task import Task
from src.util.lmstudio_models import normalize_model_key
from src.util.result_utils import ExistingResultChecker


@dataclass(frozen=True)
class ModelWorkload:
    """Bundle of tasks assigned to a specific model instance."""

    model: str
    tasks: Sequence[Task]

    @property
    def count(self) -> int:
        return len(self.tasks)


@dataclass(frozen=True)
class BatchPlan:
    """Summary of all work to execute for the current prompt hash."""

    workloads: Tuple[ModelWorkload, ...]
    skipped_by_model: Mapping[str, int]
    model_aliases: Mapping[str, str]

    @property
    def tasks_by_model(self) -> Dict[str, List[Task]]:
        return {workload.model: list(workload.tasks) for workload in self.workloads}

    @property
    def total_tasks(self) -> int:
        return sum(workload.count for workload in self.workloads)

    @property
    def total_models(self) -> int:
        return len(self.workloads)

    @property
    def total_skipped(self) -> int:
        return sum(self.skipped_by_model.values())


class BatchPlanner:
    """Prepare per-model task queues taking into account caching and aliases."""

    def __init__(
        self,
        titles: Mapping[str, Mapping[str, object]],
        prompt_hash: str,
        *,
        prompt_dynamic: str,
        prompt_formatting: str,
        result_checker: ExistingResultChecker,
        logger,
        test_limit_per_model: int | None = None,
    ) -> None:
        self._titles = titles
        self._prompt_hash = prompt_hash
        self._prompt_dynamic = prompt_dynamic
        self._prompt_formatting = prompt_formatting
        self._result_checker = result_checker
        self._logger = logger
        self._test_limit = test_limit_per_model

    # ------------------------------------------------------------------
    def build(self, models: Iterable[str]) -> BatchPlan:
        model_groups: MutableMapping[str, List[str]] = OrderedDict()
        model_aliases: Dict[str, str] = {}
        for model in models:
            alias = normalize_model_key(model)
            model_aliases[model] = alias
            model_groups.setdefault(alias, []).append(model)

        if not model_groups:
            return BatchPlan(workloads=tuple(), skipped_by_model={}, model_aliases={})

        title_items: List[Tuple[str, Mapping[str, object]]] = list(self._titles.items())

        workloads: List[ModelWorkload] = []
        skipped_by_model: Dict[str, int] = {}

        for alias, instances in model_groups.items():
            if len(instances) > 1:
                self._logger.info(
                    "Distributing %s headline(s) across %s instances of %s.",
                    len(title_items),
                    len(instances),
                    alias,
                )

            buckets = self._distribute_titles(title_items, len(instances))
            for model, bucket in zip(instances, buckets):
                prepared, skipped = self._prepare_tasks(model, bucket)
                if skipped:
                    skipped_by_model[model] = skipped
                if prepared:
                    workloads.append(
                        ModelWorkload(model=model, tasks=tuple(prepared))
                    )

        workloads.sort(key=lambda wl: (-wl.count, wl.model))
        return BatchPlan(
            workloads=tuple(workloads),
            skipped_by_model=skipped_by_model,
            model_aliases=model_aliases,
        )

    # ------------------------------------------------------------------
    def _prepare_tasks(
        self,
        model: str,
        items: Sequence[Tuple[str, Mapping[str, object]]],
    ) -> Tuple[List[Task], int]:
        queued: List[Task] = []
        skipped = 0

        for identifier, info in items:
            if self._result_checker.has_entry(identifier, model, self._prompt_hash):
                skipped += 1
                continue

            title = str(info.get("title", "")).strip()
            if not title:
                self._logger.debug(
                    "Skipping empty title for id=%s model=%s", identifier, model
                )
                continue

            queued.append(
                Task(
                    id=identifier,
                    title=title,
                    model=model,
                    prompt_hash=self._prompt_hash,
                    prompt_dynamic=self._prompt_dynamic,
                    prompt_formatting=self._prompt_formatting,
                )
            )

            if self._test_limit is not None and len(queued) >= self._test_limit:
                break

        return queued, skipped

    # ------------------------------------------------------------------
    @staticmethod
    def _distribute_titles(
        items: Sequence[Tuple[str, Mapping[str, object]]],
        slots: int,
    ) -> List[List[Tuple[str, Mapping[str, object]]]]:
        if slots <= 1:
            return [list(items)]

        buckets: List[List[Tuple[str, Mapping[str, object]]]] = [list() for _ in range(slots)]
        ring = deque(range(slots))
        for identifier in items:
            slot = ring[0]
            buckets[slot].append(identifier)
            ring.rotate(-1)
        return buckets

