"""Prompt authoring and archival utilities."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable, MutableMapping, Sequence
from datetime import datetime

USER_SNIPPET_FILENAME = "prompt_user_snippet.txt"
PROMPT_FILENAME = "prompt.txt"
PROMPT_METADATA_FILENAME = "prompt_index.json"


@dataclass(frozen=True)
class PromptField:
    name: str
    kind: str
    description: str
    constraints: Sequence[str]
    example: Sequence[str] | str

    def is_list(self) -> bool:
        return self.kind.lower() in {"array", "list"}


@dataclass(frozen=True)
class ReplacementExample:
    description: str
    original: str
    transformed: str


@dataclass(frozen=True)
class PromptSpecification:
    default_user_prompt: str
    objective: str
    transformation_steps: Sequence[str]
    style_rules: Sequence[str]
    quality_checks: Sequence[str]
    formatting_rules: Sequence[str]
    fields: Sequence[PromptField]
    replacements: Sequence[ReplacementExample]


@dataclass(frozen=True)
class PromptBundle:
    prompt: str
    prompt_hash: str
    dynamic_section: str
    formatting_section: str
    specification: PromptSpecification


DEFAULT_SPECIFICATION = PromptSpecification(
    default_user_prompt=(
        "You are a story-idea abstraction engine that turns absurd real-world news "
        "headlines into structured story seeds. Preserve the irony and factual core "
        "while keeping sentences below 50 words."
    ),
    objective="Convert each numbered headline into a compact story seed ready for development.",
    transformation_steps=(
        "Rewrite the headline as one complete sentence preserving essential details.",
        "Normalise specific names that are not critical to understanding the event.",
        "Extract conceptual metadata that helps a writer continue the story.",
    ),
    style_rules=(
        "Keep language factual yet vivid.",
        "Avoid sensational embellishment or memes.",
        "Do not add quotes or hashtags.",
    ),
    quality_checks=(
        "Sentences remain under 50 words.",
        "No facts are added or contradicted.",
        "Array values are concise phrases without trailing punctuation.",
    ),
    formatting_rules=(
        "Return valid JSON only.",
        "Respond with a list containing one object per headline.",
        "Keep keys in the documented order and never omit them.",
    ),
    fields=(
        PromptField(
            name="title",
            kind="string",
            description="A rewritten, natural-sounding title.",
            constraints=("Under 50 words.", "Preserves factual content."),
            example="{original_title}",
        ),
        PromptField(
            name="core_event",
            kind="string",
            description="One-sentence summary of the rewritten event.",
            constraints=("Complete sentence.", "Neutral voice."),
            example="A cautious example sentence for headline {n}.",
        ),
        PromptField(
            name="themes",
            kind="array",
            description="High-level concepts present in the story.",
            constraints=("2-3 entries.", "Short lowercase or Title Case fragments."),
            example=("societal tension", "unexpected consequences"),
        ),
        PromptField(
            name="tone",
            kind="string",
            description="Stylistic tone label.",
            constraints=("Single short phrase.",),
            example="deadpan irony",
        ),
        PromptField(
            name="conflict_type",
            kind="string",
            description="Phrase describing the central tension.",
            constraints=("Short phrase.",),
            example="institution vs. public",
        ),
        PromptField(
            name="stakes",
            kind="string",
            description="Sentence capturing what is at risk.",
            constraints=("Complete sentence.",),
            example="Community trust in headline {n} is strained by opaque decisions.",
        ),
        PromptField(
            name="setting_hint",
            kind="string",
            description="Brief hint about setting or context.",
            constraints=("Short fragment.",),
            example="Midwestern town hall",
        ),
        PromptField(
            name="characters",
            kind="array",
            description="List of archetypal characters involved.",
            constraints=("2-3 entries."),
            example=("skeptical resident", "overworked official"),
        ),
        PromptField(
            name="potential_story_hooks",
            kind="array",
            description="Hooks or questions a writer could explore.",
            constraints=("2-3 entries."),
            example=("What secret deal created this policy?", "Who profits from the confusion?"),
        ),
    ),
    replacements=(
        ReplacementExample(
            description="Neutralise personal names when role is more important.",
            original="Mayor Jane Doe announces mysterious midnight curfew.",
            transformed="A small-town mayor announces a mysterious midnight curfew.",
        ),
        ReplacementExample(
            description="Keep unique cultural references when essential to irony.",
            original="Florida man holds city hall hostage with army of feral peacocks.",
            transformed="A Florida resident surrounds city hall with trained peacocks to force negotiations.",
        ),
    ),
)


class PromptManager:
    """Create prompts and maintain the on-disk archive."""

    def __init__(
        self,
        base_dir: str | Path,
        *,
        archive_dir: str | Path | None = None,
        prompt_filename: str | None = None,
        specification: PromptSpecification | None = None,
    ) -> None:
        self.base_dir = Path(base_dir)
        if archive_dir:
            self.archive_dir = Path(archive_dir)
        else:
            self.archive_dir = self.base_dir / "prompts"
        self.prompt_filename = prompt_filename or PROMPT_FILENAME
        self.spec = specification or DEFAULT_SPECIFICATION

    # ------------------------------------------------------------------
    def load_prompt(self) -> PromptBundle:
        self.base_dir.mkdir(parents=True, exist_ok=True)
        user_snippet = self._ensure_user_snippet()
        builder = PromptBuilder(self.spec, user_snippet)
        dynamic, formatting, combined = builder.build()
        prompt_hash = hashlib.sha256(combined.encode("utf-8")).hexdigest()[:12]
        self._archive_prompt(combined, prompt_hash)
        return PromptBundle(
            prompt=combined,
            prompt_hash=prompt_hash,
            dynamic_section=dynamic,
            formatting_section=formatting,
            specification=self.spec,
        )

    # ------------------------------------------------------------------
    def _ensure_user_snippet(self) -> str:
        path = self.base_dir / USER_SNIPPET_FILENAME
        if path.exists():
            text = path.read_text(encoding="utf-8").strip()
            if text:
                return text
        path.write_text(self.spec.default_user_prompt.strip() + "\n", encoding="utf-8")
        return self.spec.default_user_prompt.strip()

    # ------------------------------------------------------------------
    def _archive_prompt(self, text: str, prompt_hash: str) -> None:
        prompt_file = self.base_dir / self.prompt_filename
        archive_dir = self.archive_dir
        metadata_file = self.base_dir / PROMPT_METADATA_FILENAME

        prompt_file.write_text(text + "\n", encoding="utf-8")
        archive_dir.mkdir(parents=True, exist_ok=True)
        archive_path = archive_dir / f"{prompt_hash}.txt"
        if not archive_path.exists():
            archive_path.write_text(text, encoding="utf-8")

        metadata = {
            "prompt_hash": prompt_hash,
            "generated_at": datetime.utcnow().isoformat(timespec="seconds"),
            "prompt_path": str(prompt_file),
            "user_prompt_path": str(self.base_dir / USER_SNIPPET_FILENAME),
            "fields": [
                {
                    "name": field.name,
                    "type": "array" if field.is_list() else "string",
                    "description": field.description,
                    "constraints": list(field.constraints),
                }
                for field in self.spec.fields
            ],
            "formatting_rules": list(self.spec.formatting_rules),
            "quality_checks": list(self.spec.quality_checks),
        }
        metadata_file.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class PromptBuilder:
    def __init__(self, specification: PromptSpecification, user_prompt: str) -> None:
        self.spec = specification
        self.user_prompt = user_prompt.strip() or specification.default_user_prompt.strip()

    def build(self) -> tuple[str, str, str]:
        dynamic = self._build_dynamic()
        formatting = self._build_formatting()
        combined = "\n\n".join(part for part in (dynamic, formatting) if part)
        return dynamic, formatting, combined

    def _build_dynamic(self) -> str:
        spec = self.spec
        lines: list[str] = [self.user_prompt, "", "---", "", "### Transformation Goals", spec.objective]
        if spec.transformation_steps:
            lines.append("Follow these steps for each numbered headline:")
            for index, step in enumerate(spec.transformation_steps, start=1):
                lines.append(f"{index}. {step}")
        if spec.style_rules:
            lines.extend(["", "### Style Guardrails"])
            lines.extend(f"- {rule}" for rule in spec.style_rules)
        if spec.replacements:
            lines.extend(["", "### Replacement Examples"])
            for example in spec.replacements:
                lines.append(f"- {example.description}")
                lines.append(f"  - Input: {example.original}")
                lines.append(f"  - Output: {example.transformed}")
        if spec.fields:
            lines.extend(["", "### Field Reference"])
            for field in spec.fields:
                descriptor = "array" if field.is_list() else "string"
                lines.append(f"- `{field.name}` ({descriptor}) — {field.description}")
                for constraint in field.constraints:
                    lines.append(f"  - {constraint}")
        if spec.quality_checks:
            lines.extend(["", "### Quality Checklist"])
            lines.extend(f"- {item}" for item in spec.quality_checks)
        return "\n".join(line for line in lines if line is not None).strip()

    def _build_formatting(self) -> str:
        spec = self.spec
        lines: list[str] = ["### OUTPUT RULES"]
        lines.extend(f"- {rule}" for rule in spec.formatting_rules)
        lines.extend(
            [
                "",
                "### REQUIRED JSON SHAPE",
                "Return one JSON object per headline with keys in this order:",
            ]
        )
        for field in spec.fields:
            descriptor = "array of strings" if field.is_list() else "string"
            lines.append(f"- `{field.name}` ({descriptor})")
        lines.append("- Do not omit keys; use empty strings or arrays if necessary.")
        example_lines: list[str] = []
        example_object = {}
        for field in spec.fields:
            if field.is_list():
                value = list(field.example)
            else:
                value = field.example if isinstance(field.example, str) else ""
            example_object[field.name] = value
        example_lines.append(json.dumps([example_object], ensure_ascii=False, indent=2))
        lines.extend(["", "### EXAMPLE OUTPUT", *example_lines])
        return "\n".join(lines).strip()


__all__ = [
    "PromptField",
    "ReplacementExample",
    "PromptSpecification",
    "PromptBundle",
    "DEFAULT_SPECIFICATION",
    "PromptManager",
    "PromptBuilder",
]
