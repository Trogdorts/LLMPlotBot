"""Structured prompt specification and builder utilities."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import json
from typing import Iterable, Sequence, Tuple


@dataclass(frozen=True)
class PromptField:
    """Definition of a single JSON field expected from the model."""

    name: str
    kind: str
    description: str
    constraints: Tuple[str, ...]
    example_pattern: Sequence[str] | str

    def is_list(self) -> bool:
        return self.kind.lower() in {"array", "list"}

    def example(self, index: int) -> str | list[str]:
        """Return an example value formatted for ``index``."""

        if self.is_list():
            patterns: Sequence[str]
            if isinstance(self.example_pattern, str):
                patterns = (self.example_pattern,)
            else:
                patterns = tuple(self.example_pattern)
            values: list[str] = []
            for item_index, pattern in enumerate(patterns):
                marker = chr(ord("a") + item_index)
                formatted = pattern.format(n=index, i=item_index + 1, letter=marker)
                values.append(formatted)
            return values

        pattern = (
            self.example_pattern if isinstance(self.example_pattern, str) else ""
        )
        return pattern.format(n=index, i=1, letter="a")


@dataclass(frozen=True)
class ReplacementExample:
    """Example illustrating how to neutralise or adjust headline details."""

    description: str
    original: str
    transformed: str


@dataclass(frozen=True)
class PromptSpecification:
    """Comprehensive configuration used to assemble an LLM prompt."""

    default_user_prompt: str
    objective: str
    transformation_steps: Tuple[str, ...]
    style_rules: Tuple[str, ...]
    quality_checks: Tuple[str, ...]
    formatting_rules: Tuple[str, ...]
    fields: Tuple[PromptField, ...]
    replacements: Tuple[ReplacementExample, ...]

    @property
    def required_field_names(self) -> Tuple[str, ...]:
        return tuple(field.name for field in self.fields)

    @property
    def list_field_names(self) -> Tuple[str, ...]:
        return tuple(field.name for field in self.fields if field.is_list())

    def field_reference(self) -> str:
        """Return a formatted description of the JSON fields."""

        lines: list[str] = []
        for field in self.fields:
            header = f"- `{field.name}` ({'array' if field.is_list() else 'string'}) — {field.description}"
            lines.append(header)
            for constraint in field.constraints:
                lines.append(f"  - {constraint}")
        return "\n".join(lines)

    def build_example_object(self, index: int) -> OrderedDict[str, object]:
        """Return an ordered mapping representing an example response."""

        example = OrderedDict()
        for field in self.fields:
            example[field.name] = field.example(index)
        return example

    def build_example_output(self, count: int) -> str:
        """Return a JSON array showing the expected schema for ``count`` items."""

        total = max(1, int(count or 1))
        entries = [self.build_example_object(idx + 1) for idx in range(total)]
        return json.dumps(entries, ensure_ascii=False, indent=2)


class PromptBuilder:
    """Create final prompt sections from a :class:`PromptSpecification`."""

    def __init__(self, specification: PromptSpecification, *, user_prompt: str | None = None) -> None:
        self.specification = specification
        cleaned = (user_prompt or "").strip()
        if not cleaned:
            cleaned = specification.default_user_prompt.strip()
        self.user_prompt = cleaned

    def build_dynamic_section(self) -> str:
        spec = self.specification
        sections: list[str] = [self.user_prompt, "", "---", "", "### Transformation Goals", spec.objective]

        if spec.transformation_steps:
            sections.append("Follow these steps for each numbered headline:")
            for step_index, step in enumerate(spec.transformation_steps, start=1):
                sections.append(f"{step_index}. {step}")

        if spec.style_rules:
            sections.extend(["", "### Style Guardrails"])
            for rule in spec.style_rules:
                sections.append(f"- {rule}")

        if spec.replacements:
            sections.extend(["", "### Replacement Examples"])
            for example in spec.replacements:
                sections.append(f"- {example.description}")
                sections.append(f"  - Input: {example.original}")
                sections.append(f"  - Output: {example.transformed}")

        sections.extend(["", "### Field Reference", spec.field_reference()])

        if spec.quality_checks:
            sections.extend(["", "### Quality Checklist"])
            for item in spec.quality_checks:
                sections.append(f"- {item}")

        return "\n".join(line for line in sections if line).strip()

    def build_formatting_section(self) -> str:
        spec = self.specification
        lines: list[str] = ["### OUTPUT RULES"]
        for rule in spec.formatting_rules:
            lines.append(f"- {rule}")

        lines.extend(
            [
                "",
                "### REQUIRED JSON SHAPE",
                "Return one JSON object per headline using these keys in order:",
            ]
        )

        for field in spec.fields:
            descriptor = "array of strings" if field.is_list() else "string"
            lines.append(f"- `{field.name}` ({descriptor})")

        lines.append("- Do not drop keys even if you have no data; use empty strings or arrays.")

        return "\n".join(lines).strip()

    def build(self) -> Tuple[str, str, str]:
        dynamic = self.build_dynamic_section()
        formatting = self.build_formatting_section()
        combined = "\n\n".join(part for part in (dynamic, formatting) if part)
        return dynamic, formatting, combined


DEFAULT_PROMPT_SPECIFICATION = PromptSpecification(
    default_user_prompt=(
        "You are a story-idea abstraction engine that turns absurd real-world news "
        "headlines into structured story seeds. For every headline you receive you "
        "will rewrite it as a single, natural-sounding sentence under 50 words that "
        "keeps its irony and realism. Do not invent or remove facts. Keep the "
        "original irony, tone, and absurdity. Replace names, organizations or places "
        "with neutral roles or archetypes only if not essential."
    ),
    objective=(
        "Convert each numbered headline into a compact story seed ready for "
        "creative development."
    ),
    transformation_steps=(
        "Rewrite the headline into one complete sentence that preserves factual "
        "details and irony.",
        "Normalise or anonymise specific names only when they are not core to the "
        "event.",
        "Extract conceptual metadata that would help a writer continue the story.",
    ),
    style_rules=(
        "Keep sentences factual yet vivid—no sensational embellishment.",
        "Avoid direct quotes, hashtags, or list fragments.",
        "Prefer concrete language over vague adjectives.",
    ),
    quality_checks=(
        "Every sentence stays below 50 words and reads naturally.",
        "No facts are added, removed, or contradicted.",
        "Array fields contain short, lowercase or Title Case fragments with no trailing punctuation.",
        "The JSON parses without manual fixes.",
    ),
    formatting_rules=(
        "Return only a JSON array with one object per headline in the same order as received.",
        "Do not include markdown fences, commentary, or explanations before or after the array.",
        "Preserve valid JSON syntax: double quotes, commas, and brackets only where allowed.",
        "If unsure about a value, fall back to an empty string or empty array rather than omitting the key.",
    ),
    fields=(
        PromptField(
            name="core_event",
            kind="string",
            description="One-sentence rewrite capturing the literal situation.",
            constraints=(
                "Maximum 50 words.",
                "Present tense where possible.",
                "Maintain irony or absurd tone without exaggerating it.",
            ),
            example_pattern=(
                "A ceremonial statue is quietly moved into a security guard's yard after the town forgets the anniversary {n}."
            ),
        ),
        PromptField(
            name="themes",
            kind="array",
            description="Conceptual threads a writer could explore.",
            constraints=(
                "Provide 2-5 concise nouns or noun phrases.",
                "Avoid duplicates or near-duplicates.",
            ),
            example_pattern=(
                "tradition clash {n}",
                "petty power plays {n}",
                "community memory {n}",
            ),
        ),
        PromptField(
            name="tone",
            kind="string",
            description="Primary stylistic feel of the situation.",
            constraints=("One to three words.", "Use lowercase unless a proper noun."),
            example_pattern="deadpan irony {n}",
        ),
        PromptField(
            name="conflict_type",
            kind="string",
            description="Short description of the core tension or friction.",
            constraints=("Keep it to 1-3 words.",),
            example_pattern="authority vs community {n}",
        ),
        PromptField(
            name="stakes",
            kind="string",
            description="What changes or is at risk because of the event.",
            constraints=("One concise sentence.", "State tangible consequences."),
            example_pattern="The town risks losing trust in its caretakers if the memorial stays missing {n}.",
        ),
        PromptField(
            name="setting_hint",
            kind="string",
            description="Quick cue about location or milieu without naming real places unless essential.",
            constraints=("3-6 words.",),
            example_pattern="small town municipal office {n}",
        ),
        PromptField(
            name="characters",
            kind="array",
            description="Archetypal roles involved in or affected by the event.",
            constraints=("List 2-5 roles.", "Use role nouns like 'disillusioned clerk'."),
            example_pattern=(
                "overlooked caretaker {n}",
                "nostalgic veteran {n}",
                "image-conscious mayor {n}",
            ),
        ),
        PromptField(
            name="potential_story_hooks",
            kind="array",
            description="Short prompts for how a longer story might continue.",
            constraints=("Provide 1-3 hooks.", "Each hook should be 6-12 words."),
            example_pattern=(
                "An audit uncovers why the memorial vanished {n}",
                "Caretakers stage a midnight rescue of civic pride {n}",
            ),
        ),
    ),
    replacements=(
        ReplacementExample(
            description="Neutralise non-essential proper nouns into roles.",
            original="Mayor Thompson orders a raccoon parade to save Main Street businesses.",
            transformed="A town mayor orders a raccoon parade to save downtown shops.",
        ),
        ReplacementExample(
            description="Replace exact places with descriptive archetypes unless the place is the punchline.",
            original="A Florida retirement community elects a mannequin as honorary sheriff.",
            transformed="A coastal retirement community elects a mannequin as honorary sheriff.",
        ),
        ReplacementExample(
            description="Keep essential brand names if they create the irony; otherwise generalise.",
            original="Tesla recalls cars because steering wheels detach mid-drive.",
            transformed="A luxury EV maker recalls cars after steering wheels detach mid-drive.",
        ),
    ),
)

__all__ = [
    "PromptField",
    "ReplacementExample",
    "PromptSpecification",
    "PromptBuilder",
    "DEFAULT_PROMPT_SPECIFICATION",
]
