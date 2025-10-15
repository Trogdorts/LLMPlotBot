
"""
Task dataclass for queue processing.
"""

from dataclasses import dataclass


@dataclass
class Task:
    """Represents one title × model × prompt_hash job."""

    id: str
    title: str
    model: str
    prompt_hash: str
    prompt_dynamic: str
    prompt_formatting: str
    retry_count: int = 0

    @property
    def prompt_text(self) -> str:
        """Return the combined prompt text for legacy consumers."""

        sections = [self.prompt_dynamic.strip(), self.prompt_formatting.strip()]
        return "\n\n".join(section for section in sections if section)
