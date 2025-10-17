
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
