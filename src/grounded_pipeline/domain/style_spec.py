"""Style source specification: ``library/style/style.md`` (plan 05 §4, 06 §4)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field

from grounded_pipeline.domain._base import FrozenModel

BASE_PROMPT_SECTION = "base_prompt"


class StyleFrontMatter(FrozenModel):
    schema_version: Literal[1] = 1
    name: str
    primary_reference: Path = Path("references/primary.png")
    negative_terms: list[str] = Field(default_factory=list)


class StyleDocument(FrozenModel):
    """Parsed ``style.md``: front matter plus ``## heading`` sections (lower-cased keys)."""

    folder: Path
    front: StyleFrontMatter
    sections: dict[str, str]  # "base_prompt" REQUIRED (enforced by library validation)
