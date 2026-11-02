"""Composed prompt value object (plan 06 §4, 10 §6)."""

from __future__ import annotations

from grounded_pipeline.domain._base import FrozenModel
from grounded_pipeline.domain.ids import Sha256

PROMPT_MAX_CHARS = 1_000
PROMPT_WARN_CHARS = 600


class ComposedPrompt(FrozenModel):
    text: str  # final prompt sent to provider
    sources: dict[str, str]  # placeholder → section used (for provenance)
    length: int
    sha256: Sha256
