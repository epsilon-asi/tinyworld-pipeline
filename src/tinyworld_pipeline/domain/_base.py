"""Base classes shared by every domain model.

All contracts are immutable pydantic models that reject unknown keys, so a typo in ``model.md`` or a
drifted JSON document fails loudly at the boundary instead of being silently ignored.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class FrozenModel(BaseModel):
    """Immutable, ``extra="forbid"`` pydantic model. Every contract inherits from this."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class MutableModel(BaseModel):
    """``extra="forbid"`` pydantic model that may be mutated (used for pipeline state only)."""

    model_config = ConfigDict(extra="forbid", frozen=False, validate_assignment=True)
