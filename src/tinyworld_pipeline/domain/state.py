"""Per-model pipeline state and stage outcomes (plan 06 §9)."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import Field

from tinyworld_pipeline.domain._base import FrozenModel, MutableModel
from tinyworld_pipeline.domain.ids import ModelId, RunId, Sha256

StageName = Literal["discover", "prepare", "texture", "verify", "apply", "export", "handoff"]
STAGE_ORDER: tuple[StageName, ...] = (
    "discover",
    "prepare",
    "texture",
    "verify",
    "apply",
    "export",
    "handoff",
)


def stage_index(name: StageName) -> int:
    """Position of ``name`` in :data:`STAGE_ORDER`."""
    return STAGE_ORDER.index(name)


def stages_between(start: StageName, end: StageName) -> tuple[StageName, ...]:
    """Inclusive slice of :data:`STAGE_ORDER` from ``start`` to ``end``."""
    i, j = stage_index(start), stage_index(end)
    if i > j:
        msg = f"from-stage {start!r} is after to-stage {end!r}"
        raise ValueError(msg)
    return STAGE_ORDER[i : j + 1]


def downstream_of(name: StageName) -> tuple[StageName, ...]:
    """Every stage strictly after ``name``."""
    return STAGE_ORDER[stage_index(name) + 1 :]


class StageStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"


class StageRecord(FrozenModel):
    status: StageStatus = StageStatus.PENDING
    fingerprint: Sha256 | None = None
    outputs: list[str] = Field(default_factory=list)  # relative paths that must exist
    started_at: str | None = None
    finished_at: str | None = None
    duration_s: float | None = None
    run_id: RunId | None = None
    error_code: str | None = None
    error_message: str | None = None
    attempts: int = 0
    extra: dict[str, Any] = Field(default_factory=dict)


def empty_stage_records() -> dict[StageName, StageRecord]:
    """A fresh ``pending`` record for every stage in order."""
    return {name: StageRecord() for name in STAGE_ORDER}


class ModelState(MutableModel):
    """Contents of ``<model>/.pipeline/state.json``."""

    schema_version: Literal[1] = 1
    model_id: ModelId
    pipeline_version: str
    stages: dict[StageName, StageRecord] = Field(default_factory=empty_stage_records)
    last_run_id: RunId | None = None
    credits_total: float = 0.0


class StageOutcome(FrozenModel):
    """Returned by ``Stage.run``."""

    status: Literal["succeeded", "skipped"]
    outputs: list[str]
    extra: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
