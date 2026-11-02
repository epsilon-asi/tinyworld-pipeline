"""Run report models (plan 06 §11, 14 §6)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from grounded_pipeline.domain._base import FrozenModel
from grounded_pipeline.domain.errors import Problem
from grounded_pipeline.domain.ids import ModelId, RunId
from grounded_pipeline.domain.state import StageName, StageStatus

ModelResult = Literal["succeeded", "failed", "skipped", "cancelled"]


class ModelRunSummary(FrozenModel):
    model_id: ModelId
    result: ModelResult
    stages: dict[StageName, StageStatus]
    credits: float
    duration_s: float
    error_code: str | None = None
    error_message: str | None = None


class RunEnvironment(FrozenModel):
    python: str
    pipeline_version: str
    os: str
    blender_version: str | None = None
    blender_path: str | None = None


class RunReport(FrozenModel):
    run_id: RunId
    started_at: str
    finished_at: str
    library: Path
    options: dict[str, Any]
    models: list[ModelRunSummary]
    credits_total: float
    exit_code: int
    environment: RunEnvironment | None = None
    problems: list[Problem] = Field(default_factory=list)
