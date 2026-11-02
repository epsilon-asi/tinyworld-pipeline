"""Run options (plan 13 §1). A pure value object shared by the CLI, runner and stages."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from grounded_pipeline.domain._base import FrozenModel
from grounded_pipeline.domain.ids import ModelId
from grounded_pipeline.domain.state import StageName


class RunOptions(FrozenModel):
    models: list[ModelId] | None = None
    tags: list[str] = Field(default_factory=list)
    from_stage: StageName = "discover"
    to_stage: StageName = "handoff"
    force: bool = False
    force_stages: list[StageName] = Field(default_factory=list)
    dry_run: bool = False
    route_override: Literal["auto", "extract", "bake"] | None = None
    concurrency: int | None = None
    max_credits: float | None = None
    keep_going: bool = True
    resume_tasks: bool = True
