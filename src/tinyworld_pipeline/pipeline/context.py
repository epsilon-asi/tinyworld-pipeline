"""``PipelineContext`` and ``ModelContext`` (plan 13 §5).

Stages receive both and never construct adapters themselves. ``cli/app.py`` builds the production
context; tests build fakes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tinyworld_pipeline.domain.ids import RunId
from tinyworld_pipeline.domain.library_config import LibraryConfig
from tinyworld_pipeline.domain.model_spec import ResolvedModel
from tinyworld_pipeline.domain.ports import CancelToken, Clock, Sleeper
from tinyworld_pipeline.domain.prompt import ComposedPrompt
from tinyworld_pipeline.library.paths import ModelPaths
from tinyworld_pipeline.pipeline.options import RunOptions
from tinyworld_pipeline.pipeline.state_store import StateStore

if TYPE_CHECKING:
    import structlog

    from tinyworld_pipeline.blender.runner import BlenderRunner
    from tinyworld_pipeline.library.discovery import Library
    from tinyworld_pipeline.texturing.provider import TextureProvider
    from tinyworld_pipeline.tripo.budget import CreditBudget


@dataclass(frozen=True)
class PipelineContext:
    """Everything a run shares across models: adapters, options, run id, cancellation, logger."""

    library: Library
    options: RunOptions
    run_id: RunId
    blender: BlenderRunner
    provider: TextureProvider
    state: StateStore
    budget: CreditBudget
    cancel: CancelToken
    clock: Clock
    sleeper: Sleeper
    log: structlog.stdlib.BoundLogger
    pipeline_version: str
    env: dict[str, str]
    """Process environment snapshot (``GP_*`` overrides are part of the discover fingerprint)."""
    cli_overrides: dict[str, Any]
    """``CliOverrides.model_dump(mode="json")`` for fingerprinting and reports."""


@dataclass(frozen=True)
class ModelContext:
    """One model's resolved spec, paths and bound logger."""

    resolved: ResolvedModel
    paths: ModelPaths
    log: structlog.stdlib.BoundLogger

    @property
    def model_id(self) -> str:
        return self.resolved.doc.model_id

    @property
    def config(self) -> LibraryConfig:
        return self.resolved.config

    @property
    def primary_fbx(self) -> Path:
        return self.resolved.primary_fbx

    @property
    def animation_fbx(self) -> list[Path]:
        return self.resolved.animation_fbx

    @property
    def composed_prompt(self) -> ComposedPrompt:
        return self.resolved.composed_prompt

    @property
    def material_name(self) -> str:
        return f"{self.model_id}_mat"

    def relative(self, path: Path) -> str:
        """Forward-slash path relative to the model folder (the form used in state/reports)."""
        return path.resolve().relative_to(self.paths.folder.resolve()).as_posix()
