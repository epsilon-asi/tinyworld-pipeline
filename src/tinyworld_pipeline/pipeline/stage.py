"""The ``Stage`` port (plan 03 §4). Stages are composed linearly in v1 (``pipeline/graph.py``)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from tinyworld_pipeline.domain.state import StageName, StageOutcome

if TYPE_CHECKING:
    from tinyworld_pipeline.pipeline.context import ModelContext, PipelineContext


class Stage(Protocol):
    """One idempotent pipeline step with declared inputs."""

    name: StageName

    def input_fingerprint(self, model: ModelContext, ctx: PipelineContext) -> str:
        """SHA-256 hex over everything that, if changed, must re-run this stage."""
        ...

    def run(self, model: ModelContext, ctx: PipelineContext) -> StageOutcome:
        """Perform the work. MUST be idempotent given the same fingerprint.

        MUST write outputs only under ``model.paths.work`` / ``output`` / ``pipeline``.
        Raise ``PipelineError`` subclasses for failures; never return partial outputs.
        """
        ...
