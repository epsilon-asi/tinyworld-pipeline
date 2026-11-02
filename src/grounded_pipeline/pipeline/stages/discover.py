"""Stage ``discover`` (plan 09 §1) — the state baseline.

Steps 1-6 of plan 09 §1 (parse, layer, resolve paths, compose prompt, validate) live in
``library/resolve.py``. The runner calls :func:`resolve_model` itself *before* any stage runs,
because it needs the :class:`ResolvedModel` to build the :class:`ModelContext` every stage
receives. This stage therefore performs step 7 and the bookkeeping that must be attributable to
a stage record:

* re-resolve the model and surface its warnings in the stage outcome;
* when the stored ``pipeline_version`` differs in its *major* component from the running one,
  mark every downstream stage ``pending`` with ``extra.invalidated_by = "pipeline_version"`` and
  stamp the new version into the state.

It declares no file outputs; the state file itself is the product.
"""

from __future__ import annotations

from grounded_pipeline.config.layering import CliOverrides
from grounded_pipeline.domain.state import (
    ModelState,
    StageName,
    StageOutcome,
    StageStatus,
    downstream_of,
)
from grounded_pipeline.library.resolve import discover_fingerprint, resolve_model
from grounded_pipeline.pipeline.context import ModelContext, PipelineContext

INVALIDATED_BY_VERSION = "pipeline_version"


def major_version(version: str) -> str:
    """The leading numeric component of a version string (``"1.4.2" → "1"``)."""
    head = version.strip().split(".", 1)[0]
    digits = "".join(ch for ch in head if ch.isdigit())
    return digits or head


def _cli(ctx: PipelineContext) -> CliOverrides:
    return CliOverrides.model_validate(ctx.cli_overrides)


class DiscoverStage:
    """Baseline stage: fingerprint = ``model.md`` + ``library.toml`` + ``style.md`` + overrides +
    ``pipeline_version`` (plan 09 §1)."""

    name: StageName = "discover"

    def input_fingerprint(self, model: ModelContext, ctx: PipelineContext) -> str:
        """Delegate to :func:`discover_fingerprint` (content hashes, no paths)."""
        return discover_fingerprint(
            ctx.library,
            model.model_id,
            env=ctx.env,
            cli=_cli(ctx),
            pipeline_version=ctx.pipeline_version,
        )

    def run(self, model: ModelContext, ctx: PipelineContext) -> StageOutcome:
        """Re-resolve (collecting warnings) and reconcile the state's ``pipeline_version``."""
        _, warnings = resolve_model(ctx.library, model.model_id, env=ctx.env, cli=_cli(ctx))
        state = ctx.state.load(model.model_id)
        invalidated = self._reconcile_version(state, ctx)
        extra = {
            "pipeline_version": ctx.pipeline_version,
            "warnings": len(warnings),
            "prompt_sha256": model.composed_prompt.sha256,
        }
        if invalidated:
            extra["invalidated_downstream"] = list(invalidated)
        return StageOutcome(
            status="succeeded",
            outputs=[],
            extra=extra,
            warnings=[p.render() for p in warnings],
        )

    def _reconcile_version(self, state: ModelState, ctx: PipelineContext) -> tuple[StageName, ...]:
        if major_version(state.pipeline_version) == major_version(ctx.pipeline_version):
            if state.pipeline_version != ctx.pipeline_version:
                self._stamp_version(ctx, state.model_id)
            return ()
        downstream = downstream_of(self.name)
        for name in downstream:
            ctx.state.update(
                state.model_id,
                name,
                status=StageStatus.PENDING,
                extra={"invalidated_by": INVALIDATED_BY_VERSION},
            )
        self._stamp_version(ctx, state.model_id)
        ctx.log.warning(
            "state_invalidated_by_pipeline_version",
            model_id=state.model_id,
            previous=state.pipeline_version,
            current=ctx.pipeline_version,
        )
        return downstream

    @staticmethod
    def _stamp_version(ctx: PipelineContext, model_id: str) -> None:
        fresh = ctx.state.load(model_id)
        fresh.pipeline_version = ctx.pipeline_version
        ctx.state.save(fresh)


__all__ = ["INVALIDATED_BY_VERSION", "DiscoverStage", "major_version"]
