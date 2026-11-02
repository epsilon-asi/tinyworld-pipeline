"""Dry-run planner and the run/skip decision shared with the runner (plan 13 §1-§2, §9).

The decision for one stage is a pure function of its state record, its current input fingerprint,
the existence of its outputs and the force flags (:func:`decide_stage`). The planner applies it
down the stage range; the runner applies the very same function right before each stage so the
plan and the execution can never disagree.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

from grounded_pipeline.domain.errors import ConfigError, PipelineError, StagePrereqError
from grounded_pipeline.domain.state import (
    ModelState,
    StageName,
    StageRecord,
    StageStatus,
    stage_index,
)
from grounded_pipeline.pipeline.context import ModelContext, PipelineContext
from grounded_pipeline.pipeline.graph import StageGraph
from grounded_pipeline.pipeline.options import RunOptions
from grounded_pipeline.pipeline.stage import Stage
from grounded_pipeline.tripo.budget import estimate_credits as table_estimate

Action = Literal["run", "skip"]

REASON_UP_TO_DATE = "up to date"
REASON_UPSTREAM = "upstream invalidated"
TEXTURE_STAGE: StageName = "texture"
STYLE_IMAGE_MODES = frozenset({"text+style"})


@dataclass(frozen=True)
class PlannedStage:
    """What the runner would do with one stage of one model."""

    name: StageName
    action: Action
    reason: str
    estimated_credits: float = 0.0
    fingerprint: str | None = None


@dataclass(frozen=True)
class ModelPlan:
    """Planned stages (in range only) for one model."""

    model_id: str
    stages: tuple[PlannedStage, ...]

    @property
    def estimated_credits(self) -> float:
        return round(sum(s.estimated_credits for s in self.stages if s.action == "run"), 6)

    @property
    def runs(self) -> tuple[StageName, ...]:
        return tuple(s.name for s in self.stages if s.action == "run")


@dataclass(frozen=True)
class RunPlan:
    """Every selected model's plan plus the credit budget it will run under."""

    models: dict[str, ModelPlan] = field(default_factory=dict)
    budget_limit: float | None = None

    @property
    def estimated_credits(self) -> float:
        return round(sum(m.estimated_credits for m in self.models.values()), 6)

    @property
    def stage_runs(self) -> int:
        return sum(len(m.runs) for m in self.models.values())


@runtime_checkable
class CreditEstimator(Protocol):
    """Optional stage capability: a credit estimate for ``--dry-run`` (the texture stage)."""

    def estimate_credits(self, model: ModelContext, ctx: PipelineContext) -> float:
        """Estimated provider credits of one run of this stage."""
        ...


# --------------------------------------------------------------------------- pure decisions
def force_reason(name: StageName, options: RunOptions) -> str | None:
    """Why ``name`` is forced (``--force`` or a ``--force-stage`` at or upstream of it), or None."""
    if options.force:
        return "forced (--force)"
    for forced in options.force_stages:
        if stage_index(forced) <= stage_index(name):
            return f"forced (--force-stage {forced})"
    return None


def outputs_missing(folder: Path, record: StageRecord) -> list[str]:
    """Declared outputs of ``record`` that do not exist under the model ``folder``."""
    return [rel for rel in record.outputs if not (folder / rel).exists()]


def record_is_complete(folder: Path, record: StageRecord) -> bool:
    """``SUCCEEDED`` with every output present (the ``--from-stage`` prerequisite)."""
    return record.status == StageStatus.SUCCEEDED and not outputs_missing(folder, record)


def decide_stage(
    name: StageName,
    record: StageRecord,
    fingerprint: str | None,
    *,
    folder: Path,
    force: str | None,
    upstream_ran: bool,
    fingerprint_error: str | None = None,
) -> PlannedStage:
    """Apply the skip rule of plan 13 §2 to one stage.

    Args:
        name: Stage name.
        record: Its current state record.
        fingerprint: Its current input fingerprint (``None`` when it could not be computed).
        folder: Model folder (outputs are relative to it).
        force: Force reason from :func:`force_reason`, or ``None``.
        upstream_ran: An upstream stage (other than ``discover``) runs in this pass.
        fingerprint_error: Why the fingerprint could not be computed, if so.

    Returns:
        The decision with its human-readable reason.
    """
    reason = _run_reason(
        record,
        fingerprint,
        folder=folder,
        force=force,
        upstream_ran=upstream_ran,
        fingerprint_error=fingerprint_error,
    )
    if reason is None:
        return PlannedStage(name, "skip", REASON_UP_TO_DATE, fingerprint=fingerprint)
    return PlannedStage(name, "run", reason, fingerprint=fingerprint)


def _stale_status_reason(record: StageRecord) -> str:
    invalidated_by = record.extra.get("invalidated_by")
    if record.status == StageStatus.PENDING and invalidated_by:
        return f"invalidated by {invalidated_by}"
    if record.status == StageStatus.PENDING and record.attempts == 0:
        return "never run"
    return f"status {record.status.value}"


def _run_reason(
    record: StageRecord,
    fingerprint: str | None,
    *,
    folder: Path,
    force: str | None,
    upstream_ran: bool,
    fingerprint_error: str | None,
) -> str | None:
    """Why the stage must run, or ``None`` when it is up to date (checks in priority order)."""
    if upstream_ran or force:
        return REASON_UPSTREAM if upstream_ran else force
    if fingerprint is None:
        return f"inputs unavailable ({fingerprint_error or 'unknown'})"
    if record.status != StageStatus.SUCCEEDED:
        return _stale_status_reason(record)
    if record.fingerprint != fingerprint:
        return "fingerprint changed"
    missing = outputs_missing(folder, record)
    return f"outputs missing ({missing[0]})" if missing else None


def check_prerequisites(
    graph: StageGraph, state: ModelState, *, from_stage: StageName, folder: Path
) -> None:
    """Every graph stage before ``from_stage`` must be complete (plan 08 §2).

    Raises:
        StagePrereqError: A prerequisite stage is not ``succeeded`` with existing outputs.
    """
    for name in graph.upstream(from_stage):
        record = state.stages.get(name, StageRecord())
        if record_is_complete(folder, record):
            continue
        missing = outputs_missing(folder, record)
        why = (
            f"outputs missing ({missing[0]})"
            if record.status == StageStatus.SUCCEEDED
            else f"status {record.status.value}"
        )
        raise StagePrereqError(
            f"{state.model_id}: stage {name!r} must have succeeded before "
            f"--from-stage {from_stage} ({why})",
            detail={"model_id": state.model_id, "stage": name, "status": record.status.value},
            hint=f"run `gp run --model {state.model_id} --to-stage {name}` first",
        )


# --------------------------------------------------------------------------- estimates
def estimate_stage_credits(stage: Stage, model: ModelContext, ctx: PipelineContext) -> float:
    """Credit estimate for running ``stage`` once.

    A stage implementing :class:`CreditEstimator` is asked directly; otherwise the texture stage
    is priced from ``[tripo.credit_table]`` and every other stage costs nothing.
    """
    if isinstance(stage, CreditEstimator):
        return float(stage.estimate_credits(model, ctx))
    if stage.name != TEXTURE_STAGE:
        return 0.0
    cfg = model.config
    try:
        return table_estimate(
            cfg.texture.quality,
            style_image=cfg.texture.prompt_mode in STYLE_IMAGE_MODES,
            table=cfg.tripo.credit_table,
        )
    except ConfigError:
        return 0.0


# --------------------------------------------------------------------------- planning
def compute_fingerprint(
    stage: Stage, model: ModelContext, ctx: PipelineContext
) -> tuple[str | None, str | None]:
    """``(fingerprint, None)`` or ``(None, reason)`` when inputs are missing or unreadable."""
    try:
        return stage.input_fingerprint(model, ctx), None
    except (OSError, PipelineError) as exc:
        return None, f"{type(exc).__name__}: {exc}"


def plan_model(
    graph: StageGraph,
    model: ModelContext,
    ctx: PipelineContext,
    options: RunOptions,
    state: ModelState,
) -> ModelPlan:
    """Plan the stage range of ``options`` for one model without performing any work.

    Raises:
        StagePrereqError: Stages before ``options.from_stage`` are not complete.
        UsageError: The stage range is invalid for this graph.
    """
    folder = model.paths.folder
    in_range = graph.stages_between(options.from_stage, options.to_stage)  # validates the range
    check_prerequisites(graph, state, from_stage=options.from_stage, folder=folder)
    planned: list[PlannedStage] = []
    upstream_ran = False
    for stage in in_range:
        record = state.stages.get(stage.name, StageRecord())
        fingerprint, error = (
            (None, None) if upstream_ran else compute_fingerprint(stage, model, ctx)
        )
        decision = decide_stage(
            stage.name,
            record,
            fingerprint,
            folder=folder,
            force=force_reason(stage.name, options),
            upstream_ran=upstream_ran,
            fingerprint_error=error,
        )
        if decision.action == "run":
            decision = PlannedStage(
                decision.name,
                decision.action,
                decision.reason,
                estimated_credits=estimate_stage_credits(stage, model, ctx),
                fingerprint=decision.fingerprint,
            )
            upstream_ran = upstream_ran or stage.name != "discover"
        planned.append(decision)
    return ModelPlan(model_id=model.model_id, stages=tuple(planned))


def plan_models(
    graph: StageGraph,
    models: Sequence[ModelContext],
    ctx: PipelineContext,
    options: RunOptions,
    states: Mapping[str, ModelState],
    *,
    budget_limit: float | None,
) -> RunPlan:
    """Plan every model (id order preserved)."""
    plans = {m.model_id: plan_model(graph, m, ctx, options, states[m.model_id]) for m in models}
    return RunPlan(models=plans, budget_limit=budget_limit)


__all__ = [
    "REASON_UPSTREAM",
    "REASON_UP_TO_DATE",
    "Action",
    "CreditEstimator",
    "ModelPlan",
    "PlannedStage",
    "RunPlan",
    "check_prerequisites",
    "compute_fingerprint",
    "decide_stage",
    "estimate_stage_credits",
    "force_reason",
    "outputs_missing",
    "plan_model",
    "plan_models",
    "record_is_complete",
]
