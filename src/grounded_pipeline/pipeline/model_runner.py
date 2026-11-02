"""Per-model execution: the ``run_model`` loop of plan 13 §1.

One :class:`ModelRunner` is shared by every worker thread of a run; it holds only the graph and
a fail-fast flag. Everything per model lives in a :class:`ModelRun` accumulator that becomes the
model's :class:`ModelRunSummary`.

State transitions per stage: ``RUNNING`` (fingerprint, run id, ``attempts + 1``, ``started_at``)
→ ``SUCCEEDED`` (outputs, merged ``extra``, ``finished_at``, ``duration_s``) or ``FAILED``
(``error_code``/``error_message``; a non-:class:`PipelineError` is recorded as ``E_UNEXPECTED``
with its traceback in the run log). When a stage other than ``discover`` runs, every downstream
stage is reset to ``PENDING`` with ``extra.invalidated_by = <stage>`` (plan 13 §2; ``discover``
is exempt per plan 09 §1 so a comment edit in ``model.md`` never triggers Blender work).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

from grounded_pipeline.domain.errors import CancelledError, PipelineError
from grounded_pipeline.domain.reports import ModelResult, ModelRunSummary
from grounded_pipeline.domain.state import (
    STAGE_ORDER,
    ModelState,
    StageName,
    StageRecord,
    StageStatus,
)
from grounded_pipeline.pipeline.context import ModelContext, PipelineContext
from grounded_pipeline.pipeline.graph import StageGraph
from grounded_pipeline.pipeline.planner import decide_stage, force_reason
from grounded_pipeline.pipeline.stage import Stage
from grounded_pipeline.pipeline.timing import Stopwatch, utc_now_iso

E_UNEXPECTED = "E_UNEXPECTED"
DISCOVER: StageName = "discover"
TEXTURE: StageName = "texture"
CREDITS_KEY = "credits"
INVALIDATED_BY = "invalidated_by"


@dataclass
class ModelRun:
    """Mutable per-model bookkeeping for one run."""

    model_id: str
    stages: dict[StageName, StageStatus]
    result: ModelResult = "skipped"
    error_code: str | None = None
    error_message: str | None = None
    credits: float = 0.0
    ran: list[StageName] = field(default_factory=list)

    def mark_cancelled(self, message: str) -> None:
        self.result, self.error_code, self.error_message = "cancelled", CancelledError.code, message

    def summary(self, duration_s: float) -> ModelRunSummary:
        return ModelRunSummary(
            model_id=self.model_id,
            result=self.result,
            stages=dict(self.stages),
            credits=round(self.credits, 6),
            duration_s=duration_s,
            error_code=self.error_code,
            error_message=self.error_message,
        )


def error_fields(exc: BaseException) -> tuple[str, str]:
    """``(code, message)`` of a failure: typed errors keep their code, others ``E_UNEXPECTED``."""
    if isinstance(exc, PipelineError):
        return str(exc.to_dict()["code"]), exc.message
    return E_UNEXPECTED, f"{type(exc).__name__}: {exc}"


def status_snapshot(state: ModelState) -> dict[StageName, StageStatus]:
    """Stage → status for every canonical stage present in ``state``."""
    return {name: state.stages[name].status for name in STAGE_ORDER if name in state.stages}


class ModelRunner:
    """Runs one model through the stage range sequentially (plan 13 §1 ``run_model``).

    Args:
        graph: The stage graph.
    """

    def __init__(self, graph: StageGraph) -> None:
        self.graph = graph
        self._lock = threading.Lock()
        self._fail_fast_cancel = False

    @property
    def fail_fast_cancelled(self) -> bool:
        """Whether this runner set the cancel token because of ``--fail-fast``."""
        with self._lock:
            return self._fail_fast_cancel

    def reset(self) -> None:
        """Clear the fail-fast flag (start of a run)."""
        with self._lock:
            self._fail_fast_cancel = False

    # ----------------------------------------------------------------- model level
    def run_model(self, ctx: PipelineContext, model: ModelContext) -> ModelRunSummary:
        """Execute every stage in range for ``model``; never raises for stage failures."""
        watch = Stopwatch(ctx.clock)
        run = ModelRun(model.model_id, status_snapshot(ctx.state.load(model.model_id)))
        try:
            self._run_stages(ctx, model, run)
        except Exception as exc:  # model boundary: a worker never dies silently (plan 14 §1)
            model.log.exception("model_unexpected_error")
            run.result, (run.error_code, run.error_message) = "failed", error_fields(exc)
        if run.result == "skipped" and run.ran:
            run.result = "succeeded"
        self._finalize_model(ctx, run)
        return run.summary(watch.stop())

    @staticmethod
    def cancelled_summary(ctx: PipelineContext, model_id: str) -> ModelRunSummary:
        """Summary for a model whose worker never started."""
        run = ModelRun(model_id, status_snapshot(ctx.state.load(model_id)))
        run.mark_cancelled("cancelled before the model started")
        return run.summary(0.0)

    # ----------------------------------------------------------------- stage loop
    def _run_stages(self, ctx: PipelineContext, model: ModelContext, run: ModelRun) -> None:
        options = ctx.options
        upstream_ran = False
        for stage in self.graph.stages_between(options.from_stage, options.to_stage):
            if ctx.cancel.is_set():
                run.mark_cancelled(f"cancelled before stage {stage.name}")
                return
            record = ctx.state.load(model.model_id).stages[stage.name]
            try:
                fingerprint = stage.input_fingerprint(model, ctx)
            except Exception as exc:  # stage boundary (plan 19 §3)
                self._fail_stage(ctx, model, run, stage, exc=exc, attempts_before=record.attempts)
                return
            decision = decide_stage(
                stage.name,
                record,
                fingerprint,
                folder=model.paths.folder,
                force=force_reason(stage.name, options),
                upstream_ran=upstream_ran,
            )
            if decision.action == "skip":
                run.stages[stage.name] = StageStatus.SKIPPED
                model.log.info("stage_skipped", stage=stage.name, reason=decision.reason)
                continue
            if not self._run_stage(ctx, model, run, stage, record, fingerprint, decision.reason):
                return
            upstream_ran = upstream_ran or stage.name != DISCOVER

    def _run_stage(  # noqa: PLR0917 - internal; positional mirrors the call site
        self,
        ctx: PipelineContext,
        model: ModelContext,
        run: ModelRun,
        stage: Stage,
        record: StageRecord,
        fingerprint: str,
        reason: str,
    ) -> bool:
        """RUNNING → SUCCEEDED/FAILED with timestamps; returns False when the model must stop."""
        model_id, name = model.model_id, stage.name
        log = model.log.bind(stage=name)
        watch = Stopwatch(ctx.clock)
        ctx.state.update(
            model_id,
            name,
            status=StageStatus.RUNNING,
            fingerprint=fingerprint,
            run_id=ctx.run_id,
            started_at=utc_now_iso(),
            finished_at=None,
            duration_s=None,
            error_code=None,
            error_message=None,
            attempts=record.attempts + 1,
        )
        run.stages[name] = StageStatus.RUNNING
        log.info("stage_started", reason=reason, attempt=record.attempts + 1)
        if name != DISCOVER:
            self._invalidate_downstream(ctx, model_id, name)
        try:
            outcome = stage.run(model, ctx)
        except Exception as exc:  # stage boundary (plan 19 §3): typed → code, other → E_UNEXPECTED
            self._fail_stage(ctx, model, run, stage, exc=exc, watch=watch)
            return False
        extra = {k: v for k, v in record.extra.items() if k != INVALIDATED_BY} | outcome.extra
        ctx.state.update(
            model_id,
            name,
            status=StageStatus.SUCCEEDED,
            outputs=list(outcome.outputs),
            finished_at=utc_now_iso(),
            duration_s=watch.stop(),
            extra_replace=extra,
        )
        run.stages[name] = StageStatus.SUCCEEDED
        run.ran.append(name)
        if name == TEXTURE:
            run.credits += float(outcome.extra.get(CREDITS_KEY, 0.0) or 0.0)
        for warning in outcome.warnings:
            log.warning("stage_warning", warning=warning)
        log.info("stage_succeeded", duration_s=watch.elapsed_s, outputs=len(outcome.outputs))
        return True

    def _fail_stage(
        self,
        ctx: PipelineContext,
        model: ModelContext,
        run: ModelRun,
        stage: Stage,
        *,
        exc: BaseException,
        attempts_before: int | None = None,
        watch: Stopwatch | None = None,
    ) -> None:
        code, message = error_fields(exc)
        log = model.log.bind(stage=stage.name, error_code=code)
        if code == E_UNEXPECTED:
            log.exception("stage_unexpected_error", error=message)
        else:
            log.warning("stage_failed", error=message)
        changes: dict[str, object] = {}
        if attempts_before is not None:  # failed before RUNNING was recorded (fingerprinting)
            changes = {
                "started_at": utc_now_iso(),
                "run_id": ctx.run_id,
                "attempts": attempts_before + 1,
                "fingerprint": None,
            }
        ctx.state.update(
            model.model_id,
            stage.name,
            status=StageStatus.FAILED,
            finished_at=utc_now_iso(),
            duration_s=watch.stop() if watch is not None else 0.0,
            error_code=code,
            error_message=message,
            **changes,
        )
        run.stages[stage.name] = StageStatus.FAILED
        run.error_code, run.error_message = code, message
        run.result = "cancelled" if code == CancelledError.code else "failed"
        if run.result == "failed" and not ctx.options.keep_going and not ctx.cancel.is_set():
            with self._lock:
                self._fail_fast_cancel = True
            ctx.cancel.set()
            log.warning("fail_fast_cancelling_remaining_models")

    def _invalidate_downstream(self, ctx: PipelineContext, model_id: str, stage: StageName) -> None:
        for name in self.graph.downstream(stage):
            ctx.state.update(
                model_id, name, status=StageStatus.PENDING, extra={INVALIDATED_BY: stage}
            )

    @staticmethod
    def _finalize_model(ctx: PipelineContext, run: ModelRun) -> None:
        state = ctx.state.load(run.model_id)
        state.last_run_id = ctx.run_id
        state.credits_total = round(state.credits_total + run.credits, 6)
        ctx.state.save(state)


__all__ = [
    "CREDITS_KEY",
    "E_UNEXPECTED",
    "INVALIDATED_BY",
    "ModelRun",
    "ModelRunner",
    "error_fields",
    "status_snapshot",
]
