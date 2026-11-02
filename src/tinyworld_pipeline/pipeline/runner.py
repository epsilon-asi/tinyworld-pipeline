"""``PipelineRunner`` — the plan 13 §1 algorithm at run level.

The runner owns everything that is *per run*: the library run lock, the ULID run id, model
selection and validation, the credit budget, the dry-run plan, the worker pool (one model per
worker; :mod:`model_runner` walks the stages sequentially), cancellation, the exit code and
``report.json``. Stages never see any of this; they get a :class:`PipelineContext` and a
:class:`ModelContext`.

Exit codes (plan 08 §2) are computed here and only *mapped* by the CLI: ``0`` all succeeded or
skipped, ``4`` some models failed, ``5`` a model hit ``E_BUDGET``, ``130`` cancelled. ``2``/``3``
(usage, validation, prerequisites, busy library) are raised as their :class:`PipelineError` after
``report.json`` has been written.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from ulid import ULID

from tinyworld_pipeline.config.layering import CliOverrides
from tinyworld_pipeline.domain.errors import (
    BudgetExceededError,
    CancelledError,
    PipelineError,
    Problem,
    ValidationError,
)
from tinyworld_pipeline.domain.reports import ModelRunSummary, RunEnvironment, RunReport
from tinyworld_pipeline.domain.state import StageStatus
from tinyworld_pipeline.library.discovery import Library
from tinyworld_pipeline.library.resolve import resolve_model
from tinyworld_pipeline.library.validation import raise_if_errors, validate_library
from tinyworld_pipeline.pipeline.context import ModelContext, PipelineContext
from tinyworld_pipeline.pipeline.graph import StageGraph
from tinyworld_pipeline.pipeline.locks import run_lock
from tinyworld_pipeline.pipeline.model_runner import E_UNEXPECTED, ModelRunner, status_snapshot
from tinyworld_pipeline.pipeline.options import RunOptions
from tinyworld_pipeline.pipeline.planner import (
    ModelPlan,
    PlannedStage,
    RunPlan,
    plan_model,
    plan_models,
)
from tinyworld_pipeline.pipeline.report import build_environment, write_run_report
from tinyworld_pipeline.pipeline.timing import utc_now_iso
from tinyworld_pipeline.tripo.budget import CreditBudget

EXIT_SOME_FAILED = 4


@dataclass(frozen=True)
class RunSetup:
    """What the runner decides before the context factory builds the :class:`PipelineContext`."""

    library: Library
    options: RunOptions
    run_id: str
    budget: CreditBudget


ContextFactory = Callable[[RunSetup], PipelineContext]


def context_from_template(template: PipelineContext) -> ContextFactory:
    """A factory that copies ``template`` and swaps in the per-run fields (plus a bound run id)."""

    def _factory(setup: RunSetup) -> PipelineContext:
        return dataclasses.replace(
            template,
            library=setup.library,
            options=setup.options,
            run_id=setup.run_id,
            budget=setup.budget,
            log=template.log.bind(run_id=setup.run_id),
        )

    return _factory


class PipelineRunner:
    """Run the stage graph over a library (plan 13 §1).

    Args:
        ctx: Either a :class:`PipelineContext` template whose per-run fields are replaced each run,
            or a factory receiving :class:`RunSetup` and returning the full context (the CLI passes
            a factory so adapters are built once per run).
        graph: The stage graph.
        environment: Optional environment block for ``report.json`` (Blender path/version); a
            minimal one is built from the context otherwise.
    """

    def __init__(
        self,
        ctx: PipelineContext | ContextFactory,
        graph: StageGraph,
        *,
        environment: RunEnvironment | None = None,
    ) -> None:
        self._factory: ContextFactory = (
            context_from_template(ctx) if isinstance(ctx, PipelineContext) else ctx
        )
        self.graph = graph
        self._models = ModelRunner(graph)
        self._environment = environment
        self.last_plan: RunPlan | None = None
        self.last_context: PipelineContext | None = None
        self._interrupted = False

    # ----------------------------------------------------------------- public API
    def run(self, library: Library, options: RunOptions) -> RunReport:
        """Execute (or, with ``options.dry_run``, only plan) and return the run report.

        Raises:
            LibraryBusyError: Another run holds ``library/.pipeline/run.lock``.
            UsageError: Unknown model id or invalid stage range.
            ValidationError: The library or a selected model fails validation.
            StagePrereqError: Stages before ``--from-stage`` are not complete.
        """
        with run_lock(library.paths.run_lock, timeout_s=0):
            return self._run_locked(library, options)

    def plan(self, library: Library, options: RunOptions) -> RunPlan:
        """Plan without locking, executing or writing a report."""
        ctx = self._context(library, options)
        models, _ = self._prepare(ctx)
        return self._plan(ctx, models)

    def plan_stages(self, model: ModelContext, options: RunOptions) -> list[PlannedStage]:
        """Plan one already-resolved model against the last built context (CLI/test helper).

        Raises:
            RuntimeError: No context has been built yet (call :meth:`run` or :meth:`plan` first).
        """
        if self.last_context is None:
            msg = "plan_stages() needs a context; call run()/plan() first"
            raise RuntimeError(msg)
        ctx = dataclasses.replace(self.last_context, options=options)
        state = ctx.state.load(model.model_id)
        return list(plan_model(self.graph, model, ctx, options, state).stages)

    # ----------------------------------------------------------------- setup
    def _context(self, library: Library, options: RunOptions) -> PipelineContext:
        limit = options.max_credits
        if limit is None:
            limit = library.config.tripo.max_credits_per_run or None
        setup = RunSetup(library, options, str(ULID()), CreditBudget(limit))
        ctx = self._factory(setup)
        self.last_context = ctx
        self._interrupted = False
        self._models.reset()
        return ctx

    def _prepare(self, ctx: PipelineContext) -> tuple[list[ModelContext], list[Problem]]:
        """Select, validate (errors raise) and resolve the models; returns warnings too."""
        library, options = ctx.library, ctx.options
        model_ids = library.select(options.models, options.tags)
        cli = CliOverrides.model_validate(ctx.cli_overrides)
        problems = validate_library(library, model_ids, env=ctx.env, cli=cli)
        raise_if_errors(problems)
        models: list[ModelContext] = []
        for model_id in model_ids:
            resolved, _ = resolve_model(library, model_id, env=ctx.env, cli=cli)
            models.append(
                ModelContext(
                    resolved=resolved,
                    paths=library.paths.model(model_id),
                    log=ctx.log.bind(model_id=model_id),
                )
            )
        return models, [p for p in problems if p.severity == "warning"]

    def _plan(self, ctx: PipelineContext, models: list[ModelContext]) -> RunPlan:
        states = {m.model_id: ctx.state.load(m.model_id) for m in models}
        plan = plan_models(
            self.graph, models, ctx, ctx.options, states, budget_limit=ctx.budget.limit
        )
        self.last_plan = plan
        return plan

    # ----------------------------------------------------------------- run
    def _run_locked(self, library: Library, options: RunOptions) -> RunReport:
        started_at = utc_now_iso()
        ctx = self._context(library, options)
        ctx.log.info(
            "run_started", models=options.models, tags=options.tags, dry_run=options.dry_run
        )
        try:
            models, warnings = self._prepare(ctx)
            plan = self._plan(ctx, models)
        except PipelineError as exc:
            problems = exc.problems if isinstance(exc, ValidationError) else []
            self._write_report(ctx, started_at, [], exit_code=exc.exit_code, problems=problems)
            raise
        if options.dry_run:
            summaries = [self._dry_run_summary(ctx, m, plan.models[m.model_id]) for m in models]
            return self._write_report(ctx, started_at, summaries, exit_code=0, problems=warnings)
        summaries = self._execute(ctx, models)
        exit_code = self._exit_code(ctx, summaries)
        ctx.log.info("run_finished", exit_code=exit_code, credits_total=ctx.budget.spent)
        return self._write_report(
            ctx, started_at, summaries, exit_code=exit_code, problems=warnings
        )

    def _execute(self, ctx: PipelineContext, models: list[ModelContext]) -> list[ModelRunSummary]:
        workers = max(1, ctx.options.concurrency or ctx.library.config.library.concurrency)
        results: dict[str, ModelRunSummary] = {}
        futures: dict[Future[ModelRunSummary], str] = {}
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="gp-model") as pool:
            try:
                for model in models:
                    futures[pool.submit(self._models.run_model, ctx, model)] = model.model_id
                for future in as_completed(futures):
                    results[futures[future]] = future.result()
            except KeyboardInterrupt:
                self._interrupted = True
                ctx.cancel.set()
                ctx.log.warning("run_interrupted")
                pool.shutdown(wait=True, cancel_futures=True)
        for future, model_id in futures.items():
            if model_id in results:
                continue
            finished = future.done() and not future.cancelled() and future.exception() is None
            results[model_id] = (
                future.result() if finished else self._models.cancelled_summary(ctx, model_id)
            )
        return [results[m.model_id] for m in models]

    # ----------------------------------------------------------------- summaries & report
    @staticmethod
    def _dry_run_summary(
        ctx: PipelineContext, model: ModelContext, plan: ModelPlan
    ) -> ModelRunSummary:
        stages = status_snapshot(ctx.state.load(model.model_id))
        for planned in plan.stages:
            stages[planned.name] = (
                StageStatus.PENDING if planned.action == "run" else StageStatus.SKIPPED
            )
        return ModelRunSummary(
            model_id=model.model_id,
            result="skipped",
            stages=stages,
            credits=plan.estimated_credits,
            duration_s=0.0,
        )

    def _exit_code(self, ctx: PipelineContext, summaries: list[ModelRunSummary]) -> int:
        if self._interrupted or (ctx.cancel.is_set() and not self._models.fail_fast_cancelled):
            return CancelledError.exit_code
        if any(s.error_code == BudgetExceededError.code for s in summaries):
            return BudgetExceededError.exit_code
        if any(s.result == "failed" for s in summaries):
            return EXIT_SOME_FAILED
        return 0

    def _write_report(
        self,
        ctx: PipelineContext,
        started_at: str,
        summaries: list[ModelRunSummary],
        *,
        exit_code: int,
        problems: list[Problem],
    ) -> RunReport:
        report = RunReport(
            run_id=ctx.run_id,
            started_at=started_at,
            finished_at=utc_now_iso(),
            library=ctx.library.paths.root,
            options=ctx.options.model_dump(mode="json"),
            models=summaries,
            credits_total=round(ctx.budget.spent, 6),
            exit_code=exit_code,
            environment=self._environment
            or build_environment(pipeline_version=ctx.pipeline_version),
            problems=problems,
        )
        write_run_report(ctx.library.paths.runs_dir, report)
        return report


__all__ = [
    "EXIT_SOME_FAILED",
    "E_UNEXPECTED",
    "ContextFactory",
    "PipelineRunner",
    "RunSetup",
    "context_from_template",
]
