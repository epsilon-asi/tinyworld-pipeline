"""``gp run`` (plan 08 §2): options → :class:`RunOptions`, validate, plan or execute.

The runner owns locking, state and ``report.json``; this module only wires adapters, shows
progress and maps the report to console output / the JSON document.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, wait
from enum import StrEnum
from typing import Annotated, Any

import typer
from pydantic import ValidationError as PydanticValidationError

from grounded_pipeline import __version__
from grounded_pipeline.blender.locate import BlenderInfo
from grounded_pipeline.cli import render
from grounded_pipeline.cli.boundary import command
from grounded_pipeline.cli.commands.common import LibraryOption, ModelOption, state_for
from grounded_pipeline.cli.state import CliOutput, CliState, ErrorDocument
from grounded_pipeline.cli.wiring import (
    blender_check,
    build_graph,
    cli_overrides_from,
    load_library,
    production_context_factory,
    write_blender_cache,
)
from grounded_pipeline.config.layering import CliOverrides
from grounded_pipeline.domain.errors import (
    BudgetExceededError,
    CancelledError,
    PipelineError,
    UsageError,
)
from grounded_pipeline.domain.reports import RunReport
from grounded_pipeline.domain.state import StageName, stage_index
from grounded_pipeline.library.discovery import Library
from grounded_pipeline.library.validation import raise_if_errors, validate_library
from grounded_pipeline.pipeline.options import RunOptions
from grounded_pipeline.pipeline.planner import RunPlan
from grounded_pipeline.pipeline.report import build_environment, plan_rows, report_path
from grounded_pipeline.pipeline.runner import EXIT_SOME_FAILED, PipelineRunner
from grounded_pipeline.pipeline.state_store import JsonStateStore

COMMAND = "run"
POLL_INTERVAL_S = 0.5
E_MODELS_FAILED = "E_MODELS_FAILED"


class StageChoice(StrEnum):
    """``--from-stage`` / ``--to-stage`` / ``--force-stage`` values."""

    discover = "discover"
    prepare = "prepare"
    texture = "texture"
    verify = "verify"
    apply = "apply"
    export = "export"
    handoff = "handoff"


class RouteChoice(StrEnum):
    """``--route`` values (overrides ``[verify].route``)."""

    auto = "auto"
    extract = "extract"
    bake = "bake"


# --------------------------------------------------------------------------- options
def build_run_options(
    *,
    models: list[str] | None,
    tags: list[str] | None,
    from_stage: StageChoice,
    to_stage: StageChoice,
    force: bool,
    force_stages: list[StageChoice] | None,
    dry_run: bool,
    route: RouteChoice | None,
    concurrency: int | None,
    max_credits: float | None,
    keep_going: bool,
    resume_tasks: bool,
) -> RunOptions:
    """Validate flags into :class:`RunOptions`.

    Raises:
        UsageError: A model id does not match the id pattern or a value is out of range.
    """
    try:
        return RunOptions(
            models=models or None,
            tags=list(tags or []),
            from_stage=from_stage.value,
            to_stage=to_stage.value,
            force=force,
            force_stages=[s.value for s in force_stages or []],
            dry_run=dry_run,
            route_override=route.value if route else None,
            concurrency=concurrency,
            max_credits=max_credits,
            keep_going=keep_going,
            resume_tasks=resume_tasks,
        )
    except PydanticValidationError as exc:
        first = exc.errors()[0]
        where = ".".join(str(p) for p in first["loc"])
        raise UsageError(f"invalid run option {where}: {first['msg']}") from exc


def stage_in_range(options: RunOptions, stage: StageName) -> bool:
    """Whether ``stage`` lies inside ``from_stage..to_stage``."""
    return stage_index(options.from_stage) <= stage_index(stage) <= stage_index(options.to_stage)


# --------------------------------------------------------------------------- assembly
def _blender_for_run(state: CliState, library: Library) -> BlenderInfo:
    info = state.hooks.blender_resolver(state.blender, state.env, library.config.blender)
    write_blender_cache(library.paths, info)
    return info


def make_runner(
    state: CliState, library: Library, options: RunOptions, overrides: CliOverrides
) -> PipelineRunner:
    """Production (or injected) runner for this invocation.

    Raises:
        UsageError: ``TRIPO_API_KEY`` is required for the stage range but missing.
        BlenderNotFoundError / BlenderVersionError: Blender is needed but unusable.
    """
    graph = build_graph()
    if state.hooks.ctx_factory is not None:
        return PipelineRunner(state.hooks.ctx_factory, graph)
    needs_blender = not options.dry_run and stage_index(options.to_stage) >= stage_index("prepare")
    needs_key = not options.dry_run and stage_in_range(options, "texture")
    secrets = state.secrets()
    if needs_key and not secrets.has_tripo_key:
        raise UsageError(
            "TRIPO_API_KEY is required to run the texture stage",
            hint="set it in the environment or .env, or pass --to-stage prepare",
        )
    blender = _blender_for_run(state, library) if needs_blender else None
    environment = build_environment(
        pipeline_version=__version__,
        blender_path=blender.path if blender else None,
        blender_version=blender.version_string if blender else None,
    )
    factory = production_context_factory(
        state, secrets=secrets, blender=blender, overrides=overrides
    )
    return PipelineRunner(factory, graph, environment=environment)


def pre_validate(state: CliState, library: Library, options: RunOptions, cli: CliOverrides) -> None:
    """Plan 08 §2: unknown ids → exit 2, validation errors → exit 3, before any work."""
    model_ids = library.select(options.models, options.tags)
    check = (
        blender_check(state) if stage_index(options.to_stage) >= stage_index("prepare") else None
    )
    problems = validate_library(library, model_ids, env=state.env, cli=cli, blender_check=check)
    raise_if_errors(problems)


# --------------------------------------------------------------------------- execution
def execute_with_progress(
    state: CliState, runner: PipelineRunner, library: Library, options: RunOptions
) -> RunReport:
    """Run in a worker thread while the CLI thread polls state for the progress rows."""
    if not state.console.verbose_allowed:
        return runner.run(library, options)
    model_ids = library.select(options.models, options.tags)
    store = JsonStateStore(library.paths, pipeline_version=__version__)
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="gp-run") as pool:
        future: Future[RunReport] = pool.submit(runner.run, library, options)
        with render.RunProgress(state.console, model_ids) as progress:
            try:
                while not wait([future], timeout=POLL_INTERVAL_S).done:
                    progress.refresh({m: store.load(m) for m in model_ids})
            except KeyboardInterrupt:
                progress.note("cancelling… waiting for running stages to stop")
                if state.active_cancel is not None:
                    state.active_cancel.set()
                wait([future])
        return future.result()


def plan_json(plan: RunPlan) -> dict[str, Any]:
    """Dry-run rows and totals for the JSON document."""
    return {
        "rows": [
            dict(zip(("model", "stage", "action", "reason", "credits"), r, strict=True))
            for r in plan_rows(plan)
        ],
        "estimated_credits": plan.estimated_credits,
        "budget": plan.budget_limit,
        "stage_runs": plan.stage_runs,
    }


def report_json(report: RunReport, path: str) -> dict[str, Any]:
    """``run_id``, ``report_path``, ``models`` and ``credits_total`` (plan 08 §4)."""
    return {
        "run_id": report.run_id,
        "report_path": path,
        "credits_total": report.credits_total,
        "models": [
            {
                "model_id": m.model_id,
                "result": m.result,
                "credits": m.credits,
                "error_code": m.error_code,
                "error_message": m.error_message,
            }
            for m in report.models
        ],
    }


def run_error(report: RunReport) -> ErrorDocument | None:
    """Error document for a non-zero report exit code."""
    failed = [m for m in report.models if m.result == "failed"]
    if report.exit_code == 0:
        return None
    if report.exit_code == CancelledError.exit_code:
        return ErrorDocument(code=CancelledError.code, message="run cancelled")
    if report.exit_code == BudgetExceededError.exit_code:
        return ErrorDocument(code=BudgetExceededError.code, message="credit budget exceeded")
    if report.exit_code == EXIT_SOME_FAILED:
        return ErrorDocument(
            code=E_MODELS_FAILED, message=f"{len(failed)} model(s) failed", hint="see report.json"
        )
    return ErrorDocument(code=PipelineError.code, message=f"run exited with {report.exit_code}")


# --------------------------------------------------------------------------- command
@command(COMMAND)
def run_command(  # noqa: PLR0917 - every flag of plan 08 §2
    ctx: typer.Context,
    model: ModelOption = None,
    tag: Annotated[
        list[str] | None,
        typer.Option("--tag", help="Select models by tag (repeatable).", metavar="TAG"),
    ] = None,
    from_stage: Annotated[
        StageChoice, typer.Option("--from-stage", help="First stage to run.")
    ] = StageChoice.discover,
    to_stage: Annotated[
        StageChoice, typer.Option("--to-stage", help="Last stage to run.")
    ] = StageChoice.handoff,
    force: Annotated[
        bool, typer.Option("--force", help="Ignore fingerprints for every stage in range.")
    ] = False,
    force_stage: Annotated[
        list[StageChoice] | None,
        typer.Option(
            "--force-stage", help="Re-run this stage and everything downstream (repeatable)."
        ),
    ] = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Show the plan and credit estimate; do nothing.")
    ] = False,
    route: Annotated[
        RouteChoice | None, typer.Option("--route", help="Override verify.route for this run.")
    ] = None,
    concurrency: Annotated[
        int | None,
        typer.Option("--concurrency", help="Models processed in parallel.", min=1, max=16),
    ] = None,
    max_credits: Annotated[
        float | None, typer.Option("--max-credits", help="Credit budget for this run.", min=0)
    ] = None,
    keep_going: Annotated[
        bool, typer.Option("--keep-going/--fail-fast", help="Continue other models when one fails.")
    ] = True,
    resume_tasks: Annotated[
        bool,
        typer.Option(
            "--resume-tasks/--no-resume-tasks", help="Re-attach to in-flight Tripo tasks."
        ),
    ] = True,
    library_root: LibraryOption = None,
) -> CliOutput:
    """Run the pipeline (discover → … → handoff) over the selected models."""
    state = state_for(ctx, library_root)
    options = build_run_options(
        models=model,
        tags=tag,
        from_stage=from_stage,
        to_stage=to_stage,
        force=force,
        force_stages=force_stage,
        dry_run=dry_run,
        route=route,
        concurrency=concurrency,
        max_credits=max_credits,
        keep_going=keep_going,
        resume_tasks=resume_tasks,
    )
    overrides = cli_overrides_from(
        state,
        concurrency=concurrency,
        route=route.value if route else None,
        max_credits=max_credits,
    )
    library = load_library(state)
    pre_validate(state, library, options, overrides)
    runner = make_runner(state, library, options, overrides)
    if options.dry_run:
        plan = runner.plan(library, options)
        render.plan_table(state.console, plan)
        return CliOutput.success(COMMAND, dry_run=True, **plan_json(plan))
    report = execute_with_progress(state, runner, library, options)
    path = report_path(library.paths.runs_dir, report.run_id)
    render.run_summary(state.console, report, path)
    fields = report_json(report, str(path))
    error = run_error(report)
    if error is None:
        return CliOutput.success(COMMAND, **fields)
    return CliOutput.failure(COMMAND, error, exit_code=report.exit_code).model_copy(update=fields)


def register(app: typer.Typer) -> None:
    """Attach to the root app."""
    app.command(COMMAND)(run_command)


__all__ = [
    "COMMAND",
    "E_MODELS_FAILED",
    "RouteChoice",
    "StageChoice",
    "build_run_options",
    "execute_with_progress",
    "make_runner",
    "plan_json",
    "pre_validate",
    "register",
    "report_json",
    "run_command",
    "run_error",
    "stage_in_range",
]
