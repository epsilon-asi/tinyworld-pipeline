"""Terminal rendering for the CLI (plan 08 §2, 13 §9, 14 §5).

Together with :mod:`tinyworld_pipeline.observability.console` this is the only module that writes
to the terminal. Everything goes through a :class:`Console` so ``--json`` / ``--no-color`` are
honoured in one place; the JSON document itself is written straight to stdout by
:func:`emit_json`.
"""

from __future__ import annotations

import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Self

from rich import box
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text

from tinyworld_pipeline.cli.state import CliOutput, ErrorDocument
from tinyworld_pipeline.domain.errors import Problem
from tinyworld_pipeline.domain.reports import RunReport
from tinyworld_pipeline.domain.state import STAGE_ORDER, ModelState, StageStatus
from tinyworld_pipeline.observability.console import Console
from tinyworld_pipeline.pipeline.planner import RunPlan
from tinyworld_pipeline.pipeline.report import PLAN_HEADER, plan_rows, plan_total_line

STATUS_GLYPH: dict[StageStatus, str] = {
    StageStatus.SUCCEEDED: "✓",
    StageStatus.SKIPPED: "✓",
    StageStatus.FAILED: "✗",
    StageStatus.PENDING: "○",
    StageStatus.RUNNING: "⟳",
}
VALIDATE_COLUMNS = ("model_id", "status", "problems")
STATUS_COLUMNS = ("model_id", *STAGE_ORDER, "credits", "last run", "last error")
STAGE_ABBREVIATIONS = tuple(name[:4] for name in STAGE_ORDER)
RUN_ID_TAIL = 8
RUN_COLUMNS = ("model_id", "result", "credits", "duration", "error")
DOCTOR_COLUMNS = ("check", "status", "detail")


# --------------------------------------------------------------------------- raw stdout
def emit_json(output: CliOutput) -> None:
    """Write the single ``--json`` document to stdout."""
    sys.stdout.write(output.to_json())
    sys.stdout.flush()


def emit_version(version: str) -> None:
    """``gp --version`` output."""
    sys.stdout.write(f"gp {version}\n")
    sys.stdout.flush()


# --------------------------------------------------------------------------- messages
def print_error_document(console: Console, error: ErrorDocument) -> None:
    """Render a *returned* error document like ``Console.print_error`` renders a raised error."""
    if console.silent:
        return
    line = Text("✗ ", style="bold red")
    line.append(error.code, style="bold red")
    line.append(": ")
    line.append(error.message)
    console.rich.print(line)
    if error.hint:
        console.rich.print(Text(f"  hint: {error.hint}", style="dim"))


def message(console: Console, text: str, *, style: str | None = None) -> None:
    """One informational line."""
    console.print(text, style=style)


def notice(console: Console, text: str) -> None:
    """One line that must survive ``--quiet`` (paths of produced files, confirmations)."""
    console.print(text, important=True)


# --------------------------------------------------------------------------- validate
def validation_status(problems: Sequence[Problem]) -> str:
    """``error`` > ``warning`` > ``ok``."""
    if any(p.severity == "error" for p in problems):
        return "error"
    if problems:
        return "warning"
    return "ok"


def group_problems(
    problems: Sequence[Problem], model_ids: Iterable[str]
) -> dict[str, list[Problem]]:
    """Problems keyed by model id (``""`` for library-level ones), every id present."""
    groups: dict[str, list[Problem]] = {"": []}
    for model_id in model_ids:
        groups[model_id] = []
    for problem in problems:
        groups.setdefault(problem.model_id or "", []).append(problem)
    return groups


def validate_table(console: Console, problems: Sequence[Problem], model_ids: Sequence[str]) -> None:
    """``model_id | status | problems`` plus the detailed problem list (plan 08 §2)."""
    groups = group_problems(problems, model_ids)
    rows = []
    for model_id, items in groups.items():
        if not model_id and not items:
            continue
        codes = ", ".join(dict.fromkeys(p.code for p in items)) or "-"
        rows.append((model_id or "(library)", validation_status(items), codes))
    console.table("Validation", VALIDATE_COLUMNS, rows)
    console.print_problems(problems)


# --------------------------------------------------------------------------- status
def last_error(state: ModelState) -> str:
    """``CODE: message`` of the most recent failed stage, or ``-``."""
    failed = [
        rec
        for rec in state.stages.values()
        if rec.status == StageStatus.FAILED and rec.error_code is not None
    ]
    if not failed:
        return "-"
    rec = max(failed, key=lambda r: r.finished_at or "")
    return f"{rec.error_code}: {rec.error_message or ''}".rstrip(": ")


def short_run_id(run_id: str | None) -> str:
    """``…`` + the last 8 characters of a ULID (the full id lives in the JSON output)."""
    if not run_id:
        return "-"
    return "…" + run_id[-RUN_ID_TAIL:]


def status_row(state: ModelState) -> tuple[str, ...]:
    """One :func:`status_table` row."""
    glyphs = tuple(
        STATUS_GLYPH[state.stages[name].status] if name in state.stages else "○"
        for name in STAGE_ORDER
    )
    return (
        state.model_id,
        *glyphs,
        f"{state.credits_total:g}",
        short_run_id(state.last_run_id),
        last_error(state),
    )


def status_table(console: Console, states: Sequence[ModelState]) -> None:
    """``model_id | discover … handoff | credits | last run | last error`` (compact box)."""
    if not console.verbose_allowed:
        return
    table = Table(title="Status", box=box.SIMPLE_HEAD, pad_edge=False)
    table.add_column("model_id", no_wrap=True)
    for short in STAGE_ABBREVIATIONS:
        table.add_column(short, justify="center", min_width=4, no_wrap=True)
    table.add_column("credits", justify="right", min_width=7)
    table.add_column("last run", min_width=9, no_wrap=True)
    table.add_column("last error", overflow="fold")
    for state in states:
        table.add_row(*status_row(state))
    console.rich.print(table)
    console.print(
        "stages: "
        + " ".join(f"{s}={n}" for s, n in zip(STAGE_ABBREVIATIONS, STAGE_ORDER, strict=True)),
        style="dim",
    )
    console.print("legend: ✓ succeeded/up to date  ✗ failed  ○ pending  ⟳ running", style="dim")


# --------------------------------------------------------------------------- dry run
def plan_table(console: Console, plan: RunPlan) -> None:
    """Plan 13 §9 table followed by the total line."""
    console.table("Dry run", PLAN_HEADER, plan_rows(plan))
    console.print(plan_total_line(plan), style="bold")


# --------------------------------------------------------------------------- run
def run_summary(console: Console, report: RunReport, report_path: Path) -> None:
    """Per-model result table, grouped warnings and the report path."""
    rows = [
        (
            m.model_id,
            m.result,
            f"{m.credits:g}",
            f"{m.duration_s:.1f}s",
            f"{m.error_code}: {m.error_message}" if m.error_code else "-",
        )
        for m in report.models
    ]
    console.table(f"Run {report.run_id}", RUN_COLUMNS, rows)
    warnings_block(console, report.problems)
    notice(console, f"report: {report_path}")


def warnings_block(console: Console, problems: Sequence[Problem]) -> None:
    """End-of-run “Warnings” block grouped by model (plan 14 §5)."""
    grouped: dict[str, list[str]] = {}
    for problem in problems:
        if problem.severity != "warning":
            continue
        grouped.setdefault(problem.model_id or "(library)", []).append(problem.render())
    console.print_warnings(grouped)


class RunProgress:
    """One rich progress row per model, refreshed from polled state (plan 14 §5).

    Use as a context manager; call :meth:`refresh` with the latest ``ModelState`` per model.
    Rendering is skipped entirely when the console is not in verbose mode.
    """

    def __init__(self, console: Console, model_ids: Sequence[str]) -> None:
        self._console = console
        self._model_ids = list(model_ids)
        self._enabled = console.verbose_allowed
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("{task.fields[model_id]}", style="bold"),
            TextColumn("{task.fields[stage]}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=console.rich,
            transient=True,
            disable=not self._enabled,
        )
        self._tasks = {
            model_id: self._progress.add_task(
                "", total=len(STAGE_ORDER), model_id=model_id, stage="waiting"
            )
            for model_id in self._model_ids
        }

    def __enter__(self) -> Self:
        self._progress.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._progress.stop()

    def refresh(self, states: Mapping[str, ModelState]) -> None:
        """Update every row from the given states (missing ids are left untouched)."""
        for model_id, task_id in self._tasks.items():
            state = states.get(model_id)
            if state is None:
                continue
            done = sum(
                1
                for rec in state.stages.values()
                if rec.status in (StageStatus.SUCCEEDED, StageStatus.SKIPPED)
            )
            self._progress.update(task_id, completed=done, stage=current_stage(state))

    def note(self, text: str) -> None:
        """Print a line above the bars (e.g. “cancelling…”)."""
        if self._enabled:
            self._progress.console.print(Text(text, style="yellow"))


def current_stage(state: ModelState) -> str:
    """Label for the progress row: the running stage, the failed one, or ``done``/``waiting``."""
    for name in STAGE_ORDER:
        rec = state.stages.get(name)
        if rec is not None and rec.status == StageStatus.RUNNING:
            return name
    for name in reversed(STAGE_ORDER):
        rec = state.stages.get(name)
        if rec is not None and rec.status == StageStatus.FAILED:
            return f"{name} ✗"
    if all(
        state.stages.get(n) is not None
        and state.stages[n].status in (StageStatus.SUCCEEDED, StageStatus.SKIPPED)
        for n in STAGE_ORDER
    ):
        return "done"
    return "waiting"


# --------------------------------------------------------------------------- doctor
@dataclass(frozen=True)
class Check:
    """One ``gp doctor`` line."""

    name: str
    ok: bool
    detail: str
    skipped: bool = False

    @property
    def status(self) -> str:
        if self.skipped:
            return "skip"
        return "pass" if self.ok else "FAIL"


def doctor_checklist(console: Console, checks: Sequence[Check]) -> None:
    """``check | status | detail`` table plus a one-line verdict."""
    console.table("gp doctor", DOCTOR_COLUMNS, [(c.name, c.status, c.detail) for c in checks])
    failed = [c for c in checks if not c.ok and not c.skipped]
    if failed:
        console.print(f"{len(failed)} check(s) failed", style="bold red", important=True)
    else:
        console.print("all checks passed", style="bold green")


__all__ = [
    "DOCTOR_COLUMNS",
    "RUN_COLUMNS",
    "STAGE_ABBREVIATIONS",
    "STATUS_COLUMNS",
    "STATUS_GLYPH",
    "VALIDATE_COLUMNS",
    "Check",
    "RunProgress",
    "current_stage",
    "doctor_checklist",
    "emit_json",
    "emit_version",
    "group_problems",
    "last_error",
    "message",
    "notice",
    "plan_table",
    "print_error_document",
    "run_summary",
    "short_run_id",
    "status_row",
    "status_table",
    "validate_table",
    "validation_status",
    "warnings_block",
]
