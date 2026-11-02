"""Run report assembly, on-disk layout and dry-run table rows (plan 13 §9, 14 §6).

Layout under ``library/.pipeline/runs/<run_id>/``: ``report.json`` (the :class:`RunReport`) and
``console.txt`` (whatever the CLI chooses to tee there via :func:`append_console`).
"""

from __future__ import annotations

import platform
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from grounded_pipeline.domain.reports import RunEnvironment, RunReport
from grounded_pipeline.pipeline.atomic import atomic_write_json
from grounded_pipeline.pipeline.planner import ModelPlan, PlannedStage, RunPlan

REPORT_FILENAME = "report.json"
CONSOLE_FILENAME = "console.txt"

PlanRow = tuple[str, str, str, str, str]
"""``(model, stage or "a..b", action, reason, estimated credits or "")``."""

PLAN_HEADER: PlanRow = ("model", "stage", "action", "reason", "est. credits")


# --------------------------------------------------------------------------- environment
def build_environment(
    *,
    pipeline_version: str,
    blender_path: Path | str | None = None,
    blender_version: str | None = None,
) -> RunEnvironment:
    """Describe the process environment for ``report.json`` (plan 14 §6)."""
    return RunEnvironment(
        python=platform.python_version(),
        pipeline_version=pipeline_version,
        os=f"{platform.system()} {platform.release()} ({sys.platform})",
        blender_version=blender_version,
        blender_path=str(blender_path) if blender_path is not None else None,
    )


# --------------------------------------------------------------------------- files
def run_dir(runs_dir: Path, run_id: str) -> Path:
    """``runs_dir/<run_id>``."""
    return runs_dir / run_id


def report_path(runs_dir: Path, run_id: str) -> Path:
    """Where ``report.json`` of ``run_id`` lives."""
    return run_dir(runs_dir, run_id) / REPORT_FILENAME


def console_path(runs_dir: Path, run_id: str) -> Path:
    """Where ``console.txt`` of ``run_id`` lives."""
    return run_dir(runs_dir, run_id) / CONSOLE_FILENAME


def write_run_report(runs_dir: Path, report: RunReport) -> Path:
    """Write ``report.json`` atomically and return its path."""
    path = report_path(runs_dir, report.run_id)
    atomic_write_json(path, report)
    return path


def append_console(runs_dir: Path, run_id: str, text: str) -> Path:
    """Append ``text`` to ``console.txt`` (created on first use) and return its path."""
    path = console_path(runs_dir, run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(text)
        if not text.endswith("\n"):
            handle.write("\n")
    return path


# --------------------------------------------------------------------------- dry-run table
def _credits_cell(value: float) -> str:
    return f"{value:g}" if value else ""


def _collapse(stages: Sequence[PlannedStage]) -> list[tuple[str, PlannedStage, float]]:
    """Merge consecutive stages sharing action+reason into ``"first..last"`` groups."""
    groups: list[tuple[list[PlannedStage], float]] = []
    for stage in stages:
        if groups and (groups[-1][0][-1].action, groups[-1][0][-1].reason) == (
            stage.action,
            stage.reason,
        ):
            groups[-1][0].append(stage)
            groups[-1] = (groups[-1][0], groups[-1][1] + stage.estimated_credits)
        else:
            groups.append(([stage], stage.estimated_credits))
    out: list[tuple[str, PlannedStage, float]] = []
    for members, total in groups:
        label = members[0].name if len(members) == 1 else f"{members[0].name}..{members[-1].name}"
        out.append((label, members[0], total))
    return out


def plan_rows(plan: RunPlan | Mapping[str, ModelPlan]) -> list[PlanRow]:
    """Table rows for ``gp run --dry-run`` (plan 13 §9), without the header or total line."""
    models = plan.models if isinstance(plan, RunPlan) else plan
    rows: list[PlanRow] = []
    for model_id, model_plan in models.items():
        for label, first, total in _collapse(model_plan.stages):
            rows.append((model_id, label, first.action, first.reason, _credits_cell(total)))
    return rows


def plan_total_line(plan: RunPlan) -> str:
    """``TOTAL estimated credits: N   (budget: M|unlimited)``."""
    budget = "unlimited" if plan.budget_limit is None else f"{plan.budget_limit:g}"
    return f"TOTAL estimated credits: {plan.estimated_credits:g}   (budget: {budget})"


def render_plan(plan: RunPlan) -> str:
    """Plain-text dry-run table (the CLI may render it with rich instead)."""
    rows = [PLAN_HEADER, *plan_rows(plan)]
    widths = [max(len(r[i]) for r in rows) for i in range(len(PLAN_HEADER))]
    lines = [
        "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)).rstrip() for row in rows
    ]
    lines.append(plan_total_line(plan))
    return "\n".join(lines) + "\n"


__all__ = [
    "CONSOLE_FILENAME",
    "PLAN_HEADER",
    "REPORT_FILENAME",
    "PlanRow",
    "append_console",
    "build_environment",
    "console_path",
    "plan_rows",
    "plan_total_line",
    "render_plan",
    "report_path",
    "run_dir",
    "write_run_report",
]
