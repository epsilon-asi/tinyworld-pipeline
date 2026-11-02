from __future__ import annotations

import json
from pathlib import Path

from tinyworld_pipeline.domain.reports import ModelRunSummary, RunReport
from tinyworld_pipeline.domain.state import StageStatus
from tinyworld_pipeline.pipeline.planner import ModelPlan, PlannedStage, RunPlan
from tinyworld_pipeline.pipeline.report import (
    PLAN_HEADER,
    append_console,
    build_environment,
    console_path,
    plan_rows,
    plan_total_line,
    render_plan,
    report_path,
    write_run_report,
)

RUN_ID = "01ARZ3NDEKTSV4RRFFQ69G5FAV"


def _plan() -> RunPlan:
    ant = ModelPlan(
        "ant_worker",
        (
            PlannedStage("discover", "run", "fingerprint changed"),
            PlannedStage("prepare", "skip", "up to date"),
            PlannedStage("texture", "run", "fingerprint changed", estimated_credits=20),
            PlannedStage("verify", "run", "upstream invalidated"),
            PlannedStage("apply", "run", "upstream invalidated"),
            PlannedStage("export", "run", "upstream invalidated"),
            PlannedStage("handoff", "run", "upstream invalidated"),
        ),
    )
    bee = ModelPlan("bee", (PlannedStage("discover", "skip", "up to date"),))
    return RunPlan(models={"ant_worker": ant, "bee": bee}, budget_limit=200)


def test_plan_rows_collapse_consecutive_identical_decisions() -> None:
    rows = plan_rows(_plan())
    assert rows == [
        ("ant_worker", "discover", "run", "fingerprint changed", ""),
        ("ant_worker", "prepare", "skip", "up to date", ""),
        ("ant_worker", "texture", "run", "fingerprint changed", "20"),
        ("ant_worker", "verify..handoff", "run", "upstream invalidated", ""),
        ("bee", "discover", "skip", "up to date", ""),
    ]
    assert plan_rows(_plan().models) == rows


def test_plan_totals_and_render() -> None:
    plan = _plan()
    assert plan.estimated_credits == 20
    assert plan.stage_runs == 6
    assert plan_total_line(plan) == "TOTAL estimated credits: 20   (budget: 200)"
    assert plan_total_line(RunPlan()).endswith("(budget: unlimited)")
    text = render_plan(plan)
    assert text.splitlines()[0].split() == [*PLAN_HEADER[:4], "est.", "credits"]
    assert "verify..handoff" in text
    assert text.endswith("(budget: 200)\n")


def test_write_run_report_and_console(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    report = RunReport(
        run_id=RUN_ID,
        started_at="2026-01-01T00:00:00.000Z",
        finished_at="2026-01-01T00:00:01.000Z",
        library=tmp_path,
        options={"force": False},
        models=[
            ModelRunSummary(
                model_id="ant",
                result="succeeded",
                stages={"discover": StageStatus.SUCCEEDED},
                credits=1.5,
                duration_s=0.5,
            )
        ],
        credits_total=1.5,
        exit_code=0,
        environment=build_environment(pipeline_version="0.1.0", blender_path=Path("/b")),
    )
    path = write_run_report(runs, report)
    assert path == report_path(runs, RUN_ID) == runs / RUN_ID / "report.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["run_id"] == RUN_ID
    assert data["models"][0]["stages"] == {"discover": "succeeded"}
    assert data["environment"]["blender_path"] == "/b"
    assert data["environment"]["pipeline_version"] == "0.1.0"
    assert data["environment"]["python"]
    assert data["environment"]["blender_version"] is None

    assert append_console(runs, RUN_ID, "hello") == console_path(runs, RUN_ID)
    append_console(runs, RUN_ID, "world\n")
    assert console_path(runs, RUN_ID).read_text() == "hello\nworld\n"
