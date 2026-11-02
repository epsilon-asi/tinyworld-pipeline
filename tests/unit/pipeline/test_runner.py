"""Runner tests (plan 13 §10) with scripted :class:`FakeStage` graphs."""

from __future__ import annotations

import json
import re
import threading
from pathlib import Path
from typing import Any

import pytest

from tinyworld_pipeline.config.layering import CliOverrides
from tinyworld_pipeline.domain.errors import (
    BudgetExceededError,
    CancelledError,
    LibraryBusyError,
    StagePrereqError,
    TexturePackError,
    UsageError,
    ValidationError,
)
from tinyworld_pipeline.domain.ports import CancelToken
from tinyworld_pipeline.domain.reports import RunReport
from tinyworld_pipeline.domain.state import STAGE_ORDER, StageName, StageOutcome, StageStatus
from tinyworld_pipeline.library.discovery import Library
from tinyworld_pipeline.library.resolve import discover_fingerprint
from tinyworld_pipeline.pipeline.context import ModelContext, PipelineContext
from tinyworld_pipeline.pipeline.graph import StageGraph
from tinyworld_pipeline.pipeline.locks import run_lock
from tinyworld_pipeline.pipeline.model_runner import E_UNEXPECTED
from tinyworld_pipeline.pipeline.options import RunOptions
from tinyworld_pipeline.pipeline.planner import PlannedStage
from tinyworld_pipeline.pipeline.report import plan_rows
from tinyworld_pipeline.pipeline.runner import PipelineRunner, RunSetup, context_from_template
from tinyworld_pipeline.pipeline.stages.discover import DiscoverStage
from tinyworld_pipeline.pipeline.state_store import InMemoryStateStore, JsonStateStore, StateStore

from .conftest import (
    MODEL_IDS,
    FakeStage,
    default_run,
    fake_graph,
    make_context,
    model_context,
)

ULID_RE = re.compile(r"^[0-9A-HJKMNP-TV-Z]{26}$")
LATER: tuple[StageName, ...] = STAGE_ORDER[1:]
FROM_TEXTURE: tuple[StageName, ...] = STAGE_ORDER[2:]
PEEKED: tuple[StageName, ...] = ("verify", "handoff")


class Harness:
    """A runner over the fixture library with fake stages and shared state."""

    def __init__(
        self,
        library: Library,
        *,
        state: StateStore | None = None,
        cancel: CancelToken | None = None,
        **stages: FakeStage,
    ) -> None:
        self.library = library
        self.state: StateStore = state or InMemoryStateStore("0.1.0")
        self.cancel = cancel or CancelToken()
        self.graph, self.stages = fake_graph(**stages)
        self.template = make_context(library, state=self.state, cancel=self.cancel)
        self.runner = PipelineRunner(self.template, self.graph)

    def run(self, **options: Any) -> RunReport:
        return self.runner.run(self.library, RunOptions(**options))

    def counts(self) -> dict[StageName, int]:
        return {name: stage.run_count for name, stage in self.stages.items()}

    def statuses(self, model_id: str) -> dict[StageName, StageStatus]:
        state = self.state.load(model_id)
        return {n: state.stages[n].status for n in STAGE_ORDER}


def all_counts(n: int) -> dict[StageName, int]:
    return dict.fromkeys(STAGE_ORDER, n)


@pytest.fixture
def harness(library: Library) -> Harness:
    return Harness(library)


# --------------------------------------------------------------------------- happy path
def test_first_run_executes_everything_and_second_run_nothing(harness: Harness) -> None:
    report = harness.run()
    assert report.exit_code == 0
    assert harness.counts() == all_counts(2)
    assert [m.model_id for m in report.models] == list(MODEL_IDS)
    for summary in report.models:
        assert summary.result == "succeeded"
        assert all(s == StageStatus.SUCCEEDED for s in summary.stages.values())
        assert summary.error_code is None
    for model_id in MODEL_IDS:
        state = harness.state.load(model_id)
        assert state.last_run_id == report.run_id
        rec = state.stages["prepare"]
        assert rec.status == StageStatus.SUCCEEDED
        assert rec.attempts == 1
        assert rec.outputs == ["work/prepare.txt"]
        assert rec.run_id == report.run_id
        assert rec.started_at
        assert rec.finished_at
        assert rec.duration_s is not None
        assert rec.fingerprint
        assert "invalidated_by" not in rec.extra

    second = harness.run()
    assert second.exit_code == 0
    assert harness.counts() == all_counts(2)
    assert second.run_id != report.run_id
    for summary in second.models:
        assert summary.result == "skipped"
        assert all(s == StageStatus.SKIPPED for s in summary.stages.values())
    assert harness.statuses("cube_rigged")["texture"] == StageStatus.SUCCEEDED  # state untouched


def test_fingerprint_change_reruns_stage_and_downstream(harness: Harness) -> None:
    harness.run()
    harness.stages["texture"].version = "v2"
    harness.run(models=["cube_rigged"])
    counts = harness.counts()
    assert counts["discover"] == counts["prepare"] == 2
    assert all(counts[n] == 3 for n in FROM_TEXTURE)
    assert harness.state.load("cube_rigged").stages["texture"].attempts == 2


def test_discover_change_does_not_invalidate_downstream(harness: Harness) -> None:
    harness.run(models=["cube_palette"])
    harness.stages["discover"].version = "v2"
    report = harness.run(models=["cube_palette"])
    assert harness.counts()["discover"] == 2
    assert all(harness.counts()[n] == 1 for n in LATER)
    summary = report.models[0]
    assert summary.result == "succeeded"
    assert summary.stages["prepare"] == StageStatus.SKIPPED


def test_missing_output_reruns_stage(harness: Harness, fixture_root: Path) -> None:
    harness.run(models=["cube_palette"])
    (fixture_root / "models" / "cube_palette" / "work" / "verify.txt").unlink()
    harness.run(models=["cube_palette"])
    counts = harness.counts()
    assert counts["texture"] == 1
    assert counts["verify"] == counts["apply"] == counts["handoff"] == 2


def test_downstream_marked_pending_when_stage_runs(library: Library) -> None:
    seen: dict[str, StageStatus] = {}

    def peek(model: ModelContext, ctx: PipelineContext) -> StageOutcome:
        seen.update({n: ctx.state.load(model.model_id).stages[n].status for n in PEEKED})
        return default_run("texture", model)

    h = Harness(library, texture=FakeStage("texture", run_fn=peek))
    h.run(models=["cube_palette"])
    fresh = {"verify": StageStatus.SUCCEEDED, "handoff": StageStatus.SUCCEEDED}
    reset = {"verify": StageStatus.PENDING, "handoff": StageStatus.PENDING}
    assert seen in (fresh, reset)  # first run: downstream never succeeded yet
    h.stages["texture"].version = "v2"
    seen.clear()
    h.run(models=["cube_palette"])
    assert seen == {"verify": StageStatus.PENDING, "handoff": StageStatus.PENDING}


# --------------------------------------------------------------------------- force / ranges
def test_force_reruns_everything(harness: Harness) -> None:
    harness.run()
    harness.run(force=True)
    assert harness.counts() == all_counts(4)


def test_force_stage_reruns_stage_and_downstream(harness: Harness) -> None:
    harness.run()
    harness.run(force_stages=["verify"])
    counts = harness.counts()
    assert counts["discover"] == counts["prepare"] == counts["texture"] == 2
    assert counts["verify"] == counts["apply"] == counts["export"] == counts["handoff"] == 4


def test_from_stage_requires_complete_prerequisites(harness: Harness) -> None:
    with pytest.raises(StagePrereqError) as excinfo:
        harness.run(from_stage="texture")
    assert excinfo.value.exit_code == 3
    assert excinfo.value.detail["stage"] == "discover"
    assert harness.counts() == all_counts(0)

    harness.run(to_stage="prepare")
    assert harness.counts()["prepare"] == 2
    assert harness.counts()["texture"] == 0
    assert harness.statuses("cube_rigged")["texture"] == StageStatus.PENDING

    report = harness.run(from_stage="texture")
    assert report.exit_code == 0
    assert harness.counts()["discover"] == 2
    assert harness.counts()["texture"] == harness.counts()["handoff"] == 2
    summary = report.models[0]
    assert summary.stages["discover"] == StageStatus.SUCCEEDED  # outside range: state status
    assert summary.stages["texture"] == StageStatus.SUCCEEDED


def test_from_stage_with_missing_prereq_outputs(harness: Harness, fixture_root: Path) -> None:
    harness.run(to_stage="prepare")
    (fixture_root / "models" / "cube_palette" / "work" / "prepare.txt").unlink()
    with pytest.raises(StagePrereqError, match="outputs missing"):
        harness.run(from_stage="texture")


def test_invalid_stage_range_is_usage_error(harness: Harness) -> None:
    with pytest.raises(UsageError):
        harness.run(from_stage="apply", to_stage="texture")


# --------------------------------------------------------------------------- failures
def test_stage_failure_records_error_and_exit_4(library: Library) -> None:
    def fail(model: ModelContext, ctx: PipelineContext) -> StageOutcome:
        if model.model_id == "cube_rigged":
            raise TexturePackError("bad pack", hint="fix it")
        return default_run("apply", model)

    h = Harness(library, apply=FakeStage("apply", run_fn=fail))
    report = h.run()
    assert report.exit_code == 4
    palette, rigged = report.models
    assert palette.result == "succeeded"
    assert rigged.result == "failed"
    assert rigged.error_code == "E_TEXTURE_PACK"
    assert rigged.error_message == "bad pack"
    assert rigged.stages["apply"] == StageStatus.FAILED
    assert rigged.stages["export"] == StageStatus.PENDING
    state = h.state.load("cube_rigged")
    assert state.stages["apply"].error_code == "E_TEXTURE_PACK"
    assert state.stages["apply"].attempts == 1
    assert state.stages["export"].extra["invalidated_by"] == "apply"
    assert h.counts()["export"] == 1  # only the healthy model continued
    # retry after the fix: failed stage reruns, upstream skipped
    h.stages["apply"].run_fn = None
    second = h.run()
    assert second.exit_code == 0
    assert h.counts()["apply"] == 3
    assert h.counts()["texture"] == 2
    assert h.state.load("cube_rigged").stages["apply"].error_code is None


def test_unexpected_exception_is_recorded_not_swallowed(library: Library) -> None:
    def boom(model: ModelContext, ctx: PipelineContext) -> StageOutcome:
        raise RuntimeError("kaboom")

    h = Harness(library, prepare=FakeStage("prepare", run_fn=boom))
    report = h.run(models=["cube_palette"])
    assert report.exit_code == 4
    summary = report.models[0]
    assert summary.error_code == E_UNEXPECTED
    assert summary.error_message == "RuntimeError: kaboom"
    assert h.state.load("cube_palette").stages["prepare"].error_code == E_UNEXPECTED


def test_fingerprint_error_fails_stage(library: Library) -> None:
    def missing(model: ModelContext, ctx: PipelineContext) -> str:
        raise FileNotFoundError("work/static.glb")

    h = Harness(library, texture=FakeStage("texture", fingerprint_fn=missing))
    report = h.run(models=["cube_palette"])
    assert report.exit_code == 4
    rec = h.state.load("cube_palette").stages["texture"]
    assert rec.status == StageStatus.FAILED
    assert rec.error_code == E_UNEXPECTED
    assert rec.attempts == 1
    assert rec.fingerprint is None
    assert h.counts()["texture"] == 0
    assert h.counts()["verify"] == 0

    dry = h.run(models=["cube_palette"], dry_run=True)
    assert dry.exit_code == 0
    plan = h.runner.last_plan
    assert plan is not None
    texture = next(s for s in plan.models["cube_palette"].stages if s.name == "texture")
    assert texture.action == "run"
    assert texture.reason.startswith("inputs unavailable (FileNotFoundError")


def test_fail_fast_cancels_remaining_models(library: Library) -> None:
    def fail(model: ModelContext, ctx: PipelineContext) -> StageOutcome:
        raise TexturePackError("nope")

    h = Harness(library, prepare=FakeStage("prepare", run_fn=fail))
    report = h.run(keep_going=False, concurrency=1)
    assert report.exit_code == 4
    palette, rigged = report.models
    assert palette.result == "failed"
    assert rigged.result == "cancelled"
    assert rigged.error_code == CancelledError.code
    assert h.counts()["discover"] == 1
    assert h.cancel.is_set()


# --------------------------------------------------------------------------- budget
def test_budget_exceeded_stops_texture_but_keeps_prepare(library: Library) -> None:
    def texture(model: ModelContext, ctx: PipelineContext) -> StageOutcome:
        ctx.budget.reserve(20)  # raises: limit is 10
        return default_run("texture", model)

    h = Harness(library, texture=FakeStage("texture", run_fn=texture))
    report = h.run(max_credits=10)
    assert report.exit_code == 5
    assert report.credits_total == 0
    assert h.template.budget.limit is None  # template untouched; per-run budget replaced
    ctx = h.runner.last_context
    assert ctx is not None
    assert ctx.budget.limit == 10
    for summary in report.models:
        assert summary.result == "failed"
        assert summary.error_code == BudgetExceededError.code
        assert summary.stages["prepare"] == StageStatus.SUCCEEDED
        assert summary.stages["texture"] == StageStatus.FAILED
        assert summary.stages["verify"] == StageStatus.PENDING
    assert h.counts()["prepare"] == 2
    assert h.counts()["verify"] == 0


def test_credits_flow_from_texture_extra_to_report_and_state(library: Library) -> None:
    def texture(model: ModelContext, ctx: PipelineContext) -> StageOutcome:
        ticket = ctx.budget.reserve(10)
        ctx.budget.commit(ticket, 4.5)
        return default_run("texture", model, extra={"credits": 4.5, "task_id": "t-1"})

    h = Harness(library, texture=FakeStage("texture", run_fn=texture))
    report = h.run()
    ctx = h.runner.last_context
    assert ctx is not None
    assert ctx.budget.limit == 50  # [tripo].max_credits_per_run from toml
    assert report.credits_total == 9.0
    assert [m.credits for m in report.models] == [4.5, 4.5]
    state = h.state.load("cube_rigged")
    assert state.credits_total == 4.5
    assert state.stages["texture"].extra == {"credits": 4.5, "task_id": "t-1"}


def test_running_transition_keeps_previous_extra(library: Library) -> None:
    seen: list[dict[str, object]] = []

    def texture(model: ModelContext, ctx: PipelineContext) -> StageOutcome:
        seen.append(dict(ctx.state.load(model.model_id).stages["texture"].extra))
        return default_run("texture", model, extra={"task_id": "new"})

    h = Harness(library, texture=FakeStage("texture", run_fn=texture))
    h.state.update("cube_palette", "texture", extra={"task_id": "pending-123"})
    h.run(models=["cube_palette"])
    assert seen[0]["task_id"] == "pending-123"  # kept through RUNNING (resume semantics)
    assert seen[0]["invalidated_by"] == "prepare"
    assert h.state.load("cube_palette").stages["texture"].extra == {"task_id": "new"}


# --------------------------------------------------------------------------- cancellation / locks
def test_cancellation_mid_run_exit_130(library: Library) -> None:
    started = threading.Event()

    def blocking(model: ModelContext, ctx: PipelineContext) -> StageOutcome:
        started.set()
        if ctx.cancel.wait(10):
            raise CancelledError("cancelled while waiting")
        return default_run("texture", model)

    cancel = CancelToken()
    h = Harness(library, cancel=cancel, texture=FakeStage("texture", run_fn=blocking))

    def cancel_once_started() -> None:
        started.wait(10)
        cancel.set()

    canceller = threading.Thread(target=cancel_once_started)
    canceller.start()
    report = h.run(concurrency=1)
    canceller.join(5)
    assert report.exit_code == 130
    palette, rigged = report.models
    assert palette.result == "cancelled"
    assert palette.stages["texture"] == StageStatus.FAILED
    assert h.state.load("cube_palette").stages["texture"].error_code == CancelledError.code
    assert rigged.result == "cancelled"
    assert rigged.error_message == "cancelled before stage discover"
    assert h.counts()["discover"] == 1


def test_keyboard_interrupt_cancels_and_exits_130(library: Library) -> None:
    def interrupt(model: ModelContext, ctx: PipelineContext) -> StageOutcome:
        raise KeyboardInterrupt

    h = Harness(library, prepare=FakeStage("prepare", run_fn=interrupt))
    report = h.run(models=["cube_palette"])
    assert report.exit_code == 130
    assert report.models[0].result == "cancelled"
    assert h.cancel.is_set()


def test_run_lock_busy(harness: Harness) -> None:
    held, release = threading.Event(), threading.Event()

    def holder() -> None:
        with run_lock(harness.library.paths.run_lock):
            held.set()
            release.wait(10)

    thread = threading.Thread(target=holder)
    thread.start()
    assert held.wait(5)
    try:
        with pytest.raises(LibraryBusyError) as excinfo:
            harness.run()
        assert excinfo.value.exit_code == 2
    finally:
        release.set()
        thread.join(5)
    assert harness.counts() == all_counts(0)
    assert harness.run().exit_code == 0


# --------------------------------------------------------------------------- selection / validation
def test_unknown_model_id_is_usage_error_before_any_work(harness: Harness) -> None:
    with pytest.raises(UsageError, match="unknown model id"):
        harness.run(models=["nope"])
    assert harness.counts() == all_counts(0)
    reports = list(harness.library.paths.runs_dir.glob("*/report.json"))
    assert len(reports) == 1
    assert json.loads(reports[0].read_text())["exit_code"] == 2


def test_tag_selection(harness: Harness) -> None:
    report = harness.run(tags=["rigged"])
    assert [m.model_id for m in report.models] == ["cube_rigged"]
    assert harness.counts() == all_counts(1)


def test_validation_failure_exit_3(harness: Harness, fixture_root: Path) -> None:
    (fixture_root / "models" / "cube_palette" / "source" / "cube_palette.fbx").unlink()
    with pytest.raises(ValidationError) as excinfo:
        harness.run()
    assert excinfo.value.exit_code == 3
    assert any(p.code == "E_SOURCE_MISSING" for p in excinfo.value.problems)
    assert harness.counts() == all_counts(0)
    report = json.loads(next(harness.library.paths.runs_dir.glob("*/report.json")).read_text())
    assert report["exit_code"] == 3
    assert report["problems"][0]["code"] == "E_SOURCE_MISSING"


# --------------------------------------------------------------------------- dry run / plan
def test_dry_run_executes_nothing_and_lists_actions(harness: Harness) -> None:
    report = harness.run(dry_run=True)
    assert report.exit_code == 0
    assert harness.counts() == all_counts(0)
    assert all(m.result == "skipped" for m in report.models)
    assert report.models[0].stages["texture"] == StageStatus.PENDING
    assert report.models[0].credits == 10
    plan = harness.runner.last_plan
    assert plan is not None
    assert plan.budget_limit == 50
    assert plan.estimated_credits == 20
    assert plan_rows(plan) == [
        ("cube_palette", "discover..prepare", "run", "never run", ""),
        ("cube_palette", "texture..handoff", "run", "upstream invalidated", "10"),
        ("cube_rigged", "discover..prepare", "run", "never run", ""),
        ("cube_rigged", "texture..handoff", "run", "upstream invalidated", "10"),
    ]
    assert harness.state.load("cube_rigged").last_run_id is None

    harness.run()
    harness.stages["prepare"].version = "v2"
    harness.run(dry_run=True, models=["cube_rigged"])
    plan = harness.runner.last_plan
    assert plan is not None
    assert plan_rows(plan) == [
        ("cube_rigged", "discover", "skip", "up to date", ""),
        ("cube_rigged", "prepare", "run", "fingerprint changed", ""),
        ("cube_rigged", "texture..handoff", "run", "upstream invalidated", "10"),
    ]
    assert harness.counts()["prepare"] == 2


def test_plan_and_plan_stages_helpers(harness: Harness, library: Library) -> None:
    with pytest.raises(RuntimeError, match="call run"):
        harness.runner.plan_stages(
            model_context(library, harness.template, "cube_rigged"), RunOptions()
        )
    plan = harness.runner.plan(library, RunOptions(models=["cube_rigged"]))
    assert plan.stage_runs == 7
    assert harness.counts() == all_counts(0)
    harness.run()
    rows = harness.runner.plan_stages(
        model_context(library, harness.template, "cube_rigged"), RunOptions()
    )
    assert rows == [
        PlannedStage(n, "skip", "up to date", 0.0, rows[i].fingerprint)
        for i, n in enumerate(STAGE_ORDER)
    ]
    forced = harness.runner.plan_stages(
        model_context(library, harness.template, "cube_rigged"),
        RunOptions(force_stages=["apply"]),
    )
    assert [(s.name, s.action) for s in forced][3:] == [
        ("verify", "skip"),
        ("apply", "run"),
        ("export", "run"),
        ("handoff", "run"),
    ]
    assert forced[4].reason == "forced (--force-stage apply)"


# --------------------------------------------------------------------------- concurrency
def test_concurrency_two_models_overlap(library: Library) -> None:
    barrier = threading.Barrier(2, timeout=10)

    def meet(model: ModelContext, ctx: PipelineContext) -> StageOutcome:
        barrier.wait()  # only passes if both models are inside `prepare` at the same time
        return default_run("prepare", model)

    h = Harness(library, prepare=FakeStage("prepare", run_fn=meet))
    report = h.run(concurrency=2)
    assert report.exit_code == 0
    assert h.counts()["prepare"] == 2


def test_concurrency_one_is_serial(library: Library) -> None:
    events: list[str] = []
    lock = threading.Lock()

    def trace(name: StageName) -> FakeStage:
        def run(model: ModelContext, ctx: PipelineContext) -> StageOutcome:
            with lock:
                events.append(f"{name}:{model.model_id}")
            return default_run(name, model)

        return FakeStage(name, run_fn=run)

    h = Harness(library, discover=trace("discover"), handoff=trace("handoff"))
    h.run()  # library.toml says concurrency = 1
    assert events == [
        "discover:cube_palette",
        "handoff:cube_palette",
        "discover:cube_rigged",
        "handoff:cube_rigged",
    ]


# --------------------------------------------------------------------------- report & context
def test_report_contents_and_file(harness: Harness) -> None:
    report = harness.run(models=["cube_rigged"], force=True)
    assert ULID_RE.match(report.run_id)
    assert report.library == harness.library.paths.root
    assert report.options["force"] is True
    assert report.options["models"] == ["cube_rigged"]
    assert report.environment is not None
    assert report.environment.pipeline_version == "0.1.0"
    assert report.started_at <= report.finished_at
    assert report.problems == []
    path = harness.library.paths.runs_dir / report.run_id / "report.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["models"][0]["stages"]["handoff"] == "succeeded"
    assert data["models"][0]["duration_s"] >= 0
    assert data["exit_code"] == 0


def test_context_factory_receives_run_setup(library: Library) -> None:
    seen: list[RunSetup] = []
    template = make_context(library)

    def factory(setup: RunSetup) -> PipelineContext:
        seen.append(setup)
        return context_from_template(template)(setup)

    graph, _ = fake_graph()
    runner = PipelineRunner(factory, graph)
    report = runner.run(library, RunOptions(max_credits=7))
    assert len(seen) == 1
    assert seen[0].run_id == report.run_id
    assert seen[0].budget.limit == 7
    assert seen[0].options.max_credits == 7
    assert runner.last_context is not None
    assert runner.last_context.run_id == report.run_id


# --------------------------------------------------------------------------- end to end
def test_end_to_end_with_real_discover_stage_and_json_state(library: Library) -> None:
    store = JsonStateStore(library.paths, pipeline_version="0.1.0")
    fakes = [FakeStage(n) for n in LATER]
    graph = StageGraph([DiscoverStage(), *fakes])
    template = make_context(library, state=store)
    runner = PipelineRunner(template, graph)

    report = runner.run(library, RunOptions())

    assert report.exit_code == 0
    for model_id in MODEL_IDS:
        state_file = library.paths.model(model_id).state_file
        assert state_file.is_file()
        state = store.load(model_id)
        assert state.last_run_id == report.run_id
        assert state.stages["discover"].status == StageStatus.SUCCEEDED
        assert state.stages["discover"].fingerprint == discover_fingerprint(
            library, model_id, env={}, cli=CliOverrides(), pipeline_version="0.1.0"
        )
        assert state.stages["discover"].outputs == []
        assert state.stages["handoff"].outputs == ["work/handoff.txt"]
    assert (library.paths.runs_dir / report.run_id / "report.json").is_file()

    second = runner.run(library, RunOptions())
    assert second.exit_code == 0
    assert all(f.run_count == 2 for f in fakes)
    assert all(m.result == "skipped" for m in second.models)
