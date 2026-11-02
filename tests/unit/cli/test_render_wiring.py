"""Rendering helpers and production wiring pieces that the command tests do not reach."""

from __future__ import annotations

import io
from pathlib import Path
from typing import cast

import pytest
from rich.console import Console as RichConsole

from grounded_pipeline import __version__
from grounded_pipeline.blender.protocol import JobRequest
from grounded_pipeline.blender.runner import SubprocessBlenderRunner
from grounded_pipeline.cli import render, wiring
from grounded_pipeline.cli.state import CliHooks, CliState
from grounded_pipeline.config.layering import CliOverrides
from grounded_pipeline.domain.errors import (
    BlenderNotFoundError,
    PipelineError,
    Problem,
    TripoAuthError,
)
from grounded_pipeline.domain.library_config import BlenderSection
from grounded_pipeline.domain.ports import CancelToken
from grounded_pipeline.domain.state import (
    STAGE_ORDER,
    ModelState,
    StageName,
    StageRecord,
    StageStatus,
)
from grounded_pipeline.library.discovery import Library
from grounded_pipeline.library.paths import LibraryPaths
from grounded_pipeline.library.resolve import resolve_model
from grounded_pipeline.observability.console import Console
from grounded_pipeline.observability.logging import configure_logging
from grounded_pipeline.pipeline.context import ModelContext, PipelineContext
from grounded_pipeline.pipeline.options import RunOptions
from grounded_pipeline.pipeline.runner import RunSetup
from grounded_pipeline.texturing.provider import TextureRequest
from grounded_pipeline.texturing.tripo_provider import TripoTextureProvider
from grounded_pipeline.tripo.budget import CreditBudget

from .conftest import FAKE_BLENDER, TEST_KEY, mocked_client_factory, stub_resolver

RUN_ID = "01ARZ3NDEKTSV4RRFFQ69G5FAV"


def _model_state(**statuses: StageStatus) -> ModelState:
    state = ModelState(model_id="cube", pipeline_version="0.1.0")
    stages = dict(state.stages)
    for name, status in statuses.items():
        stages[cast("StageName", name)] = StageRecord(
            status=status, error_code="E_X", error_message="bad", finished_at="t"
        )
    state.stages = stages
    return state


# --------------------------------------------------------------------------- render
def test_current_stage_and_last_error() -> None:
    assert render.current_stage(_model_state()) == "waiting"
    assert render.current_stage(_model_state(prepare=StageStatus.RUNNING)) == "prepare"
    assert render.current_stage(_model_state(texture=StageStatus.FAILED)) == "texture ✗"
    done = _model_state(**dict.fromkeys(STAGE_ORDER, StageStatus.SUCCEEDED))
    assert render.current_stage(done) == "done"
    assert render.last_error(_model_state()) == "-"
    assert render.last_error(_model_state(texture=StageStatus.FAILED)) == "E_X: bad"
    assert render.short_run_id(None) == "-"
    assert render.short_run_id(RUN_ID) == "…" + RUN_ID[-8:]


def test_validation_status_and_grouping() -> None:
    err = Problem(code="E_A", severity="error", model_id="m1", message="x")
    warn = Problem(code="W_B", severity="warning", message="y")
    assert render.validation_status([]) == "ok"
    assert render.validation_status([warn]) == "warning"
    assert render.validation_status([warn, err]) == "error"
    groups = render.group_problems([err, warn], ["m1", "m2"])
    assert list(groups) == ["", "m1", "m2"]
    assert groups[""] == [warn]
    assert groups["m2"] == []


def test_run_progress_renders_rows_on_a_terminal() -> None:
    buffer = io.StringIO()
    console = Console(no_color=True, force_terminal=True, width=100, file=buffer)
    console._rich = RichConsole(file=buffer, force_terminal=True, width=100, no_color=True)
    with render.RunProgress(console, ["cube"]) as progress:
        progress.refresh({"cube": _model_state(prepare=StageStatus.RUNNING)})
        progress.refresh({"other": _model_state()})
        progress.note("hello")
    assert "cube" in buffer.getvalue()
    assert "hello" in buffer.getvalue()


def test_run_progress_is_silent_in_json_mode() -> None:
    console = Console.recording(json_mode=True)
    with render.RunProgress(console, ["cube"]) as progress:
        progress.refresh({"cube": _model_state()})
        progress.note("hidden")
    assert console.text == ""


def test_status_table_silent_in_json_mode() -> None:
    console = Console.recording(json_mode=True)
    render.status_table(console, [_model_state()])
    assert console.text == ""


def test_doctor_checklist_reports_failures() -> None:
    console = Console.recording()
    render.doctor_checklist(
        console, [render.Check("a", True, "ok"), render.Check("b", False, "no")]
    )
    assert "1 check(s) failed" in console.text


# --------------------------------------------------------------------------- wiring
def _state(tmp_path: Path, env: dict[str, str] | None = None) -> CliState:
    return CliState(
        library=None,
        json=False,
        log_level=None,
        blender=None,
        no_color=True,
        env=env or {},
        cwd=tmp_path,
        console=Console.recording(),
        hooks=CliHooks(
            None,
            mocked_client_factory,
            stub_resolver(),
            lambda: wiring.secrets_from_mapping(env or {}),
        ),
    )


def _setup(lib: Path) -> RunSetup:
    return RunSetup(Library.load(lib), RunOptions(), RUN_ID, CreditBudget(None))


def _any_model(ctx: PipelineContext) -> ModelContext:
    resolved, _ = resolve_model(ctx.library, "cube_rigged", env={}, cli=CliOverrides())
    return ModelContext(
        resolved=resolved, paths=ctx.library.paths.model("cube_rigged"), log=ctx.log
    )


def test_build_graph_covers_every_stage_and_placeholder_fails(lib: Path, tmp_path: Path) -> None:
    graph = wiring.build_graph()
    assert graph.names == STAGE_ORDER
    assert set(wiring.STAGE_FACTORIES) == set(STAGE_ORDER)
    ctx = wiring.build_production_context(
        _setup(lib), state=_state(tmp_path), secrets=wiring.secrets_from_mapping({}), blender=None
    )
    model = _any_model(ctx)
    placeholder = wiring.NotYetImplementedStage("handoff")
    assert placeholder.input_fingerprint(model, ctx) == placeholder.input_fingerprint(model, ctx)
    with pytest.raises(PipelineError, match="not implemented yet"):
        placeholder.run(model, ctx)


def test_production_context_without_blender_or_key(lib: Path, tmp_path: Path) -> None:
    state = _state(tmp_path, {"GP_TEXTURE__QUALITY": "detailed", "HOME": "/x"})
    ctx = wiring.build_production_context(
        _setup(lib), state=state, secrets=wiring.secrets_from_mapping({}), blender=None
    )
    assert ctx.run_id == RUN_ID
    assert ctx.env == {"GP_TEXTURE__QUALITY": "detailed"}
    assert ctx.pipeline_version == __version__
    assert state.active_cancel is ctx.cancel
    request = JobRequest(job="probe", job_id=RUN_ID, params={})
    with pytest.raises(BlenderNotFoundError):
        ctx.blender.run(request, timeout_s=1, log_path=tmp_path / "l", cancel=CancelToken())
    req = cast("TextureRequest", object())  # the stand-in never inspects the request
    assert ctx.provider.estimate_credits(req) == 0.0
    with pytest.raises(TripoAuthError):
        ctx.provider.texture(req, resume_task_id=None, cancel=CancelToken())


def test_production_context_with_blender_key_and_log_tee(lib: Path, tmp_path: Path) -> None:
    state = _state(tmp_path, {"TRIPO_API_KEY": TEST_KEY})
    state.logging_handle = configure_logging(
        level="INFO", json_console=False, no_color=True, run_log_path=None
    )
    try:
        ctx = wiring.build_production_context(
            _setup(lib),
            state=state,
            secrets=wiring.secrets_from_mapping({"TRIPO_API_KEY": TEST_KEY}),
            blender=FAKE_BLENDER,
            logging_handle=state.logging_handle,
        )
        assert isinstance(ctx.blender, SubprocessBlenderRunner)
        assert isinstance(ctx.provider, TripoTextureProvider)
        run_log = Library.load(lib).paths.runs_dir / RUN_ID / wiring.RUN_LOG_FILENAME
        assert run_log.is_file()
    finally:
        state.close()
    assert state.logging_handle is None


def test_blender_problems_and_cache(tmp_path: Path) -> None:
    state = _state(tmp_path)
    info, problems = wiring.blender_problems(state, BlenderSection())
    assert info == FAKE_BLENDER
    assert problems == []
    cache = wiring.write_blender_cache(LibraryPaths.from_root(tmp_path / "cache_lib"), FAKE_BLENDER)
    assert cache.name == wiring.BLENDER_CACHE_FILENAME
    assert "4.5.12" in cache.read_text(encoding="utf-8")


def test_cli_overrides_from_state(tmp_path: Path) -> None:
    state = _state(tmp_path)
    state.blender = Path("/b")
    state.log_level = "DEBUG"
    overrides = wiring.cli_overrides_from(state, concurrency=3, route="bake", max_credits=7.5)
    assert overrides.blender_executable == Path("/b")
    assert overrides.concurrency == 3
    assert overrides.route == "bake"
    assert overrides.max_credits == 7.5
    assert overrides.log_level == "DEBUG"


def test_secrets_from_mapping_and_default_loader() -> None:
    assert wiring.secrets_from_mapping({}).has_tripo_key is False
    secrets = wiring.secrets_from_mapping({"TRIPO_API_KEY": TEST_KEY})
    assert secrets.has_tripo_key is True
    assert secrets.redacted_tripo_key() == "****cdef"
    assert wiring.default_secrets_loader() is not None
    assert wiring.process_environment() is not None
