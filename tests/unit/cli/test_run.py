"""``gp run``: every exit code of plan 08 §2 with the injected fakes, plus production wiring."""

from __future__ import annotations

import json
from concurrent import futures
from concurrent.futures import Future
from pathlib import Path
from typing import cast

import pytest

from fakes.fake_blender_runner import FakeBlenderRunner, job_error_response
from fakes.fake_texture_provider import FakeTextureProvider
from tinyworld_pipeline.cli.commands import run as run_module
from tinyworld_pipeline.domain.state import StageStatus
from tinyworld_pipeline.library.paths import LibraryPaths
from tinyworld_pipeline.pipeline.locks import run_lock
from tinyworld_pipeline.pipeline.state_store import JsonStateStore

from .conftest import Harness, MakeHarness


def _state(lib: Path, model_id: str) -> dict[str, StageStatus]:
    store = JsonStateStore(LibraryPaths.from_root(lib), pipeline_version="0.1.0")
    return {name: rec.status for name, rec in store.load(model_id).stages.items()}


# --------------------------------------------------------------------------- success paths
def test_run_to_prepare_succeeds_and_prints_report_path(harness: Harness, lib: Path) -> None:
    result, document = harness.json("run", "--library", str(lib), "--to-stage", "prepare")
    assert result.exit_code == 0, result.output
    assert harness.blender.call_count("prepare_model") == 2
    report_path = Path(document["report_path"])
    assert report_path.is_file()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["run_id"] == document["run_id"]
    assert [m["result"] for m in document["models"]] == ["succeeded", "succeeded"]
    assert _state(lib, "cube_rigged")["prepare"] == StageStatus.SUCCEEDED
    assert "report:" in harness.text
    assert document["run_id"] in harness.text


def test_run_to_texture_uses_fake_provider_credits(harness: Harness, lib: Path) -> None:
    result, document = harness.json(
        "run", "--library", str(lib), "--to-stage", "texture", "--model", "cube_rigged"
    )
    assert result.exit_code == 0, result.output
    assert document["credits_total"] == 10.0
    assert document["models"] == [
        {
            "model_id": "cube_rigged",
            "result": "succeeded",
            "credits": 10.0,
            "error_code": None,
            "error_message": None,
        }
    ]
    assert len(harness.provider.calls) == 1


def test_second_run_skips_everything(harness: Harness, lib: Path) -> None:
    assert harness.invoke("run", "--library", str(lib), "--to-stage", "prepare").exit_code == 0
    _, document = harness.json("run", "--library", str(lib), "--to-stage", "prepare")
    assert [m["result"] for m in document["models"]] == ["skipped", "skipped"]
    assert harness.blender.call_count("prepare_model") == 2


def test_run_with_tag_and_force(harness: Harness, lib: Path) -> None:
    assert harness.invoke("run", "--library", str(lib), "--to-stage", "prepare").exit_code == 0
    _, document = harness.json(
        "run", "--library", str(lib), "--to-stage", "prepare", "--tag", "rigged", "--force"
    )
    assert [m["model_id"] for m in document["models"]] == ["cube_rigged"]
    assert document["models"][0]["result"] == "succeeded"
    assert harness.blender.call_count("prepare_model") == 3


# --------------------------------------------------------------------------- dry run
def test_dry_run_prints_plan_and_exits_zero_without_lock(harness: Harness, lib: Path) -> None:
    paths = LibraryPaths.from_root(lib)
    with run_lock(paths.run_lock):  # dry-run must not need the run lock
        result, document = harness.json("run", "--library", str(lib), "--dry-run")
    assert result.exit_code == 0
    assert document["dry_run"] is True
    assert document["estimated_credits"] == 20.0
    assert document["budget"] == 50.0
    assert {r["model"] for r in document["rows"]} == {"cube_palette", "cube_rigged"}
    assert "TOTAL estimated credits: 20   (budget: 50)" in harness.text
    assert "est. credits" in harness.text
    assert harness.blender.call_count("prepare_model") == 0
    assert not paths.runs_dir.exists()


# --------------------------------------------------------------------------- failures
def test_run_unknown_model_exits_2(harness: Harness, lib: Path) -> None:
    result, document = harness.json("run", "--library", str(lib), "--model", "ghost")
    assert result.exit_code == 2
    assert document["error"]["code"] == "E_USAGE"


def test_run_invalid_model_id_pattern_exits_2(harness: Harness, lib: Path) -> None:
    result = harness.invoke("run", "--library", str(lib), "--model", "Bad-Id")
    assert result.exit_code == 2
    assert "invalid run option" in harness.text or "unknown model id" in harness.text


def test_run_validation_error_exits_3(harness: Harness, lib: Path) -> None:
    (lib / "models" / "cube_palette" / "source" / "cube_palette.fbx").unlink()
    result, document = harness.json("run", "--library", str(lib), "--to-stage", "prepare")
    assert result.exit_code == 3
    assert document["error"]["code"] == "E_VALIDATION"
    assert harness.blender.call_count("prepare_model") == 0


def test_run_some_failed_exits_4(make_harness: MakeHarness, lib: Path) -> None:
    harness = make_harness(
        blender=FakeBlenderRunner(
            {"prepare_model": job_error_response("E_BJ_SHAPEKEYS", "mesh has shape keys")}
        )
    )
    result, document = harness.json("run", "--library", str(lib), "--to-stage", "prepare")
    assert result.exit_code == 4
    assert document["error"]["code"] == run_module.E_MODELS_FAILED
    assert all(m["result"] == "failed" for m in document["models"])
    assert document["models"][0]["error_code"] == "E_BJ_SHAPEKEYS"
    assert _state(lib, "cube_rigged")["prepare"] == StageStatus.FAILED
    assert "E_BJ_SHAPEKEYS" in harness.text


def test_run_budget_exceeded_exits_5(make_harness: MakeHarness, lib: Path) -> None:
    harness = make_harness(provider=FakeTextureProvider(credits=100.0))
    result, document = harness.json(
        "run", "--library", str(lib), "--to-stage", "texture", "--max-credits", "10"
    )
    assert result.exit_code == 5
    assert document["error"]["code"] == "E_BUDGET"
    assert harness.provider.calls == []  # refused before any provider call
    assert _state(lib, "cube_rigged")["prepare"] == StageStatus.SUCCEEDED


def test_run_library_busy_exits_2(harness: Harness, lib: Path) -> None:
    with run_lock(LibraryPaths.from_root(lib).run_lock):
        result, document = harness.json("run", "--library", str(lib), "--to-stage", "discover")
    assert result.exit_code == 2
    assert document["error"]["code"] == "E_LIBRARY_BUSY"


def test_run_from_stage_prerequisite_exits_3(harness: Harness, lib: Path) -> None:
    result, document = harness.json(
        "run", "--library", str(lib), "--from-stage", "texture", "--to-stage", "texture"
    )
    assert result.exit_code == 3
    assert document["error"]["code"] == "E_STAGE_PREREQ"


def test_run_inverted_stage_range_exits_2(harness: Harness, lib: Path) -> None:
    result = harness.invoke(
        "run", "--library", str(lib), "--from-stage", "apply", "--to-stage", "prepare"
    )
    assert result.exit_code == 2


def test_run_default_range_fails_downstream_with_fake_static_glb(
    harness: Harness, lib: Path
) -> None:
    """Without ``--to-stage`` the fakes carry the run past texture, where the fake GLB fails."""
    result, document = harness.json("run", "--library", str(lib), "--model", "cube_rigged")
    assert result.exit_code == 4
    assert document["models"][0]["result"] == "failed"
    assert document["models"][0]["error_code"]
    assert _state(lib, "cube_rigged")["texture"] == StageStatus.SUCCEEDED
    assert _state(lib, "cube_rigged")["handoff"] == StageStatus.PENDING


def test_run_keyboard_interrupt_while_polling_sets_cancel(
    harness: Harness, lib: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = {"n": 0}

    def interrupting_wait(fs: object, timeout: float | None = None) -> object:
        calls["n"] += 1
        if calls["n"] == 1:
            raise KeyboardInterrupt
        return futures.wait(cast("list[Future[object]]", fs), timeout=timeout)

    monkeypatch.setattr("tinyworld_pipeline.cli.commands.run.wait", interrupting_wait)
    result = harness.invoke("run", "--library", str(lib), "--to-stage", "prepare")
    assert result.exit_code in (0, 130)
    assert "cancelling" in harness.text


# --------------------------------------------------------------------------- production wiring
def test_production_wiring_runs_discover_without_blender_or_key(
    make_harness: MakeHarness, lib: Path
) -> None:
    harness = make_harness(inject_context=False)
    result, document = harness.json("run", "--library", str(lib), "--to-stage", "discover")
    assert result.exit_code == 0, result.output
    run_dir = LibraryPaths.from_root(lib).runs_dir / document["run_id"]
    assert (run_dir / "report.json").is_file()
    assert (run_dir / "run.log").is_file()  # tee attached by the production context


def test_production_wiring_requires_key_for_texture(make_harness: MakeHarness, lib: Path) -> None:
    harness = make_harness(inject_context=False)
    result, document = harness.json("run", "--library", str(lib), "--to-stage", "texture")
    assert result.exit_code == 2
    assert "TRIPO_API_KEY" in document["error"]["message"]


def test_production_wiring_unusable_blender_fails_prepare(
    make_harness: MakeHarness, lib: Path
) -> None:
    harness = make_harness(inject_context=False)  # stub resolver points at /fake/blender
    result, document = harness.json(
        "run", "--library", str(lib), "--to-stage", "prepare", "--model", "cube_rigged"
    )
    assert result.exit_code == 4
    assert document["models"][0]["error_code"] == "E_BLENDER_NOT_FOUND"
    assert (LibraryPaths.from_root(lib).pipeline_dir / "blender.json").is_file()
