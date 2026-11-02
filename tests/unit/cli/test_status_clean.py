"""``gp status`` and ``gp clean``."""

from __future__ import annotations

from pathlib import Path

from fakes.fake_blender_runner import FakeBlenderRunner, job_error_response

from .conftest import Harness, MakeHarness


def _prepared(harness: Harness, lib: Path) -> None:
    assert harness.invoke("run", "--library", str(lib), "--to-stage", "prepare").exit_code == 0


# --------------------------------------------------------------------------- status
def test_status_fresh_library_all_pending(harness: Harness, lib: Path) -> None:
    result, document = harness.json("status", "--library", str(lib))
    assert result.exit_code == 0
    assert [m["model_id"] for m in document["models"]] == ["cube_palette", "cube_rigged"]
    assert set(document["models"][0]["stages"].values()) == {"pending"}
    assert document["models"][0]["last_run_id"] is None
    assert document["models"][0]["last_error"] is None
    assert "○" in harness.text
    assert "legend:" in harness.text


def test_status_after_run_shows_glyphs_and_last_run(harness: Harness, lib: Path) -> None:
    _prepared(harness, lib)
    harness.console.clear()
    result, document = harness.json("status", "--library", str(lib), "--model", "cube_rigged")
    assert result.exit_code == 0
    model = document["models"][0]
    assert model["stages"]["discover"] == "succeeded"
    assert model["stages"]["prepare"] == "succeeded"
    assert model["stages"]["texture"] == "pending"
    assert model["last_run_id"]
    assert "✓" in harness.text
    assert "…" + model["last_run_id"][-8:] in harness.text


def test_status_shows_last_error(make_harness: MakeHarness, lib: Path) -> None:
    harness = make_harness(
        blender=FakeBlenderRunner({"prepare_model": job_error_response("E_BJ_CRASH", "bang")})
    )
    assert harness.invoke("run", "--library", str(lib), "--to-stage", "prepare").exit_code == 4
    _, document = harness.json("status", "--library", str(lib), "--model", "cube_palette")
    assert document["models"][0]["last_error"] == "E_BJ_CRASH: bang"
    assert "✗" in harness.text


# --------------------------------------------------------------------------- clean
def test_clean_requires_a_target(harness: Harness, lib: Path) -> None:
    result = harness.invoke("clean", "--library", str(lib))
    assert result.exit_code == 2
    assert "--all" in harness.text


def test_clean_nothing_to_do(harness: Harness, lib: Path) -> None:
    result, document = harness.json("clean", "--library", str(lib), "--work")
    assert result.exit_code == 0
    assert document["removed"] == []


def test_clean_confirmation_no_keeps_files(harness: Harness, lib: Path) -> None:
    _prepared(harness, lib)
    work = lib / "models" / "cube_rigged" / "work"
    result = harness.invoke("clean", "--library", str(lib), "--work", stdin="n\n")
    assert result.exit_code == 0
    assert work.is_dir()
    assert "aborted" in harness.text


def test_clean_confirmation_yes_removes_work_only(harness: Harness, lib: Path) -> None:
    _prepared(harness, lib)
    model = lib / "models" / "cube_rigged"
    result = harness.invoke(
        "clean", "--library", str(lib), "--work", "--model", "cube_rigged", stdin="y\n"
    )
    assert result.exit_code == 0
    assert not (model / "work").exists()
    assert (model / ".pipeline" / "state.json").is_file()
    assert (model / "source" / "cube_rigged.fbx").is_file()
    assert (model / "model.md").is_file()
    assert (lib / "models" / "cube_palette" / "work").is_dir()


def test_clean_yes_flag_and_all(harness: Harness, lib: Path) -> None:
    _prepared(harness, lib)
    (lib / "models" / "cube_rigged" / "output").mkdir()
    result, document = harness.json("clean", "--library", str(lib), "--all", "--yes")
    assert result.exit_code == 0
    kinds = {(r["model_id"], r["kind"]) for r in document["removed"]}
    assert kinds == {
        ("cube_palette", "work"),
        ("cube_palette", "state"),
        ("cube_rigged", "work"),
        ("cube_rigged", "output"),
        ("cube_rigged", "state"),
    }
    for model_id in ("cube_palette", "cube_rigged"):
        folder = lib / "models" / model_id
        assert not (folder / "work").exists()
        assert not (folder / ".pipeline").exists()
        assert (folder / "source").is_dir()
        assert (folder / "model.md").is_file()


def test_clean_json_does_not_prompt(harness: Harness, lib: Path) -> None:
    _prepared(harness, lib)
    result, document = harness.json("clean", "--library", str(lib), "--state")
    assert result.exit_code == 0
    assert document["confirmed"] is True
    assert not (lib / "models" / "cube_rigged" / ".pipeline").exists()
