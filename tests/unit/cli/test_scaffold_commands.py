"""``gp init`` and ``gp new-model``."""

from __future__ import annotations

from pathlib import Path

from .conftest import FIXTURE_LIBRARY, Harness, MakeHarness

SOURCE_FBX = FIXTURE_LIBRARY / "models" / "cube_rigged" / "source" / "cube_rigged.fbx"


# --------------------------------------------------------------------------- init
def test_init_creates_library(harness: Harness, tmp_path: Path) -> None:
    target = tmp_path / "newlib"
    result, document = harness.json("init", str(target))
    assert result.exit_code == 0
    assert (target / "library.toml").is_file()
    assert (target / "style" / "style.md").is_file()
    assert (target / "models" / ".gitkeep").is_file()
    assert (target / ".gitignore").is_file()
    assert document["path"] == str(target.resolve())
    assert document["forced"] is False


def test_init_existing_exits_2_and_force_overwrites(harness: Harness, lib: Path) -> None:
    result = harness.invoke("init", str(lib))
    assert result.exit_code == 2
    assert "already exists" in harness.text
    assert "--force" in harness.text
    forced = harness.invoke("init", str(lib), "--force")
    assert forced.exit_code == 0
    assert "initialised library" in harness.text


def test_init_default_targets(make_harness: MakeHarness, tmp_path: Path) -> None:
    via_env = tmp_path / "from_env"
    harness = make_harness(env={"GP_LIBRARY": str(via_env)}, cwd=tmp_path)
    assert harness.invoke("init").exit_code == 0
    assert (via_env / "library.toml").is_file()

    harness = make_harness(cwd=tmp_path)
    assert harness.invoke("init").exit_code == 0
    assert (tmp_path / "library" / "library.toml").is_file()

    via_flag = tmp_path / "from_flag"
    assert harness.invoke("--library", str(via_flag), "init").exit_code == 0
    assert (via_flag / "library.toml").is_file()


# --------------------------------------------------------------------------- new-model
def test_new_model_copies_source_and_writes_model_md(harness: Harness, lib: Path) -> None:
    result, document = harness.json(
        "new-model",
        "crate_big",
        "--library",
        str(lib),
        "--source",
        str(SOURCE_FBX),
        "--display-name",
        "Big Crate",
        "--anim",
        "source/animations/*.fbx",
    )
    assert result.exit_code == 0, result.output
    folder = lib / "models" / "crate_big"
    assert (folder / "source" / "cube_rigged.fbx").is_file()
    assert SOURCE_FBX.is_file()  # copy, not move
    assert "Big Crate" in (folder / "model.md").read_text(encoding="utf-8")
    assert document["model_id"] == "crate_big"
    assert document["mode"] == "copy"


def test_new_model_existing_folder_exits_2(harness: Harness, lib: Path) -> None:
    result = harness.invoke(
        "new-model", "cube_rigged", "--library", str(lib), "--source", str(SOURCE_FBX)
    )
    assert result.exit_code == 2
    assert "already exists" in harness.text


def test_new_model_move_and_link(harness: Harness, lib: Path, tmp_path: Path) -> None:
    movable = tmp_path / "movable.fbx"
    movable.write_bytes(SOURCE_FBX.read_bytes())
    moved = harness.invoke(
        "new-model", "moved_one", "--library", str(lib), "--source", str(movable), "--move"
    )
    assert moved.exit_code == 0
    assert not movable.exists()
    assert (lib / "models" / "moved_one" / "source" / "movable.fbx").is_file()

    linked = harness.invoke(
        "new-model", "linked_one", "--library", str(lib), "--source", str(SOURCE_FBX), "--link"
    )
    assert linked.exit_code == 0
    assert (lib / "models" / "linked_one" / "source" / "cube_rigged.fbx").is_symlink()


def test_new_model_conflicting_modes_and_bad_inputs(
    harness: Harness, lib: Path, tmp_path: Path
) -> None:
    both = harness.invoke(
        "new-model", "x_one", "--library", str(lib), "--source", str(SOURCE_FBX), "--move", "--link"
    )
    assert both.exit_code == 2
    assert "mutually exclusive" in harness.text

    bad_id = harness.invoke(
        "new-model", "Bad-Id", "--library", str(lib), "--source", str(SOURCE_FBX)
    )
    assert bad_id.exit_code == 2

    missing = harness.invoke(
        "new-model", "y_one", "--library", str(lib), "--source", str(tmp_path / "nope.fbx")
    )
    assert missing.exit_code == 2
