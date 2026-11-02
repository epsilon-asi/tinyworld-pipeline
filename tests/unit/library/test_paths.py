from __future__ import annotations

from pathlib import Path

from tinyworld_pipeline.library.paths import LibraryPaths, ModelPaths, find_library_root


def test_model_paths_derive_from_folder(tmp_path: Path) -> None:
    paths = ModelPaths.for_model(tmp_path / "models", "ant")

    assert paths.folder == tmp_path / "models" / "ant"
    assert paths.model_md == paths.folder / "model.md"
    assert paths.source == paths.folder / "source"
    assert paths.animations == paths.source / "animations"
    assert paths.textures == paths.source / "textures"
    assert paths.references == paths.folder / "references"
    assert paths.work == paths.folder / "work"
    assert paths.output == paths.folder / "output"
    assert paths.pipeline == paths.folder / ".pipeline"
    assert paths.state_file == paths.pipeline / "state.json"
    assert paths.lock_file == paths.pipeline / "state.lock"
    assert paths.logs == paths.pipeline / "logs"


def test_library_paths_layout(tmp_path: Path) -> None:
    lib = LibraryPaths.from_root(tmp_path / "lib")
    root = (tmp_path / "lib").resolve()

    assert lib.root == root
    assert lib.library_toml == root / "library.toml"
    assert lib.style_md == root / "style" / "style.md"
    assert lib.style_references == root / "style" / "references"
    assert lib.models_dir == root / "models"
    assert lib.generative_dir == root / "generative"
    assert lib.pipeline_dir == root / ".pipeline"
    assert lib.runs_dir == root / ".pipeline" / "runs"
    assert lib.run_lock == root / ".pipeline" / "run.lock"
    assert lib.gitignore == root / ".gitignore"
    assert lib.model("ant").folder == root / "models" / "ant"


def test_find_library_root_explicit_flag_wins(tmp_path: Path) -> None:
    (tmp_path / "env_lib").mkdir()
    env = {"GP_LIBRARY": str(tmp_path / "env_lib")}

    found = find_library_root(tmp_path, env, explicit=tmp_path / "flag_lib")

    assert found == (tmp_path / "flag_lib").resolve()


def test_find_library_root_env_wins_over_ancestor(tmp_path: Path) -> None:
    (tmp_path / "library.toml").write_text("schema_version = 1\n")
    env = {"GP_LIBRARY": str(tmp_path / "env_lib")}

    assert find_library_root(tmp_path, env) == (tmp_path / "env_lib").resolve()


def test_find_library_root_walks_up_to_nearest_library_toml(tmp_path: Path) -> None:
    (tmp_path / "library.toml").write_text("schema_version = 1\n")
    nested = tmp_path / "models" / "ant" / "source"
    nested.mkdir(parents=True)

    assert find_library_root(nested, {}) == tmp_path.resolve()


def test_find_library_root_falls_back_to_dot_library_dir(tmp_path: Path) -> None:
    (tmp_path / "library").mkdir()

    assert find_library_root(tmp_path, {}) == (tmp_path / "library").resolve()


def test_find_library_root_returns_none_when_nothing_matches(tmp_path: Path) -> None:
    assert find_library_root(tmp_path, {"GP_LIBRARY": ""}) is None
