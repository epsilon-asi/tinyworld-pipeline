from __future__ import annotations

import re
from pathlib import Path

import pytest

from tinyworld_pipeline.config.layering import load_library_toml
from tinyworld_pipeline.domain.errors import UsageError
from tinyworld_pipeline.domain.library_config import SECTION_MODELS, LibraryConfig
from tinyworld_pipeline.library.discovery import Library
from tinyworld_pipeline.library.frontmatter_io import parse_model_md
from tinyworld_pipeline.library.scaffold import (
    default_display_name,
    init_library,
    new_model,
    render_library_toml,
)

from .helpers import FBX_PLACEHOLDER, codes, errors, validate, write_png


# --------------------------------------------------------------------------- init
def test_init_creates_every_scaffold_file(tmp_path: Path) -> None:
    paths = init_library(tmp_path / "lib")

    assert paths.library_toml.is_file()
    assert paths.style_md.is_file()
    assert (paths.style_references / ".gitkeep").is_file()
    assert (paths.models_dir / ".gitkeep").is_file()
    assert (paths.generative_dir / ".gitkeep").is_file()
    ignored = paths.gitignore.read_text().split()
    assert {"work/", ".pipeline/", "output/"} <= set(ignored)


def test_init_refuses_existing_library_without_force(tmp_path: Path) -> None:
    init_library(tmp_path / "lib")

    with pytest.raises(UsageError) as excinfo:
        init_library(tmp_path / "lib")

    assert excinfo.value.exit_code == 2


def test_init_force_overwrites_toml_but_keeps_style(tmp_path: Path) -> None:
    paths = init_library(tmp_path / "lib")
    paths.library_toml.write_text("schema_version = 1\n[library]\nconcurrency = 9\n")
    paths.style_md.write_text("---\nname: mine\n---\n## base_prompt\nmine\n")

    init_library(tmp_path / "lib", force=True)

    assert load_library_toml(paths.library_toml) == LibraryConfig()
    assert "mine" in paths.style_md.read_text()


def test_generated_toml_round_trips_to_defaults(tmp_path: Path) -> None:
    paths = init_library(tmp_path / "lib")

    assert load_library_toml(paths.library_toml) == LibraryConfig()


def test_generated_toml_mentions_every_key_with_a_comment_or_value() -> None:
    text = render_library_toml()

    assert text.startswith("#")
    assert "schema_version = 1" in text
    for section, model in SECTION_MODELS.items():
        assert f"\n[{section}]\n" in text
        for key in model.model_fields:
            pattern = rf"^(#\s*)?{re.escape(key)} = |^\[{section}\.{key}\]"
            assert re.search(pattern, text, re.MULTILINE), f"{section}.{key} missing"
    assert '# unity_import_dir = ""' in text  # None → commented out
    assert "[tripo.credit_table]" in text
    assert 'quality = "standard"  #' in text  # keys carry comments


def test_render_library_toml_of_custom_config_round_trips(tmp_path: Path) -> None:
    config = LibraryConfig.model_validate(
        {
            "library": {"unity_import_dir": str(tmp_path / "u"), "concurrency": 3},
            "bake": {"maps": ["normal"]},
        }
    )
    path = tmp_path / "library.toml"

    path.write_text(render_library_toml(config))

    assert load_library_toml(path) == config


def test_initialised_library_loads_and_validates(tmp_path: Path) -> None:
    paths = init_library(tmp_path / "lib")

    lib = Library.load(paths.root)

    assert lib.load_problems == []
    assert lib.style is not None
    assert "base_prompt" in lib.style.sections
    assert validate(lib) == []


# --------------------------------------------------------------------------- new-model
@pytest.fixture
def source_fbx(tmp_path: Path) -> Path:
    path = tmp_path / "vendor" / "Crate.fbx"
    path.parent.mkdir()
    path.write_bytes(FBX_PLACEHOLDER)
    return path


def test_new_model_copy_writes_template_and_keeps_source(tmp_path: Path, source_fbx: Path) -> None:
    lib = init_library(tmp_path / "lib")

    paths = new_model(lib.root, "crate_small", source_fbx)

    assert (paths.source / "Crate.fbx").read_bytes() == FBX_PLACEHOLDER
    assert source_fbx.exists()
    assert paths.references.is_dir()
    doc = parse_model_md(paths.model_md, "crate_small")
    assert doc.front.display_name == "Crate Small"
    assert doc.front.source.primary == Path("source/Crate.fbx")
    assert "animations" not in doc.front.source.model_fields_set
    assert set(doc.sections) == {"prompt", "notes"}


def test_new_model_move(tmp_path: Path, source_fbx: Path) -> None:
    lib = init_library(tmp_path / "lib")

    paths = new_model(lib.root, "crate", source_fbx, mode="move")

    assert (paths.source / "Crate.fbx").is_file()
    assert not source_fbx.exists()


def test_new_model_link(tmp_path: Path, source_fbx: Path) -> None:
    lib = init_library(tmp_path / "lib")

    paths = new_model(lib.root, "crate", source_fbx, mode="link")

    dest = paths.source / "Crate.fbx"
    assert dest.is_symlink()
    assert dest.resolve() == source_fbx.resolve()


def test_new_model_display_name_and_anim_globs(tmp_path: Path, source_fbx: Path) -> None:
    lib = init_library(tmp_path / "lib")

    paths = new_model(
        lib.root,
        "crate",
        source_fbx,
        display_name="Wooden Crate: v2",
        anim_globs=("source/animations/*.fbx", "source/clips/*.fbx"),
    )

    doc = parse_model_md(paths.model_md, "crate")
    assert doc.front.display_name == "Wooden Crate: v2"
    assert doc.front.source.animations == ["source/animations/*.fbx", "source/clips/*.fbx"]


def test_new_model_then_validate(tmp_path: Path, source_fbx: Path) -> None:
    lib = init_library(tmp_path / "lib")
    write_png(lib.style_references / "primary.png")
    new_model(lib.root, "crate", source_fbx)

    problems = validate(Library.load(lib.root))

    assert errors(problems) == []
    assert sorted(codes(problems)) == ["E_ANIM_GLOB_EMPTY", "W_LICENSE_MISSING"]


@pytest.mark.parametrize("bad_id", ["Crate", "a", "crate-1", "_x"])
def test_new_model_rejects_invalid_id(tmp_path: Path, source_fbx: Path, bad_id: str) -> None:
    lib = init_library(tmp_path / "lib")

    with pytest.raises(UsageError):
        new_model(lib.root, bad_id, source_fbx)


def test_new_model_rejects_missing_source(tmp_path: Path) -> None:
    lib = init_library(tmp_path / "lib")

    with pytest.raises(UsageError):
        new_model(lib.root, "crate", tmp_path / "nope.fbx")


def test_new_model_refuses_existing_folder(tmp_path: Path, source_fbx: Path) -> None:
    lib = init_library(tmp_path / "lib")
    new_model(lib.root, "crate", source_fbx)

    with pytest.raises(UsageError) as excinfo:
        new_model(lib.root, "crate", source_fbx)

    assert excinfo.value.exit_code == 2


def test_default_display_name() -> None:
    assert default_display_name("worker_ant") == "Worker Ant"
    assert default_display_name("x2__y") == "X2 Y"
