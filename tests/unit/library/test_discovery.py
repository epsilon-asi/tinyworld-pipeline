from __future__ import annotations

import pytest

from tinyworld_pipeline.domain.errors import UsageError, ValidationError
from tinyworld_pipeline.domain.library_config import LibraryConfig
from tinyworld_pipeline.library.discovery import (
    Library,
    find_animation_fbx,
    find_primary_fbx,
)

from .helpers import FBX_PLACEHOLDER, LibraryBuilder, codes


# --------------------------------------------------------------------------- load
def test_load_fixture_library(fixture_library: Library) -> None:
    assert fixture_library.load_problems == []
    assert fixture_library.config.library.name == "library_minimal"
    assert fixture_library.config.texture.quality == "fast"
    assert fixture_library.config.geometry.subdivision_type == "catmull"  # default kept
    assert fixture_library.style is not None
    assert "base_prompt" in fixture_library.style.sections


def test_load_invalid_toml_syntax_records_e_library_toml(builder: LibraryBuilder) -> None:
    builder.toml("schema_version = \n")

    lib = builder.load()

    assert codes(lib.load_problems) == ["E_LIBRARY_TOML"]
    assert lib.config == LibraryConfig()


def test_load_unknown_toml_key_records_e_library_toml(builder: LibraryBuilder) -> None:
    builder.toml('schema_version = 1\n[texture]\nqualty = "fast"\n')

    lib = builder.load()

    [problem] = lib.load_problems
    assert problem.code == "E_LIBRARY_TOML"
    assert "texture.qualty" in problem.message


def test_load_missing_library_toml_records_e_library_toml(builder: LibraryBuilder) -> None:
    (builder.root / "library.toml").unlink()

    lib = builder.load()

    assert lib.config == LibraryConfig()
    assert codes(lib.load_problems) == ["E_LIBRARY_TOML"]


def test_load_missing_style_md(builder: LibraryBuilder) -> None:
    (builder.root / "style" / "style.md").unlink()

    lib = builder.load()

    assert lib.style is None
    [problem] = lib.load_problems
    assert (problem.code, problem.path) == ("E_FRONTMATTER", "style/style.md")


def test_load_broken_style_md(builder: LibraryBuilder) -> None:
    builder.style("---\nname: [\n---\n## base_prompt\nx\n")

    lib = builder.load()

    assert lib.style is None
    assert codes(lib.load_problems) == ["E_FRONTMATTER"]


# --------------------------------------------------------------------------- enumeration
def test_model_ids_sorted_and_hidden_skipped(builder: LibraryBuilder) -> None:
    builder.model("zebra")
    builder.model("ant")
    builder.model("Bad-Name")
    (builder.root / "models" / ".hidden").mkdir()
    (builder.root / "models" / ".gitkeep").touch()

    assert builder.load().model_ids() == ["Bad-Name", "ant", "zebra"]


def test_model_ids_without_models_dir(builder: LibraryBuilder) -> None:
    assert builder.load().model_ids() == []


def test_select_defaults_to_all(fixture_library: Library) -> None:
    assert fixture_library.select() == ["cube_palette", "cube_rigged"]
    assert fixture_library.select(ids=[], tags=[]) == ["cube_palette", "cube_rigged"]


def test_select_unknown_id_is_usage_error(fixture_library: Library) -> None:
    with pytest.raises(UsageError) as excinfo:
        fixture_library.select(ids=["nope"])

    assert excinfo.value.exit_code == 2
    assert "cube_rigged" in (excinfo.value.hint or "")


def test_select_by_tag(fixture_library: Library) -> None:
    assert fixture_library.select(tags=["rigged"]) == ["cube_rigged"]
    assert fixture_library.select(tags=["fixture"]) == ["cube_palette", "cube_rigged"]
    assert fixture_library.select(tags=["none_such"]) == []


def test_select_ids_and_tags_are_unioned(fixture_library: Library) -> None:
    assert fixture_library.select(ids=["cube_palette"], tags=["rigged"]) == [
        "cube_palette",
        "cube_rigged",
    ]


def test_select_by_tag_skips_invalid_folder_names(builder: LibraryBuilder) -> None:
    builder.model("ok", {"tags": ["t"]})
    builder.model("Bad-Name", {"tags": ["t"]})

    assert builder.load().select(tags=["t"]) == ["ok"]


# --------------------------------------------------------------------------- documents
def test_model_document_is_cached(fixture_library: Library) -> None:
    first = fixture_library.model_document("cube_rigged")

    assert fixture_library.model_document("cube_rigged") is first


def test_model_document_unknown_or_invalid_id(fixture_library: Library) -> None:
    with pytest.raises(UsageError):
        fixture_library.model_document("missing")
    with pytest.raises(UsageError):
        fixture_library.model_document("Bad-Name")


def test_model_document_missing_model_md(builder: LibraryBuilder) -> None:
    (builder.root / "models" / "empty").mkdir(parents=True)

    with pytest.raises(ValidationError) as excinfo:
        builder.load().model_document("empty")

    [problem] = excinfo.value.problems
    assert (problem.code, problem.model_id, problem.path) == (
        "E_MODEL_MD_MISSING",
        "empty",
        "model.md",
    )


# --------------------------------------------------------------------------- primary fbx
def test_find_primary_fbx_explicit(fixture_library: Library) -> None:
    doc = fixture_library.model_document("cube_rigged")

    primary, problems = find_primary_fbx(doc)

    assert problems == []
    assert primary == doc.folder / "source" / "cube_rigged.fbx"


def test_find_primary_fbx_explicit_missing(builder: LibraryBuilder) -> None:
    builder.model("cube", {"source": {"primary": "source/other.fbx"}})

    primary, problems = find_primary_fbx(builder.load().model_document("cube"))

    assert primary is None
    [problem] = problems
    assert (problem.code, problem.path) == ("E_SOURCE_MISSING", "model.md:source.primary")


def test_find_primary_fbx_single_autodetected_case_insensitive(builder: LibraryBuilder) -> None:
    builder.model("cube", fbx=("source/Cube.FBX", "source/animations/walk.fbx"))

    primary, problems = find_primary_fbx(builder.load().model_document("cube"))

    assert problems == []
    assert primary is not None
    assert primary.name == "Cube.FBX"


def test_find_primary_fbx_none_in_source(builder: LibraryBuilder) -> None:
    builder.model("cube", fbx=())

    primary, problems = find_primary_fbx(builder.load().model_document("cube"))

    assert primary is None
    assert codes(problems) == ["E_SOURCE_MISSING"]


def test_find_primary_fbx_ambiguous(builder: LibraryBuilder) -> None:
    builder.model("cube", fbx=("source/a.fbx", "source/b.fbx"))

    primary, problems = find_primary_fbx(builder.load().model_document("cube"))

    assert primary is None
    [problem] = problems
    assert problem.code == "E_SOURCE_AMBIGUOUS"
    assert "a.fbx, b.fbx" in problem.message


# --------------------------------------------------------------------------- animations
def test_find_animation_fbx_sorted_deduplicated_and_excludes_primary(
    builder: LibraryBuilder,
) -> None:
    builder.model(
        "cube",
        {
            "source": {
                "primary": "source/cube.fbx",
                "animations": ["source/animations/*.fbx", "source/**/*.fbx"],
            }
        },
        fbx=("source/cube.fbx", "source/animations/b_run.fbx", "source/animations/a_walk.fbx"),
    )
    doc = builder.load().model_document("cube")

    anims, problems = find_animation_fbx(doc, primary=doc.folder / "source" / "cube.fbx")

    assert problems == []
    assert [a.name for a in anims] == ["a_walk.fbx", "b_run.fbx"]


def test_find_animation_fbx_explicit_empty_glob_is_error(builder: LibraryBuilder) -> None:
    builder.model("cube", {"source": {"animations": ["source/clips/*.fbx"]}})

    anims, problems = find_animation_fbx(builder.load().model_document("cube"))

    assert anims == []
    [problem] = problems
    assert (problem.code, problem.severity) == ("E_ANIM_GLOB_EMPTY", "error")


def test_find_animation_fbx_default_empty_glob_is_warning(builder: LibraryBuilder) -> None:
    builder.model("cube")

    anims, problems = find_animation_fbx(builder.load().model_document("cube"))

    assert anims == []
    [problem] = problems
    assert (problem.code, problem.severity) == ("E_ANIM_GLOB_EMPTY", "warning")


def test_find_animation_fbx_explicit_empty_list_is_silent(builder: LibraryBuilder) -> None:
    builder.model("cube", {"source": {"animations": []}})
    (builder.root / "models/cube/source/animations").mkdir()
    (builder.root / "models/cube/source/animations/x.fbx").write_bytes(FBX_PLACEHOLDER)

    assert find_animation_fbx(builder.load().model_document("cube")) == ([], [])
