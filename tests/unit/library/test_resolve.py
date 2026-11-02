from __future__ import annotations

from pathlib import Path

import pytest

from tinyworld_pipeline.config.layering import CliOverrides
from tinyworld_pipeline.domain.errors import Problem, UsageError, ValidationError
from tinyworld_pipeline.domain.model_spec import ResolvedModel
from tinyworld_pipeline.library.discovery import Library
from tinyworld_pipeline.library.resolve import discover_fingerprint, overrides_dump, resolve_model

from .helpers import LibraryBuilder, codes

NO_CLI = CliOverrides()


def resolve(
    lib: Library, model_id: str, env: dict[str, str] | None = None
) -> tuple[ResolvedModel, list[Problem]]:
    return resolve_model(lib, model_id, env=env or {}, cli=NO_CLI)


# --------------------------------------------------------------------------- resolve_model
def test_resolve_model_fixture_cube_rigged(fixture_library: Library) -> None:
    resolved, warnings = resolve(fixture_library, "cube_rigged")

    model = fixture_library.paths.model("cube_rigged")
    assert warnings == []
    assert resolved.doc.model_id == "cube_rigged"
    assert resolved.primary_fbx == model.source / "cube_rigged.fbx"
    assert resolved.animation_fbx == [model.animations / "cube_rigged@walk.fbx"]
    assert resolved.config.texture.quality == "fast"  # library.toml
    assert resolved.config.texture.seed == 4242  # model.md wins over toml's 7
    assert resolved.config.library.concurrency == 1
    assert resolved.style is fixture_library.style
    assert "Painted wood" in resolved.composed_prompt.text
    assert resolved.composed_prompt.sources["material_notes"] == "style.md:material_notes"


def test_resolve_model_applies_env_layer(fixture_library: Library) -> None:
    resolved, _ = resolve(fixture_library, "cube_palette", {"GP_TEXTURE__QUALITY": "detailed"})

    assert resolved.config.texture.quality == "detailed"
    assert resolved.animation_fbx == []


def test_resolve_model_returns_warnings(builder: LibraryBuilder) -> None:
    builder.model("cube")

    _, warnings = resolve(builder.load(), "cube")

    assert sorted(codes(warnings)) == ["E_ANIM_GLOB_EMPTY", "W_LICENSE_MISSING"]
    assert all(w.severity == "warning" for w in warnings)


def test_resolve_model_raises_on_model_errors(builder: LibraryBuilder) -> None:
    builder.model("cube", fbx=())

    with pytest.raises(ValidationError) as excinfo:
        resolve(builder.load(), "cube")

    assert "E_SOURCE_MISSING" in codes(excinfo.value.problems)


def test_resolve_model_raises_on_library_load_errors(builder: LibraryBuilder) -> None:
    builder.toml("broken = [\n")
    builder.model("cube")

    with pytest.raises(ValidationError) as excinfo:
        resolve(builder.load(), "cube")

    assert codes(excinfo.value.problems) == ["E_LIBRARY_TOML"]


def test_resolve_model_unknown_id(fixture_library: Library) -> None:
    with pytest.raises(UsageError):
        resolve(fixture_library, "nope")


# --------------------------------------------------------------------------- fingerprint
def fingerprint(
    lib: Library,
    env: dict[str, str] | None = None,
    cli: CliOverrides | None = None,
    version: str = "0.1.0",
) -> str:
    return discover_fingerprint(
        lib, "cube_rigged", env=env or {}, cli=cli or NO_CLI, pipeline_version=version
    )


def test_fingerprint_is_stable_and_well_formed(fixture_library: Library) -> None:
    first = fingerprint(fixture_library)

    assert first == fingerprint(fixture_library)
    assert len(first) == 64
    assert fingerprint(fixture_library, env={"HOME": "/elsewhere", "PATH": "/x"}) == first


def test_fingerprint_differs_between_models(fixture_library: Library) -> None:
    other = discover_fingerprint(
        fixture_library, "cube_palette", env={}, cli=NO_CLI, pipeline_version="0.1.0"
    )

    assert other != fingerprint(fixture_library)


@pytest.mark.parametrize("rel", ["models/cube_rigged/model.md", "library.toml", "style/style.md"])
def test_fingerprint_changes_when_input_file_changes(fixture_root: Path, rel: str) -> None:
    lib = Library.load(fixture_root)
    before = fingerprint(lib)

    path = fixture_root / rel
    path.write_text(path.read_text() + "\n# touched\n")

    assert fingerprint(lib) != before


def test_fingerprint_changes_with_gp_env_cli_and_version(fixture_library: Library) -> None:
    base = fingerprint(fixture_library)

    assert fingerprint(fixture_library, env={"GP_TEXTURE__QUALITY": "fast"}) != base
    assert fingerprint(fixture_library, cli=CliOverrides(route="bake")) != base
    assert fingerprint(fixture_library, version="0.2.0") != base


def test_fingerprint_missing_style_hashes_as_empty(builder: LibraryBuilder) -> None:
    builder.model("cube")
    lib = builder.load()
    with_style = discover_fingerprint(lib, "cube", env={}, cli=NO_CLI, pipeline_version="0.1.0")
    (builder.root / "style" / "style.md").unlink()

    after = discover_fingerprint(lib, "cube", env={}, cli=NO_CLI, pipeline_version="0.1.0")
    assert after != with_style


def test_overrides_dump_is_canonical_and_filters_env() -> None:
    dump = overrides_dump({"Z": "1", "GP_B": "2", "GP_A": "1"}, CliOverrides(route="bake"))

    assert dump.startswith('{"cli": {"blender_executable": null, ')
    assert '"route": "bake"' in dump
    assert dump.endswith('"env": {"GP_A": "1", "GP_B": "2"}}')
