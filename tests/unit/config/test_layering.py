from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pytest

from tinyworld_pipeline.config.defaults import DEFAULTS, SECTION_NAMES
from tinyworld_pipeline.config.layering import (
    CliOverrides,
    env_layer,
    load_library_toml,
    model_layer,
    resolve_config,
    toml_layer,
)
from tinyworld_pipeline.domain.errors import ConfigError
from tinyworld_pipeline.domain.library_config import LibraryConfig
from tinyworld_pipeline.domain.model_spec import ModelFrontMatter

NO_ENV: dict[str, str] = {}
NO_CLI = CliOverrides()


def resolve(
    toml: Mapping[str, object] | None = None,
    model: Mapping[str, object] | None = None,
    env: Mapping[str, str] | None = None,
    cli: CliOverrides = NO_CLI,
) -> LibraryConfig:
    lib = LibraryConfig.model_validate(toml or {})
    front = (
        ModelFrontMatter.model_validate({"display_name": "x", **(model or {})}) if model else None
    )
    cfg, _ = resolve_config(lib, front, env or NO_ENV, cli)
    return cfg


def test_defaults_module_exposes_section_order() -> None:
    assert LibraryConfig() == DEFAULTS
    assert SECTION_NAMES[0] == "library"
    assert "tripo" in SECTION_NAMES


def test_no_layers_yields_defaults() -> None:
    assert resolve() == LibraryConfig()


def test_each_layer_wins_over_the_one_below() -> None:
    toml = {"texture": {"quality": "fast"}}
    model = {"texture": {"quality": "detailed"}}
    env = {"GP_TEXTURE__QUALITY": "extreme"}

    assert resolve(toml).texture.quality == "fast"
    assert resolve(toml, model).texture.quality == "detailed"
    assert resolve(toml, model, env).texture.quality == "extreme"
    # CLI has no quality flag; use route to prove the top layer
    assert (
        resolve({"verify": {"route": "bake"}}, cli=CliOverrides(route="extract")).verify.route
        == "extract"
    )


def test_explicit_value_equal_to_default_still_wins_over_lower_layer() -> None:
    toml = {"geometry": {"subdivision_levels": 3}}
    model = {"geometry": {"subdivision_levels": 1}}  # 1 is the built-in default, set explicitly
    assert resolve(toml, model).geometry.subdivision_levels == 1


def test_unset_override_keys_do_not_clobber_lower_layers() -> None:
    toml = {"geometry": {"subdivision_levels": 3, "shapekeys": "skip"}}
    model = {"geometry": {"shade_smooth": False}}
    cfg = resolve(toml, model)
    assert cfg.geometry.subdivision_levels == 3
    assert cfg.geometry.shapekeys == "skip"
    assert cfg.geometry.shade_smooth is False


def test_lists_replace_rather_than_append() -> None:
    toml = {"bake": {"maps": ["basecolor", "normal"]}}
    model = {"bake": {"maps": ["basecolor"]}}
    assert resolve(toml, model).bake.maps == ["basecolor"]


def test_model_extra_fields_are_not_merged_into_config() -> None:
    model = {
        "texture": {"guidance_images": {"front": "references/front.png"}},
        "unity": {"source_asset_path": "Assets/x.fbx", "shader": "Custom/Lit"},
    }
    cfg = resolve(model=model)
    assert cfg.unity.shader == "Custom/Lit"
    assert "guidance_images" not in type(cfg.texture).model_fields
    front = ModelFrontMatter.model_validate({"display_name": "x", **model})
    assert model_layer(front) == {"unity": {"shader": "Custom/Lit"}}


def test_toml_layer_only_reports_explicit_keys() -> None:
    lib = LibraryConfig.model_validate({"tripo": {"poll_timeout_s": 60}, "uv": {}})
    assert toml_layer(lib) == {"tripo": {"poll_timeout_s": 60}}


@pytest.mark.parametrize(
    ("env", "section", "key", "expected"),
    [
        ({"GP_GEOMETRY__SHADE_SMOOTH": "false"}, "geometry", "shade_smooth", False),
        ({"GP_GEOMETRY__SHADE_SMOOTH": "1"}, "geometry", "shade_smooth", True),
        ({"GP_LIBRARY__CONCURRENCY": "4"}, "library", "concurrency", 4),
        ({"GP_TRIPO__POLL_INTERVAL_S": "3.5"}, "tripo", "poll_interval_s", 3.5),
        ({"GP_BAKE__MAPS": '["basecolor","normal"]'}, "bake", "maps", ["basecolor", "normal"]),
        ({"GP_TRIPO__CREDIT_TABLE": '{"fast": 1}'}, "tripo", "credit_table", {"fast": 1.0}),
        ({"GP_BLENDER__EXECUTABLE": "/opt/blender"}, "blender", "executable", Path("/opt/blender")),
        (
            {"gp_texture__quality": "fast"},
            "texture",
            "quality",
            "standard",
        ),  # prefix is case-sensitive
    ],
)
def test_env_parsing_coerces_types(
    env: dict[str, str], section: str, key: str, expected: object
) -> None:
    cfg = resolve(env=env)
    assert getattr(getattr(cfg, section), key) == expected


def test_unknown_env_vars_warn_but_known_non_config_vars_do_not() -> None:
    env = {
        "GP_TEXTURE__QUALTY": "fast",
        "GP_NOPE": "1",
        "GP_BLENDER_EXE": "/x",
        "GP_LIBRARY": "/lib",
        "HOME": "/home",
    }
    sections, problems = env_layer(env)
    assert sections == {}
    assert [p.code for p in problems] == ["W_UNKNOWN_ENV", "W_UNKNOWN_ENV"]
    assert {p.path for p in problems} == {"GP_TEXTURE__QUALTY", "GP_NOPE"}
    assert all(p.severity == "warning" for p in problems)


def test_env_list_values_must_be_json() -> None:
    with pytest.raises(ConfigError, match="JSON"):
        env_layer({"GP_BAKE__MAPS": "basecolor,normal"})


def test_invalid_values_from_any_layer_raise_config_error() -> None:
    with pytest.raises(ConfigError, match="subdivision_levels"):
        resolve(env={"GP_GEOMETRY__SUBDIVISION_LEVELS": "9"})
    with pytest.raises(ConfigError):
        resolve(cli=CliOverrides(concurrency=99))


def test_cli_overrides_map_to_the_documented_keys() -> None:
    cli = CliOverrides(
        blender_executable=Path("/b"),
        concurrency=3,
        route="bake",
        max_credits=50,
        log_level="DEBUG",
    )
    assert cli.as_sections() == {
        "blender": {"executable": Path("/b")},
        "library": {"concurrency": 3},
        "verify": {"route": "bake"},
        "tripo": {"max_credits_per_run": 50},
        "logging": {"level": "DEBUG"},
    }
    cfg = resolve(cli=cli)
    assert cfg.blender.executable == Path("/b")
    assert cfg.tripo.max_credits_per_run == 50
    assert NO_CLI.as_sections() == {}


def test_load_library_toml_valid_and_invalid(tmp_path: Path) -> None:
    good = tmp_path / "library.toml"
    good.write_text('schema_version = 1\n[texture]\nquality = "detailed"\n', encoding="utf-8")
    assert load_library_toml(good).texture.quality == "detailed"

    bad_key = tmp_path / "bad_key.toml"
    bad_key.write_text("[texture]\nqualty = 'x'\n", encoding="utf-8")
    with pytest.raises(ConfigError, match=r"texture\.qualty"):
        load_library_toml(bad_key)

    bad_syntax = tmp_path / "bad_syntax.toml"
    bad_syntax.write_text("[texture\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_library_toml(bad_syntax)

    with pytest.raises(ConfigError):
        load_library_toml(tmp_path / "missing.toml")


def test_resolve_returns_env_warnings() -> None:
    _, problems = resolve_config(LibraryConfig(), None, {"GP_X": "1"}, NO_CLI)
    assert [p.code for p in problems] == ["W_UNKNOWN_ENV"]
