"""One test per code in plan 05 §8 (plus the library-level extras)."""

from __future__ import annotations

from pathlib import Path

import pytest

from tinyworld_pipeline.config.layering import CliOverrides, resolve_config
from tinyworld_pipeline.domain.errors import Problem, UsageError, ValidationError
from tinyworld_pipeline.domain.library_config import LibraryConfig
from tinyworld_pipeline.library import validation
from tinyworld_pipeline.library.discovery import Library
from tinyworld_pipeline.library.validation import (
    ImageInfo,
    image_format_problems,
    inspect_image,
    raise_if_errors,
    resolve_style_image,
)

from .helpers import (
    LibraryBuilder,
    codes,
    errors,
    validate,
    write_jpeg,
    write_png,
)


def only(problems: list[Problem], code: str) -> Problem:
    matching = [p for p in problems if p.code == code]
    assert len(matching) == 1, codes(problems)
    return matching[0]


# --------------------------------------------------------------------------- the clean fixture
def test_fixture_library_validates_clean(fixture_library: Library) -> None:
    assert validate(fixture_library) == []


def test_baseline_builder_model_has_only_expected_warnings(builder: LibraryBuilder) -> None:
    builder.model("cube")

    problems = validate(builder.load())

    assert errors(problems) == []
    assert sorted(codes(problems)) == ["E_ANIM_GLOB_EMPTY", "W_LICENSE_MISSING"]


# --------------------------------------------------------------------------- per-model codes
def test_e_model_id(builder: LibraryBuilder) -> None:
    builder.model("Bad-Name")

    problem = only(validate(builder.load()), "E_MODEL_ID")

    assert problem.path == "models/Bad-Name/"
    assert problem.model_id is None


def test_e_model_md_missing(builder: LibraryBuilder) -> None:
    (builder.root / "models" / "ghost").mkdir(parents=True)

    assert codes(validate(builder.load())) == ["E_MODEL_MD_MISSING"]


def test_e_frontmatter(builder: LibraryBuilder) -> None:
    builder.model("cube", raw="---\ndisplay_name: [\n---\n## prompt\nx\n")

    assert codes(validate(builder.load())) == ["E_FRONTMATTER"]


def test_e_frontmatter_from_invalid_effective_config(builder: LibraryBuilder) -> None:
    builder.model("cube")

    problems = validate(builder.load(), env={"GP_GEOMETRY__SUBDIVISION_LEVELS": "9"})

    problem = only(problems, "E_FRONTMATTER")
    assert "subdivision_levels" in problem.message


def test_e_unknown_key(builder: LibraryBuilder) -> None:
    builder.model("cube", {"texture": {"qualty": "fast"}})

    assert codes(validate(builder.load())) == ["E_UNKNOWN_KEY"]


def test_e_source_missing(builder: LibraryBuilder) -> None:
    builder.model("cube", fbx=())

    only(validate(builder.load()), "E_SOURCE_MISSING")


def test_e_source_ambiguous(builder: LibraryBuilder) -> None:
    builder.model("cube", fbx=("source/a.fbx", "source/b.fbx"))

    only(validate(builder.load()), "E_SOURCE_AMBIGUOUS")


def test_e_anim_glob_empty(builder: LibraryBuilder) -> None:
    builder.model("cube", {"source": {"animations": ["source/clips/*.fbx"]}})

    problem = only(validate(builder.load()), "E_ANIM_GLOB_EMPTY")

    assert problem.severity == "error"


@pytest.mark.parametrize("mode", ["text", "text+style"])
def test_e_prompt_missing_in_text_modes(builder: LibraryBuilder, mode: str) -> None:
    builder.model("cube", {"texture": {"prompt_mode": mode}}, body="## notes\nnothing\n")

    only(validate(builder.load()), "E_PROMPT_MISSING")


def test_e_prompt_missing_not_required_in_image_mode(builder: LibraryBuilder) -> None:
    write_png(builder.root / "models/cube/references/front.png")
    builder.model(
        "cube",
        {"texture": {"prompt_mode": "image", "guidance_images": {"front": "references/front.png"}}},
        body="## notes\nnothing\n",
    )

    assert "E_PROMPT_MISSING" not in codes(validate(builder.load()))


def test_e_prompt_too_long(builder: LibraryBuilder) -> None:
    builder.model("cube", body="## prompt\n" + "word " * 250 + "\n")

    problem = only(validate(builder.load()), "E_PROMPT_TOO_LONG")

    assert problem.severity == "error"
    assert "W_PROMPT_LONG" not in codes(validate(builder.load()))


def test_w_prompt_long(builder: LibraryBuilder) -> None:
    builder.model("cube", body="## prompt\n" + "word " * 130 + "\n")

    problem = only(validate(builder.load()), "W_PROMPT_LONG")

    assert problem.severity == "warning"


def test_e_template_placeholder(builder: LibraryBuilder) -> None:
    builder.model("cube", {"prompt": {"template": "{base_prompt} {colour}"}})

    problem = only(validate(builder.load()), "E_TEMPLATE_PLACEHOLDER")

    assert problem.path == "model.md:prompt.template"
    assert "{colour}" in problem.message


def test_e_style_image_missing(builder: LibraryBuilder) -> None:
    (builder.root / "style/references/primary.png").unlink()
    builder.model("cube")

    problem = only(validate(builder.load()), "E_STYLE_IMAGE_MISSING")

    assert problem.path == "library.toml:texture.style_image"


def test_style_image_not_required_in_text_mode(builder: LibraryBuilder) -> None:
    (builder.root / "style/references/primary.png").unlink()
    builder.model("cube", {"texture": {"prompt_mode": "text"}})

    assert "E_STYLE_IMAGE_MISSING" not in codes(validate(builder.load()))


def test_e_path_ambiguous(builder: LibraryBuilder) -> None:
    write_png(builder.root / "refs/s.png")
    write_png(builder.root / "models/cube/refs/s.png")
    builder.model("cube", {"texture": {"style_image": "refs/s.png"}})

    problem = only(validate(builder.load()), "E_PATH_AMBIGUOUS")

    assert problem.path == "model.md:texture.style_image"


def test_model_relative_style_image_is_accepted(builder: LibraryBuilder) -> None:
    write_png(builder.root / "models/cube/references/look.png")
    builder.model("cube", {"texture": {"style_image": "references/look.png"}})
    lib = builder.load()

    doc = lib.model_document("cube")
    config, _ = resolve_config(lib.config, doc.front, {}, CliOverrides())

    path, problems = resolve_style_image(lib, doc, config)

    assert problems == []
    assert path == lib.paths.model("cube").references / "look.png"


def test_implicit_references_style_png_overrides_library_image(builder: LibraryBuilder) -> None:
    (builder.root / "style/references/primary.png").unlink()  # would be E_STYLE_IMAGE_MISSING
    write_png(builder.root / "models/cube/references/style.png")
    builder.model("cube")

    assert "E_STYLE_IMAGE_MISSING" not in codes(validate(builder.load()))


def test_absolute_style_image_path(builder: LibraryBuilder, tmp_path: Path) -> None:
    image = write_png(tmp_path / "elsewhere" / "style.png")
    builder.model("cube", {"texture": {"style_image": str(image)}})

    assert errors(validate(builder.load())) == []


def test_e_ref_image_format_not_png_or_jpeg(builder: LibraryBuilder) -> None:
    (builder.root / "style/references/primary.png").write_bytes(b"GIF89a not really")
    builder.model("cube")

    problem = only(validate(builder.load()), "E_REF_IMAGE_FORMAT")

    assert "not a PNG or JPEG" in problem.message


def test_e_ref_image_format_too_small(builder: LibraryBuilder) -> None:
    write_png(builder.root / "style/references/primary.png", (64, 300))
    builder.model("cube")

    problem = only(validate(builder.load()), "E_REF_IMAGE_FORMAT")

    assert "64x300" in problem.message


def test_e_ref_image_format_too_large(
    builder: LibraryBuilder, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(validation, "REF_IMAGE_MAX_BYTES", 10)
    builder.model("cube")

    problem = only(validate(builder.load()), "E_REF_IMAGE_FORMAT")

    assert "bytes" in problem.message


def test_e_ref_image_format_truncated_png(builder: LibraryBuilder) -> None:
    (builder.root / "style/references/primary.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\0" * 8)
    builder.model("cube")

    only(validate(builder.load()), "E_REF_IMAGE_FORMAT")


def test_jpeg_style_image_is_accepted(builder: LibraryBuilder) -> None:
    write_jpeg(builder.root / "style/references/primary.jpg")
    builder.toml('schema_version = 1\n[texture]\nstyle_image = "style/references/primary.jpg"\n')
    builder.model("cube")

    assert errors(validate(builder.load())) == []


def test_e_guidance_images_missing_block(builder: LibraryBuilder) -> None:
    builder.model("cube", {"texture": {"prompt_mode": "image"}})

    problem = only(validate(builder.load()), "E_GUIDANCE_IMAGES")

    assert "front" in problem.message


def test_e_guidance_images_too_few_views(builder: LibraryBuilder) -> None:
    write_png(builder.root / "models/cube/references/front.png")
    builder.model(
        "cube",
        {
            "texture": {
                "prompt_mode": "images",
                "guidance_images": {"front": "references/front.png"},
            }
        },
    )

    problem = only(validate(builder.load()), "E_GUIDANCE_IMAGES")

    assert "at least 2 views" in problem.message


def test_e_guidance_images_file_missing(builder: LibraryBuilder) -> None:
    builder.model(
        "cube",
        {"texture": {"prompt_mode": "image", "guidance_images": {"front": "references/front.png"}}},
    )

    problem = only(validate(builder.load()), "E_GUIDANCE_IMAGES")

    assert problem.path == "model.md:texture.guidance_images.front"
    assert "not found" in problem.message


def test_e_guidance_images_file_too_large(
    builder: LibraryBuilder, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(validation, "REF_IMAGE_MAX_BYTES", 10)
    write_png(builder.root / "models/cube/references/front.png")
    builder.model(
        "cube",
        {"texture": {"prompt_mode": "image", "guidance_images": {"front": "references/front.png"}}},
    )

    problem = only(validate(builder.load()), "E_GUIDANCE_IMAGES")

    assert "exceeds" in problem.message


def test_guidance_image_bad_format_is_e_ref_image_format(builder: LibraryBuilder) -> None:
    write_png(builder.root / "models/cube/references/front.png", (100, 100))
    builder.model(
        "cube",
        {"texture": {"prompt_mode": "image", "guidance_images": {"front": "references/front.png"}}},
    )

    problem = only(validate(builder.load()), "E_REF_IMAGE_FORMAT")

    assert problem.path == "model.md:texture.guidance_images.front"


def test_w_guidance_unverified_on_valid_images_mode(builder: LibraryBuilder) -> None:
    write_png(builder.root / "models/cube/references/front.png")
    write_jpeg(builder.root / "models/cube/references/back.jpg")
    builder.model(
        "cube",
        {
            "source": {"license": "CC0", "animations": []},
            "texture": {
                "prompt_mode": "images",
                "guidance_images": {"front": "references/front.png", "back": "references/back.jpg"},
            },
        },
    )

    problems = validate(builder.load())

    assert codes(problems) == ["W_GUIDANCE_UNVERIFIED"]
    assert problems[0].severity == "warning"


def test_w_license_missing(builder: LibraryBuilder) -> None:
    builder.model("cube")

    problem = only(validate(builder.load()), "W_LICENSE_MISSING")

    assert problem.severity == "warning"


def test_w_license_satisfied_by_license_file(builder: LibraryBuilder) -> None:
    builder.model("cube")
    builder.write("models/cube/source/LICENSE.txt", "CC0\n")

    assert "W_LICENSE_MISSING" not in codes(validate(builder.load()))


def test_w_license_satisfied_by_front_matter(builder: LibraryBuilder) -> None:
    builder.model("cube", {"source": {"license": "CC-BY 4.0"}})

    assert "W_LICENSE_MISSING" not in codes(validate(builder.load()))


# --------------------------------------------------------------------------- library-level codes
def test_e_library_toml(builder: LibraryBuilder) -> None:
    builder.toml("not = [valid\n")

    assert codes(validate(builder.load())) == ["E_LIBRARY_TOML"]


def test_e_unity_dir(builder: LibraryBuilder) -> None:
    builder.toml('schema_version = 1\n[library]\nunity_import_dir = "/definitely/not/here"\n')

    problem = only(validate(builder.load()), "E_UNITY_DIR")

    assert problem.path == "library.toml:library.unity_import_dir"


def test_unity_dir_existing_is_fine(builder: LibraryBuilder, tmp_path: Path) -> None:
    unity = tmp_path / "Assets"
    unity.mkdir()
    builder.toml(f'schema_version = 1\n[library]\nunity_import_dir = "{unity}"\n')

    assert validate(builder.load()) == []


def test_e_style_base_prompt_missing(builder: LibraryBuilder) -> None:
    builder.style("---\nname: s\n---\n## material_notes\nmatte\n")
    builder.model("cube", {"source": {"license": "CC0", "animations": []}})

    problems = validate(builder.load())

    assert codes(problems) == ["E_STYLE_BASE_PROMPT_MISSING"]


def test_missing_style_md_still_validates_models(builder: LibraryBuilder) -> None:
    (builder.root / "style" / "style.md").unlink()
    builder.model("cube", {"source": {"license": "CC0", "animations": []}}, fbx=())

    problems = validate(builder.load())

    assert codes(problems) == ["E_FRONTMATTER", "E_SOURCE_MISSING"]
    assert problems[0].path == "style/style.md"


def test_blender_check_problems_are_appended(builder: LibraryBuilder) -> None:
    seen: list[LibraryConfig] = []

    def check(config: LibraryConfig) -> list[Problem]:
        seen.append(config)
        return [Problem(code="E_BLENDER_NOT_FOUND", severity="error", message="no blender")]

    problems = validate(builder.load(), blender_check=check)

    assert codes(problems) == ["E_BLENDER_NOT_FOUND"]
    assert seen == [LibraryConfig.model_validate({"library": {"name": "test"}})]


def test_validate_unknown_model_id_is_usage_error(fixture_library: Library) -> None:
    with pytest.raises(UsageError):
        validate(fixture_library, ["nope"])


def test_validate_selected_model_only(builder: LibraryBuilder) -> None:
    builder.model("good", {"source": {"license": "CC0", "animations": []}})
    builder.model("bad", fbx=())

    assert validate(builder.load(), ["good"]) == []
    assert "E_SOURCE_MISSING" in codes(validate(builder.load(), ["bad"]))


# --------------------------------------------------------------------------- helpers
def test_raise_if_errors() -> None:
    warning = Problem(code="W_X", severity="warning", message="w")
    error = Problem(code="E_X", severity="error", message="e")

    raise_if_errors([warning])
    with pytest.raises(ValidationError) as excinfo:
        raise_if_errors([warning, error])

    assert excinfo.value.problems == [warning, error]
    assert excinfo.value.exit_code == 3


def test_inspect_image_missing_file(tmp_path: Path) -> None:
    assert inspect_image(tmp_path / "nope.png") == ImageInfo(False, None, 0, 0, 0)


def test_inspect_image_reads_jpeg_header(tmp_path: Path) -> None:
    info = inspect_image(write_jpeg(tmp_path / "a.jpg", (300, 257)))

    assert (info.exists, info.format, info.width, info.height) == (True, "jpeg", 300, 257)
    assert info.size_bytes > 0


def test_image_format_problems_accepts_good_image() -> None:
    info = ImageInfo(True, "png", 256, 256, 1024)

    assert image_format_problems(info, model_id=None, path="x") == []
