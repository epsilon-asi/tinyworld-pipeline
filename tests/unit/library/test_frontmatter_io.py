from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import Field
from pydantic import ValidationError as PydanticValidationError

from tinyworld_pipeline.domain._base import FrozenModel
from tinyworld_pipeline.domain.errors import ValidationError
from tinyworld_pipeline.library.frontmatter_io import (
    parse_model_md,
    parse_style_md,
    problems_from_pydantic,
    split_sections,
)

from .helpers import LibraryBuilder, codes, model_md_text


def test_parse_model_md_valid_fixture(fixture_root: Path) -> None:
    doc = parse_model_md(fixture_root / "models" / "cube_rigged" / "model.md", "cube_rigged")

    assert doc.model_id == "cube_rigged"
    assert doc.folder == (fixture_root / "models" / "cube_rigged").resolve()
    assert doc.front.display_name == "Rigged Cube"
    assert doc.front.source.primary == Path("source/cube_rigged.fbx")
    assert doc.front.texture is not None
    assert doc.front.texture.seed == 4242
    assert doc.front.prompt.include_sections == ["base_prompt", "material_notes"]
    assert doc.front.tags == ["fixture", "rigged"]
    assert set(doc.sections) == {"prompt", "notes"}
    assert doc.sections["prompt"].startswith("A small wooden crate cube")


def test_parse_model_md_yaml_error_is_e_frontmatter(builder: LibraryBuilder) -> None:
    folder = builder.model("cube", raw="---\ndisplay_name: [unclosed\n---\n## prompt\nx\n")

    with pytest.raises(ValidationError) as excinfo:
        parse_model_md(folder / "model.md", "cube")

    [problem] = excinfo.value.problems
    assert problem.code == "E_FRONTMATTER"
    assert problem.path == "model.md"
    assert problem.model_id == "cube"
    assert "YAML" in problem.message


def test_parse_model_md_unknown_nested_key_has_did_you_mean_hint(builder: LibraryBuilder) -> None:
    folder = builder.model("cube", {"texture": {"qualty": "fast"}})

    with pytest.raises(ValidationError) as excinfo:
        parse_model_md(folder / "model.md", "cube")

    [problem] = excinfo.value.problems
    assert problem.code == "E_UNKNOWN_KEY"
    assert problem.path == "model.md:texture.qualty"
    assert problem.hint == "did you mean 'quality'?"


def test_parse_model_md_unknown_top_level_key_lists_known_keys(builder: LibraryBuilder) -> None:
    folder = builder.model("cube", {"zzzz": 1})

    with pytest.raises(ValidationError) as excinfo:
        parse_model_md(folder / "model.md", "cube")

    [problem] = excinfo.value.problems
    assert problem.code == "E_UNKNOWN_KEY"
    assert problem.path == "model.md:zzzz"
    assert problem.hint is not None
    assert problem.hint.startswith("known keys: ")
    assert "display_name" in problem.hint


def test_parse_model_md_reports_every_error_at_once(builder: LibraryBuilder) -> None:
    folder = builder.model(
        "cube", {"geometry": {"subdivision_levels": 9}, "texture": {"qualty": "fast"}}
    )

    with pytest.raises(ValidationError) as excinfo:
        parse_model_md(folder / "model.md", "cube")

    by_code = {p.code: p for p in excinfo.value.problems}
    assert set(by_code) == {"E_FRONTMATTER", "E_UNKNOWN_KEY"}
    assert by_code["E_FRONTMATTER"].path == "model.md:geometry.subdivision_levels"


def test_parse_model_md_without_front_matter(builder: LibraryBuilder) -> None:
    folder = builder.model("cube", raw="## prompt\nno front matter here\n")

    with pytest.raises(ValidationError) as excinfo:
        parse_model_md(folder / "model.md", "cube")

    assert codes(excinfo.value.problems) == ["E_FRONTMATTER"]
    assert "missing YAML front matter" in excinfo.value.problems[0].message


def test_parse_model_md_unreadable_path(tmp_path: Path) -> None:
    with pytest.raises(ValidationError) as excinfo:
        parse_model_md(tmp_path, "cube")  # a directory cannot be read as text

    assert codes(excinfo.value.problems) == ["E_FRONTMATTER"]
    assert "cannot read" in excinfo.value.problems[0].message


def test_parse_model_md_missing_prompt_section_parses(builder: LibraryBuilder) -> None:
    folder = builder.model("cube", body="## notes\nonly notes\n")

    doc = parse_model_md(folder / "model.md", "cube")

    assert "prompt" not in doc.sections
    assert doc.sections == {"notes": "only notes"}


def test_parse_model_md_headings_are_case_insensitive(builder: LibraryBuilder) -> None:
    folder = builder.model("cube", body="## PROMPT\nshouty\n\n## Notes\nmixed\n")

    doc = parse_model_md(folder / "model.md", "cube")

    assert doc.sections == {"prompt": "shouty", "notes": "mixed"}


def test_parse_model_md_crlf_file(builder: LibraryBuilder) -> None:
    text = model_md_text({"display_name": "Cube"}, "## prompt\nline one\nline two\n\n## notes\nn\n")
    folder = builder.model("cube", raw=text.replace("\n", "\r\n"))

    doc = parse_model_md(folder / "model.md", "cube")

    assert doc.front.display_name == "Cube"
    assert doc.sections["prompt"] == "line one\nline two"
    assert "\r" not in doc.sections["notes"]


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ("intro text\n## a\n1\n## b\n2", {"a": "1", "b": "2"}),
        ("## a\nfirst\n## a\nsecond", {"a": "second"}),
        ("##\tTabbed  \n  padded  \n", {"tabbed": "padded"}),
        ("## a\n### not a section\nbody", {"a": "### not a section\nbody"}),
        ("", {}),
        ("## empty\n", {"empty": ""}),
    ],
)
def test_split_sections(body: str, expected: dict[str, str]) -> None:
    assert split_sections(body) == expected


def test_parse_style_md_valid_fixture(fixture_root: Path) -> None:
    style = parse_style_md(fixture_root / "style" / "style.md")

    assert style.front.name == "fixture-style"
    assert style.front.negative_terms == ["outline", "pixel art"]
    assert style.folder == (fixture_root / "style").resolve()
    assert set(style.sections) == {"base_prompt", "material_notes"}


def test_parse_style_md_unknown_key(builder: LibraryBuilder) -> None:
    path = builder.style("---\nname: s\nnegativ_terms: []\n---\n## base_prompt\nx\n")

    with pytest.raises(ValidationError) as excinfo:
        parse_style_md(path)

    [problem] = excinfo.value.problems
    assert problem.code == "E_UNKNOWN_KEY"
    assert problem.path == "style.md:negativ_terms"
    assert problem.model_id is None
    assert problem.hint == "did you mean 'negative_terms'?"


class _Inner(FrozenModel):
    x: int = 0


class _Root(FrozenModel):
    meta: dict[str, _Inner] = Field(default_factory=dict)


def test_problems_from_pydantic_without_a_hint_for_unresolvable_locations() -> None:
    with pytest.raises(PydanticValidationError) as excinfo:
        _Root.model_validate({"meta": {"k": {"bogus": 1}}})

    [problem] = problems_from_pydantic(
        excinfo.value, root=_Root, file_name="model.md", model_id="mm"
    )

    assert problem.code == "E_UNKNOWN_KEY"
    assert problem.path == "model.md:meta.k.bogus"
    assert problem.hint is None
