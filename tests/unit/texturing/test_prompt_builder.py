from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from grounded_pipeline.domain.errors import PromptTooLongError, ValidationError
from grounded_pipeline.domain.model_spec import ModelDocument, ModelFrontMatter, PromptSpec
from grounded_pipeline.domain.prompt import PROMPT_MAX_CHARS, PROMPT_WARN_CHARS
from grounded_pipeline.domain.style_spec import StyleDocument, StyleFrontMatter
from grounded_pipeline.texturing.prompt_builder import (
    compose_prompt,
    compose_with_warnings,
    normalize_whitespace,
    prompt_warnings,
)

STYLE = StyleDocument(
    folder=Path("/lib/style"),
    front=StyleFrontMatter(name="s"),
    sections={
        "base_prompt": "Hand-painted   stylized game asset.",
        "material_notes": "Matte paint.",
        "palette": "#FF0000 red",
    },
)


def make_doc(
    prompt: str | None = "A red cube.",
    *,
    tags: list[str] | None = None,
    display_name: str = "Red Cube",
) -> ModelDocument:
    sections = {"prompt": prompt} if prompt is not None else {}
    return ModelDocument(
        model_id="red_cube",
        folder=Path("/lib/models/red_cube"),
        front=ModelFrontMatter(display_name=display_name, tags=tags or []),
        sections=sections,
    )


def test_default_template_uses_base_prompt_and_model_prompt() -> None:
    cp = compose_prompt(STYLE, make_doc(), PromptSpec())

    assert cp.text == "Hand-painted stylized game asset.\n\nA red cube."
    assert cp.sources == {"base_prompt": "style.md:base_prompt", "model_prompt": "model.md:prompt"}
    assert cp.length == len(cp.text)


def test_every_placeholder_kind_resolves() -> None:
    spec = PromptSpec(
        include_sections=["base_prompt", "material_notes", "palette"],
        template="{display_name} [{tags}]: {base_prompt} {model_prompt} {material_notes} {palette}",
    )

    cp = compose_prompt(STYLE, make_doc(tags=["insect", "enemy"]), spec)

    assert cp.text == (
        "Red Cube [insect, enemy]: Hand-painted stylized game asset. A red cube. Matte paint. "
        "#FF0000 red"
    )
    assert cp.sources["tags"] == "model.md:tags"
    assert cp.sources["display_name"] == "model.md:display_name"


def test_sections_not_included_resolve_to_empty() -> None:
    spec = PromptSpec(template="{base_prompt}|{material_notes}|{palette}|{model_prompt}")

    cp = compose_prompt(STYLE, make_doc(), spec)

    assert cp.text == "Hand-painted stylized game asset.|||A red cube."
    assert cp.sources["material_notes"] == ""
    assert cp.sources["palette"] == ""


def test_included_section_absent_from_style_is_empty_not_error() -> None:
    spec = PromptSpec(include_sections=["base_prompt", "lighting"], template="{lighting}X")

    assert compose_prompt(STYLE, make_doc(), spec).text == "X"


def test_missing_model_prompt_and_tags_record_empty_sources() -> None:
    spec = PromptSpec(template="{model_prompt}{tags}{base_prompt}")

    cp = compose_prompt(STYLE, make_doc(prompt=None), spec)

    assert cp.sources == {"model_prompt": "", "tags": "", "base_prompt": "style.md:base_prompt"}


def test_whitespace_normalisation() -> None:
    doc = make_doc("  line   one \t two\r\n\r\n\r\n\r\nline\u000bthree  \n\n\n")

    cp = compose_prompt(STYLE, doc, PromptSpec(template="\n\n{model_prompt}\n\n\n\n"))

    assert cp.text == "line one two\n\nline three"


def test_unknown_placeholder_is_e_template_placeholder() -> None:
    spec = PromptSpec(template="{base_prompt} {colour} {size}")

    with pytest.raises(ValidationError) as excinfo:
        compose_prompt(STYLE, make_doc(), spec)

    [problem] = excinfo.value.problems
    assert problem.code == "E_TEMPLATE_PLACEHOLDER"
    assert problem.path == "model.md:prompt.template"
    assert problem.model_id == "red_cube"
    assert "{colour}, {size}" in problem.message
    assert problem.hint is not None
    assert "material_notes" in problem.hint


@pytest.mark.parametrize("template", ["{base_prompt", "{0}", "{base_prompt!x}", "{a.b}"])
def test_malformed_or_indexed_template_is_e_template_placeholder(template: str) -> None:
    with pytest.raises(ValidationError) as excinfo:
        compose_prompt(STYLE, make_doc(), PromptSpec(template=template))

    assert excinfo.value.problems[0].code == "E_TEMPLATE_PLACEHOLDER"


def test_escaped_braces_are_literal() -> None:
    cp = compose_prompt(STYLE, make_doc(), PromptSpec(template="{{x}} {model_prompt}"))

    assert cp.text == "{x} A red cube."


def test_warning_above_600_chars() -> None:
    spec = PromptSpec(template="{model_prompt}")
    assert prompt_warnings(compose_prompt(STYLE, make_doc("w" * PROMPT_WARN_CHARS), spec)) == []

    cp = compose_prompt(STYLE, make_doc("w" * (PROMPT_WARN_CHARS + 1)), spec)

    [warning] = prompt_warnings(cp, model_id="red_cube")
    assert (warning.code, warning.severity) == ("W_PROMPT_LONG", "warning")
    assert warning.model_id == "red_cube"


def test_error_above_1000_chars() -> None:
    spec = PromptSpec(template="{model_prompt}")
    compose_prompt(STYLE, make_doc("w" * PROMPT_MAX_CHARS), spec)  # exactly the limit is fine

    with pytest.raises(PromptTooLongError) as excinfo:
        compose_prompt(STYLE, make_doc("w" * (PROMPT_MAX_CHARS + 1)), spec)

    assert excinfo.value.code == "E_PROMPT_TOO_LONG"
    assert excinfo.value.exit_code == 3
    [problem] = excinfo.value.problems
    assert problem.code == "E_PROMPT_TOO_LONG"
    assert "1001" in problem.message


def test_length_measured_after_normalisation() -> None:
    padded = ("w" * 499) + "    \n\n\n\n    " + ("w" * 499)  # 1 003 raw, 1 000 normalised

    cp = compose_prompt(STYLE, make_doc(padded), PromptSpec(template="{model_prompt}"))

    assert cp.length == PROMPT_MAX_CHARS  # 499 + "\n\n" + 499


def test_sha_is_stable_and_matches_text() -> None:
    a = compose_prompt(STYLE, make_doc(), PromptSpec())
    b = compose_prompt(STYLE, make_doc(), PromptSpec())

    assert a.sha256 == b.sha256 == hashlib.sha256(a.text.encode("utf-8")).hexdigest()
    assert compose_prompt(STYLE, make_doc("A blue cube."), PromptSpec()).sha256 != a.sha256


def test_compose_with_warnings_bundles_prompt_and_warnings() -> None:
    spec = PromptSpec(template="{model_prompt}")

    cp, warnings = compose_with_warnings(STYLE, make_doc("w" * 700), spec)

    assert cp.length == 700
    assert [w.code for w in warnings] == ["W_PROMPT_LONG"]
    assert warnings[0].model_id == "red_cube"


# --------------------------------------------------------------------------- properties
text = st.text(max_size=300)
template = st.sampled_from(
    [
        "{base_prompt}\n\n{model_prompt}",
        "{model_prompt}\n\n\n{material_notes}\n\n{palette}",
        "  {display_name}: {model_prompt}  {tags}  ",
        "\n\n\n{model_prompt}\n\n\n",
    ]
)


@settings(max_examples=200, deadline=None)
@given(prompt=text, base=text, notes=text, tpl=template)
def test_output_is_normalised_for_any_input(prompt: str, base: str, notes: str, tpl: str) -> None:
    style = StyleDocument(
        folder=Path("/s"),
        front=StyleFrontMatter(name="s"),
        sections={"base_prompt": base, "material_notes": notes, "palette": ""},
    )
    spec = PromptSpec(include_sections=["base_prompt", "material_notes", "palette"], template=tpl)

    cp = compose_prompt(style, make_doc(prompt, tags=["a", "b"]), spec)

    assert "\n\n\n" not in cp.text
    assert "  " not in cp.text
    assert not re.search(r"[ \t]\n|\n[ \t]", cp.text)
    assert cp.text == cp.text.strip()
    assert cp.length == len(cp.text) <= PROMPT_MAX_CHARS
    assert cp.sha256 == hashlib.sha256(cp.text.encode("utf-8")).hexdigest()


@settings(max_examples=200, deadline=None)
@given(text)
def test_normalize_whitespace_is_idempotent(value: str) -> None:
    once = normalize_whitespace(value)

    assert normalize_whitespace(once) == once
