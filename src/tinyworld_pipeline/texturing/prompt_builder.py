"""Prompt composition (plan 10 §6, 05 §5).

Pure: no I/O. :func:`compose_prompt` fills the model's ``prompt.template`` with the style sections,
the model's ``## prompt`` body, display name and tags; normalises whitespace; enforces the length
rule; records provenance.
"""

from __future__ import annotations

import hashlib
import re
from string import Formatter

from tinyworld_pipeline.domain.errors import Problem, PromptTooLongError, ValidationError
from tinyworld_pipeline.domain.model_spec import ModelDocument, PromptSpec
from tinyworld_pipeline.domain.prompt import PROMPT_MAX_CHARS, PROMPT_WARN_CHARS, ComposedPrompt
from tinyworld_pipeline.domain.style_spec import StyleDocument

E_TEMPLATE_PLACEHOLDER = "E_TEMPLATE_PLACEHOLDER"
E_PROMPT_TOO_LONG = "E_PROMPT_TOO_LONG"
W_PROMPT_LONG = "W_PROMPT_LONG"

MODEL_PROMPT_SECTION = "prompt"
BUILTIN_PLACEHOLDERS: tuple[str, ...] = ("model_prompt", "display_name", "tags")

_SPACES_RE = re.compile(r"[ \t\f\v]+")
_BLANK_RUN_RE = re.compile(r"\n{3,}")
_TEMPLATE_PATH = "model.md:prompt.template"


def _placeholders(template: str) -> list[str]:
    """Field names referenced by ``template`` (``str.format`` syntax), in order."""
    names: list[str] = []
    for _literal, field, _spec, _conv in Formatter().parse(template):
        if field is not None:
            names.append(field)
    return names


def _template_problem(doc: ModelDocument, message: str, hint: str) -> ValidationError:
    return ValidationError(
        [
            Problem(
                code=E_TEMPLATE_PLACEHOLDER,
                severity="error",
                model_id=doc.model_id,
                path=_TEMPLATE_PATH,
                message=message,
                hint=hint,
            )
        ]
    )


def normalize_whitespace(text: str) -> str:
    """Collapse runs of blanks to one space, strip each line, collapse 3+ newlines to 2, strip."""
    unified = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [_SPACES_RE.sub(" ", line).strip() for line in unified.split("\n")]
    return _BLANK_RUN_RE.sub("\n\n", "\n".join(lines)).strip()


def compose_prompt(style: StyleDocument, doc: ModelDocument, spec: PromptSpec) -> ComposedPrompt:
    """Compose the provider prompt for one model.

    Placeholders: every ``style.md`` section name, ``model_prompt`` (the ``## prompt`` body),
    ``display_name`` and ``tags`` (comma-joined). Style sections not listed in
    ``spec.include_sections`` resolve to ``""`` so templates stay stable. ``sources`` maps each
    placeholder used by the template to where its text came from (``""`` when excluded/empty).

    Args:
        style: Parsed ``style/style.md``.
        doc: Parsed ``model.md``.
        spec: The model's ``prompt`` front-matter section.

    Returns:
        The composed prompt. Use :func:`prompt_warnings` for the > 600-character warning.

    Raises:
        ValidationError: ``E_TEMPLATE_PLACEHOLDER`` — unknown ``{placeholder}`` or malformed
            template.
        PromptTooLongError: ``E_PROMPT_TOO_LONG`` — more than 1 000 characters after templating.
    """
    included = set(spec.include_sections)
    values: dict[str, str] = {}
    sources: dict[str, str] = {}
    for name, body in style.sections.items():
        used = name in included
        values[name] = body if used else ""
        sources[name] = f"style.md:{name}" if used else ""
    for name in included - set(style.sections):
        values[name] = ""  # listed but absent from style.md: empty, never an error here
        sources[name] = ""
    model_prompt = doc.sections.get(MODEL_PROMPT_SECTION, "")
    values["model_prompt"] = model_prompt
    sources["model_prompt"] = "model.md:prompt" if model_prompt else ""
    values["display_name"] = doc.front.display_name
    sources["display_name"] = "model.md:display_name"
    values["tags"] = ", ".join(doc.front.tags)
    sources["tags"] = "model.md:tags" if doc.front.tags else ""

    try:
        requested = _placeholders(spec.template)
    except ValueError as exc:
        raise _template_problem(
            doc, f"malformed template: {exc}", "escape literal braces as '{{' and '}}'"
        ) from exc
    unknown = [name for name in requested if name not in values]
    if unknown:
        known = ", ".join(sorted(values))
        raise _template_problem(
            doc,
            "unknown placeholder(s): " + ", ".join(f"{{{n}}}" for n in unknown),
            f"available placeholders: {known}",
        )

    try:
        rendered = spec.template.format_map(values)
    except ValueError as exc:  # bad conversion/format spec, e.g. "{base_prompt!x}"
        raise _template_problem(
            doc, f"malformed template: {exc}", "use plain {placeholder} fields"
        ) from exc
    text = normalize_whitespace(rendered)
    if len(text) > PROMPT_MAX_CHARS:
        raise PromptTooLongError(
            [
                Problem(
                    code=E_PROMPT_TOO_LONG,
                    severity="error",
                    model_id=doc.model_id,
                    path="model.md:prompt",
                    message=f"composed prompt is {len(text)} characters (max {PROMPT_MAX_CHARS})",
                    hint="shorten ## prompt or drop sections from prompt.include_sections",
                )
            ]
        )
    return ComposedPrompt(
        text=text,
        sources={name: sources[name] for name in requested},
        length=len(text),
        sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )


def prompt_warnings(prompt: ComposedPrompt, *, model_id: str | None = None) -> list[Problem]:
    """Non-fatal findings about a composed prompt (``W_PROMPT_LONG`` above 600 characters)."""
    if prompt.length <= PROMPT_WARN_CHARS:
        return []
    return [
        Problem(
            code=W_PROMPT_LONG,
            severity="warning",
            model_id=model_id,
            path="model.md:prompt",
            message=(
                f"composed prompt is {prompt.length} characters (> {PROMPT_WARN_CHARS}); "
                "long prompts dilute style guidance"
            ),
            hint="consider trimming ## prompt",
        )
    ]


def compose_with_warnings(
    style: StyleDocument, doc: ModelDocument, spec: PromptSpec
) -> tuple[ComposedPrompt, list[Problem]]:
    """:func:`compose_prompt` plus :func:`prompt_warnings` in one call.

    This is the ``PromptComposer`` shape that ``library.validation`` / ``library.resolve`` accept
    by injection (the library layer may not import ``texturing``).
    """
    composed = compose_prompt(style, doc, spec)
    return composed, prompt_warnings(composed, model_id=doc.model_id)
