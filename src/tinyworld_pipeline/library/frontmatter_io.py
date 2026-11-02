"""Parse ``model.md`` and ``style/style.md`` (plan 05 §4-§5, 09 §1 step 1-2).

Front matter is YAML (``python-frontmatter``); the body is Markdown split on ``## `` headings.
Every parse failure becomes a :class:`ValidationError` carrying ``E_FRONTMATTER`` /
``E_UNKNOWN_KEY`` problems whose ``path`` names the file and the offending key.
"""

from __future__ import annotations

import difflib
import re
from pathlib import Path
from typing import Any, get_args

import frontmatter
import yaml
from pydantic import BaseModel
from pydantic import ValidationError as PydanticValidationError

from tinyworld_pipeline.domain.errors import Problem, ValidationError
from tinyworld_pipeline.domain.model_spec import ModelDocument, ModelFrontMatter
from tinyworld_pipeline.domain.style_spec import StyleDocument, StyleFrontMatter
from tinyworld_pipeline.library.paths import MODEL_MD, STYLE_MD

E_FRONTMATTER = "E_FRONTMATTER"
E_UNKNOWN_KEY = "E_UNKNOWN_KEY"

_HEADING_RE = re.compile(r"^##[ \t]+(?P<title>.+?)[ \t]*$")
_CLOSEST_CUTOFF = 0.6


# --------------------------------------------------------------------------- sections
def split_sections(body: str) -> dict[str, str]:
    """Split a Markdown body on ``## `` headings.

    Keys are the heading text lower-cased and stripped (headings match case-insensitively);
    values are the section bodies with surrounding whitespace removed. ``\\r\\n`` line endings are
    normalised first. Text before the first heading is ignored; a repeated heading overwrites the
    earlier one.

    Args:
        body: Markdown text (everything after the front matter).

    Returns:
        Mapping of lower-cased heading → stripped section body.
    """
    sections: dict[str, str] = {}
    current: str | None = None
    buffer: list[str] = []
    for line in body.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        match = _HEADING_RE.match(line)
        if match:
            if current is not None:
                sections[current] = "\n".join(buffer).strip()
            current = match.group("title").strip().lower()
            buffer = []
        elif current is not None:
            buffer.append(line)
    if current is not None:
        sections[current] = "\n".join(buffer).strip()
    return sections


# --------------------------------------------------------------------------- pydantic → problems
def _nested_model(annotation: Any) -> type[BaseModel] | None:  # noqa: ANN401  # pydantic annotation
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation
    for arg in get_args(annotation):
        found = _nested_model(arg)
        if found is not None:
            return found
    return None


def _known_keys_at(root: type[BaseModel], loc: tuple[str, ...]) -> list[str]:
    """Field names of the (nested) model that owns the key at ``loc``; empty if unresolvable."""
    model: type[BaseModel] | None = root
    for part in loc:
        if model is None or part not in model.model_fields:
            return []
        model = _nested_model(model.model_fields[part].annotation)
    return list(model.model_fields) if model is not None else []


def _unknown_key_hint(root: type[BaseModel], loc: tuple[str, ...]) -> str | None:
    *prefix, key = loc
    known = _known_keys_at(root, tuple(prefix))
    close = difflib.get_close_matches(key, known, n=1, cutoff=_CLOSEST_CUTOFF)
    if close:
        return f"did you mean {close[0]!r}?"
    if known:
        return "known keys: " + ", ".join(sorted(known))
    return None


def problems_from_pydantic(
    exc: PydanticValidationError,
    *,
    root: type[BaseModel],
    file_name: str,
    model_id: str | None,
) -> list[Problem]:
    """Translate a pydantic error into ``E_UNKNOWN_KEY`` / ``E_FRONTMATTER`` problems.

    Args:
        exc: The pydantic validation error.
        root: Top-level front-matter model (to compute "did you mean" hints).
        file_name: ``model.md`` or ``style.md`` (used as the ``path`` prefix).
        model_id: Owning model, if any.

    Returns:
        One problem per pydantic error, in pydantic's order.
    """
    problems: list[Problem] = []
    for err in exc.errors():
        loc = tuple(str(part) for part in err["loc"])
        key_path = ".".join(loc)
        path = f"{file_name}:{key_path}" if key_path else file_name
        if err["type"] == "extra_forbidden":
            problems.append(
                Problem(
                    code=E_UNKNOWN_KEY,
                    severity="error",
                    model_id=model_id,
                    path=path,
                    message=f"unknown key {key_path!r}",
                    hint=_unknown_key_hint(root, loc),
                )
            )
        else:
            problems.append(
                Problem(
                    code=E_FRONTMATTER,
                    severity="error",
                    model_id=model_id,
                    path=path,
                    message=err["msg"],
                )
            )
    return problems


# --------------------------------------------------------------------------- generic loader
def _problem(
    file_name: str, model_id: str | None, message: str, hint: str | None = None
) -> Problem:
    return Problem(
        code=E_FRONTMATTER,
        severity="error",
        model_id=model_id,
        path=file_name,
        message=message,
        hint=hint,
    )


def _load_document[FrontT: BaseModel](
    path: Path, *, root: type[FrontT], file_name: str, model_id: str | None
) -> tuple[FrontT, dict[str, str]]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValidationError([_problem(file_name, model_id, f"cannot read: {exc}")]) from exc
    try:
        post = frontmatter.loads(text.replace("\r\n", "\n"))
    except yaml.YAMLError as exc:
        raise ValidationError(
            [_problem(file_name, model_id, f"YAML front matter is invalid: {exc}")]
        ) from exc
    metadata: object = post.metadata
    if not metadata:
        raise ValidationError(
            [
                _problem(
                    file_name,
                    model_id,
                    "missing YAML front matter",
                    "start the file with a '---' block containing at least display_name/name",
                )
            ]
        )
    try:
        front = root.model_validate(metadata)
    except PydanticValidationError as exc:
        raise ValidationError(
            problems_from_pydantic(exc, root=root, file_name=file_name, model_id=model_id)
        ) from exc
    return front, split_sections(post.content)


# --------------------------------------------------------------------------- public API
def parse_model_md(path: Path, model_id: str) -> ModelDocument:
    """Parse ``models/<model_id>/model.md``.

    Args:
        path: The ``model.md`` file.
        model_id: Folder name; MUST already match the ``model_id`` regex.

    Returns:
        The parsed document (front matter + lower-cased ``## `` sections).

    Raises:
        ValidationError: ``E_FRONTMATTER`` (unreadable, bad YAML, pydantic error) or
            ``E_UNKNOWN_KEY`` (``extra="forbid"`` violation) problems.
    """
    front, sections = _load_document(
        path, root=ModelFrontMatter, file_name=MODEL_MD, model_id=model_id
    )
    return ModelDocument(
        model_id=model_id, folder=path.parent.resolve(), front=front, sections=sections
    )


def parse_style_md(path: Path) -> StyleDocument:
    """Parse ``style/style.md``.

    Args:
        path: The ``style.md`` file.

    Returns:
        The parsed style document. Presence of ``## base_prompt`` is enforced by library
        validation (``E_STYLE_BASE_PROMPT_MISSING``), not here.

    Raises:
        ValidationError: ``E_FRONTMATTER`` / ``E_UNKNOWN_KEY`` problems (``model_id`` is ``None``).
    """
    front, sections = _load_document(path, root=StyleFrontMatter, file_name=STYLE_MD, model_id=None)
    return StyleDocument(folder=path.parent.resolve(), front=front, sections=sections)
