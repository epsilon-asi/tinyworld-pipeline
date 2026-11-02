"""Turn a model folder into a :class:`ResolvedModel` and compute the ``discover`` fingerprint.

This is the library-side body of stage ``discover`` (plan 09 §1 steps 1-6). The
``DiscoverStage`` wrapper (state handling, step 7) lives in ``pipeline/stages/discover.py``
(WP-10) and calls these two functions.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path

from tinyworld_pipeline.config.layering import ENV_PREFIX, CliOverrides, resolve_config
from tinyworld_pipeline.domain.errors import Problem, ValidationError
from tinyworld_pipeline.domain.ids import Sha256
from tinyworld_pipeline.domain.model_spec import ResolvedModel
from tinyworld_pipeline.library.discovery import Library, find_animation_fbx, find_primary_fbx
from tinyworld_pipeline.library.validation import raise_if_errors, validate_model
from tinyworld_pipeline.texturing.prompt_builder import compose_with_warnings


def resolve_model(
    lib: Library,
    model_id: str,
    *,
    env: Mapping[str, str],
    cli: CliOverrides,
) -> tuple[ResolvedModel, list[Problem]]:
    """Parse, layer, resolve paths and compose the prompt for one model.

    The model is validated first (library-level load problems + every per-model rule); a
    ``ResolvedModel`` is therefore always a valid one.

    Args:
        lib: Loaded library.
        model_id: Model folder name.
        env: Process environment (``GP_*`` config overrides).
        cli: CLI overrides.

    Returns:
        ``(resolved model, warnings)`` — warnings are the non-error problems found on the way.

    Raises:
        ValidationError: Any error-severity problem (library-level or per-model).
        UsageError: Unknown ``model_id``.
    """
    raise_if_errors(lib.load_problems)
    problems = validate_model(lib, model_id, env=env, cli=cli)
    raise_if_errors(problems)
    style = lib.style
    if style is None:  # pragma: no cover  # a missing/broken style.md is always an error above
        raise ValidationError(lib.load_problems, message="style/style.md could not be loaded")

    doc = lib.model_document(model_id)
    config, _ = resolve_config(lib.config, doc.front, env, cli)
    primary, p_problems = find_primary_fbx(doc)
    if primary is None:  # pragma: no cover  # validate_model already failed on E_SOURCE_*
        raise ValidationError(p_problems)
    animations, _ = find_animation_fbx(doc, primary=primary)
    composed, _ = compose_with_warnings(style, doc, doc.front.prompt)
    resolved = ResolvedModel(
        doc=doc,
        config=config,
        primary_fbx=primary,
        animation_fbx=animations,
        style=style,
        composed_prompt=composed,
    )
    return resolved, [p for p in problems if p.severity == "warning"]


# --------------------------------------------------------------------------- fingerprint
def _read_bytes(path: Path) -> bytes:
    return path.read_bytes() if path.is_file() else b""


def overrides_dump(env: Mapping[str, str], cli: CliOverrides) -> str:
    """Canonical JSON of the ``GP_*`` environment subset and the CLI overrides."""
    gp_env = {k: v for k, v in env.items() if k.startswith(ENV_PREFIX)}
    return json.dumps(
        {"env": gp_env, "cli": cli.model_dump(mode="json")}, sort_keys=True, ensure_ascii=False
    )


def _digest(parts: Iterable[tuple[str, bytes]]) -> Sha256:
    h = hashlib.sha256()
    for label, blob in parts:  # length-prefixed so concatenations cannot collide
        h.update(label.encode("utf-8"))
        h.update(b"\0")
        h.update(str(len(blob)).encode("ascii"))
        h.update(b"\0")
        h.update(blob)
    return h.hexdigest()


def discover_fingerprint(
    lib: Library,
    model_id: str,
    *,
    env: Mapping[str, str],
    cli: CliOverrides,
    pipeline_version: str,
) -> Sha256:
    """Fingerprint of the ``discover`` inputs (plan 09 §1).

    sha256 over ``model.md`` bytes, ``library.toml`` bytes, ``style/style.md`` bytes, the
    env/CLI override dump (:func:`overrides_dump`) and ``pipeline_version``. Missing files hash as
    empty.

    Args:
        lib: Loaded library.
        model_id: Model folder name.
        env: Process environment (``GP_*`` subset is hashed).
        cli: CLI overrides.
        pipeline_version: The package version string.

    Returns:
        Hex sha256.
    """
    model = lib.paths.model(model_id)
    return _digest(
        [
            ("model.md", _read_bytes(model.model_md)),
            ("library.toml", _read_bytes(lib.paths.library_toml)),
            ("style.md", _read_bytes(lib.paths.style_md)),
            ("overrides", overrides_dump(env, cli).encode("utf-8")),
            ("pipeline_version", pipeline_version.encode("utf-8")),
        ]
    )
