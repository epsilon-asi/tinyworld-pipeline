"""Configuration layering (plan 07 §1).

Precedence, lowest → highest: built-in defaults < ``library.toml`` < ``model.md`` front matter <
environment (``GP_<SECTION>__<KEY>``) < CLI flags. A higher layer wins only for keys it set
*explicitly* (``model_fields_set``), never for keys that merely have defaults. Lists replace.
"""

from __future__ import annotations

import json
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, get_args, get_origin

from pydantic import BaseModel, create_model
from pydantic import ValidationError as PydanticValidationError

from grounded_pipeline.domain._base import FrozenModel
from grounded_pipeline.domain.errors import ConfigError, Problem
from grounded_pipeline.domain.library_config import SECTION_MODELS, LibraryConfig
from grounded_pipeline.domain.model_spec import OVERRIDE_EXTRA_FIELDS, ModelFrontMatter

ENV_PREFIX = "GP_"
ENV_DELIMITER = "__"

#: ``GP_*`` variables that are not configuration keys and must not trigger the unknown-var warning.
KNOWN_NON_CONFIG_ENV: frozenset[str] = frozenset(
    {
        "GP_LIBRARY",
        "GP_BLENDER_EXE",
        "GP_KEEP_TEMP",
        "GP_SRC_DIR",
        "GP_TRIPO__RECORD_DIR",
        "GP_BLENDER_CYCLES_DEVICE",
    }
)


class CliOverrides(FrozenModel):
    """The documented subset of CLI flags that map onto config keys (plan 08 §3)."""

    blender_executable: Path | None = None  # --blender
    concurrency: int | None = None  # --concurrency
    route: Literal["auto", "extract", "bake"] | None = None  # --route
    max_credits: float | None = None  # --max-credits
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] | None = None  # --log-level

    def as_sections(self) -> dict[str, dict[str, Any]]:
        """Translate flags into ``{section: {key: value}}`` for the layering merge."""
        out: dict[str, dict[str, Any]] = {}
        if self.blender_executable is not None:
            out.setdefault("blender", {})["executable"] = self.blender_executable
        if self.concurrency is not None:
            out.setdefault("library", {})["concurrency"] = self.concurrency
        if self.route is not None:
            out.setdefault("verify", {})["route"] = self.route
        if self.max_credits is not None:
            out.setdefault("tripo", {})["max_credits_per_run"] = self.max_credits
        if self.log_level is not None:
            out.setdefault("logging", {})["level"] = self.log_level
        return out


# --------------------------------------------------------------------------- make_partial
def make_partial(section_cls: type[BaseModel], *, name: str) -> type[BaseModel]:
    """Build a pydantic model with every field of ``section_cls`` made optional (default ``None``).

    Used by the parity test to assert the hand-written ``*Overrides`` classes in
    ``domain/model_spec.py`` match their sections (ADR-0003), and available for ad-hoc partials.
    """
    fields: dict[str, Any] = {}
    for fname, finfo in section_cls.model_fields.items():
        annotation: Any = finfo.annotation
        fields[fname] = (annotation | None, None)
    return create_model(name, __base__=FrozenModel, **fields)


# --------------------------------------------------------------------------- merging
def _explicit_updates(override: BaseModel, *, skip: frozenset[str] = frozenset()) -> dict[str, Any]:
    """Keys the override set explicitly and non-``None`` (``None`` means "not overridden")."""
    return {
        key: getattr(override, key)
        for key in override.model_fields_set
        if key not in skip and getattr(override, key) is not None
    }


def _overlay(base: BaseModel, updates: Mapping[str, Any]) -> BaseModel:
    """Return ``base`` re-validated with ``updates`` applied (frozen models are never mutated)."""
    if not updates:
        return base
    merged = {**base.model_dump(), **dict(updates)}
    return type(base).model_validate(merged)


def _overlay_config(
    config: LibraryConfig, sections: Mapping[str, Mapping[str, Any]]
) -> LibraryConfig:
    changes: dict[str, BaseModel] = {}
    for section, updates in sections.items():
        if section not in SECTION_MODELS:
            msg = f"unknown config section {section!r}"
            raise ConfigError(msg)
        current: BaseModel = getattr(config, section)
        changes[section] = _overlay(current, updates)
    if not changes:
        return config
    return LibraryConfig.model_validate(
        {**config.model_dump(), **{k: v.model_dump() for k, v in changes.items()}}
    )


def toml_layer(library_toml: LibraryConfig) -> dict[str, dict[str, Any]]:
    """Explicitly-set ``{section: {key: value}}`` from a parsed ``library.toml`` model."""
    out: dict[str, dict[str, Any]] = {}
    for section in library_toml.model_fields_set:
        if section not in SECTION_MODELS:
            continue
        sec_model: BaseModel = getattr(library_toml, section)
        updates = {key: getattr(sec_model, key) for key in sec_model.model_fields_set}
        if updates:
            out[section] = updates
    return out


def model_layer(front: ModelFrontMatter) -> dict[str, dict[str, Any]]:
    """Explicitly-set overrides from ``model.md`` front matter (extra fields excluded)."""
    out: dict[str, dict[str, Any]] = {}
    for section in SECTION_MODELS:
        override = getattr(front, section, None)
        if override is None:
            continue
        skip = OVERRIDE_EXTRA_FIELDS.get(section, frozenset())
        updates = _explicit_updates(override, skip=skip)
        if updates:
            out[section] = updates
    return out


# --------------------------------------------------------------------------- environment
def _is_collection(annotation: Any) -> bool:  # noqa: ANN401
    origin = get_origin(annotation)
    if origin in (list, dict, tuple, set):
        return True
    return any(_is_collection(arg) for arg in get_args(annotation)) if origin is not None else False


def env_layer(env: Mapping[str, str]) -> tuple[dict[str, dict[str, Any]], list[Problem]]:
    """Parse ``GP_<SECTION>__<KEY>`` variables. Unknown ones produce warnings, not errors."""
    out: dict[str, dict[str, Any]] = {}
    problems: list[Problem] = []
    for raw_key, raw_value in sorted(env.items()):
        if not raw_key.startswith(ENV_PREFIX) or raw_key in KNOWN_NON_CONFIG_ENV:
            continue
        body = raw_key[len(ENV_PREFIX) :]
        section, sep, key = body.partition(ENV_DELIMITER)
        section, key = section.lower(), key.lower()
        model = SECTION_MODELS.get(section)
        if not sep or model is None or key not in model.model_fields:
            problems.append(
                Problem(
                    code="W_UNKNOWN_ENV",
                    severity="warning",
                    path=raw_key,
                    message=f"environment variable {raw_key} does not match any config key",
                    hint="expected GP_<SECTION>__<KEY>, e.g. GP_TEXTURE__QUALITY=detailed",
                )
            )
            continue
        value: Any = raw_value
        if _is_collection(model.model_fields[key].annotation):
            try:
                value = json.loads(raw_value)
            except json.JSONDecodeError as exc:
                msg = f"{raw_key}: list/dict values must be JSON ({exc.msg})"
                raise ConfigError(
                    msg, hint='e.g. GP_BAKE__MAPS=\'["basecolor","normal"]\''
                ) from exc
        out.setdefault(section, {})[key] = value
    return out, problems


# --------------------------------------------------------------------------- public API
def load_library_toml(path: Path) -> LibraryConfig:
    """Parse ``library.toml``; unknown keys and invalid values raise :class:`ConfigError`."""
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        msg = f"{path}: {exc}"
        raise ConfigError(msg, detail={"code": "E_LIBRARY_TOML"}) from exc
    try:
        return LibraryConfig.model_validate(data)
    except PydanticValidationError as exc:
        raise ConfigError(
            f"{path}: {format_pydantic_error(exc)}", detail={"code": "E_LIBRARY_TOML"}
        ) from exc


def format_pydantic_error(exc: PydanticValidationError) -> str:
    """``key.path: message`` lines, one per error, for humans."""
    lines = []
    for err in exc.errors():
        loc = ".".join(str(part) for part in err["loc"]) or "<root>"
        lines.append(f"{loc}: {err['msg']}")
    return "; ".join(lines)


def resolve_config(
    library_toml: LibraryConfig,
    model_overrides: ModelFrontMatter | None,
    env: Mapping[str, str],
    cli: CliOverrides,
) -> tuple[LibraryConfig, list[Problem]]:
    """Deep-merge in precedence order; returns the effective config and any warnings.

    Partial layers only override keys that were explicitly set, never keys that merely have
    defaults. Validation of the merged result (``ge``/``le``/literals) is re-run at every layer.
    """
    env_sections, problems = env_layer(env)
    config = LibraryConfig()
    try:
        config = _overlay_config(config, toml_layer(library_toml))
        if model_overrides is not None:
            config = _overlay_config(config, model_layer(model_overrides))
        config = _overlay_config(config, env_sections)
        config = _overlay_config(config, cli.as_sections())
    except PydanticValidationError as exc:
        raise ConfigError(format_pydantic_error(exc)) from exc
    return config, problems
