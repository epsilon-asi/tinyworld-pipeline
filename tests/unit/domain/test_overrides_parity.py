"""ADR-0003: hand-written ``*Overrides`` classes must mirror their sections exactly."""

from __future__ import annotations

from typing import get_args, get_origin

import pytest
from pydantic import BaseModel

from grounded_pipeline.config.layering import make_partial
from grounded_pipeline.domain import model_spec
from grounded_pipeline.domain.library_config import SECTION_MODELS

OVERRIDE_CLASSES: dict[str, type[BaseModel]] = {
    "geometry": model_spec.GeometryOverrides,
    "uv": model_spec.UvOverrides,
    "texture": model_spec.TextureOverrides,
    "verify": model_spec.VerifyOverrides,
    "bake": model_spec.BakeOverrides,
    "export": model_spec.ExportOverrides,
    "unity": model_spec.UnityOverrides,
}


def _strip_optional(annotation: object) -> object:
    args = [a for a in get_args(annotation) if a is not type(None)]
    return args[0] if get_origin(annotation) is not None and len(args) == 1 else annotation


@pytest.mark.parametrize("section", sorted(OVERRIDE_CLASSES))
def test_override_fields_equal_section_fields_plus_declared_extras(section: str) -> None:
    section_cls = SECTION_MODELS[section]
    override_cls = OVERRIDE_CLASSES[section]
    extras = model_spec.OVERRIDE_EXTRA_FIELDS.get(section, frozenset())

    assert set(override_cls.model_fields) == set(section_cls.model_fields) | extras


@pytest.mark.parametrize("section", sorted(OVERRIDE_CLASSES))
def test_override_fields_are_optional_with_none_default(section: str) -> None:
    override_cls = OVERRIDE_CLASSES[section]
    generated = make_partial(SECTION_MODELS[section], name=f"Gen{section}")

    for name, info in override_cls.model_fields.items():
        assert info.default is None, f"{section}.{name} must default to None"
        if name in generated.model_fields:
            expected = _strip_optional(generated.model_fields[name].annotation)
            actual = _strip_optional(info.annotation)
            assert actual == expected, f"{section}.{name}: {actual} != {expected}"


def test_model_front_matter_exposes_every_overridable_section() -> None:
    fm_fields = set(model_spec.ModelFrontMatter.model_fields)
    assert set(OVERRIDE_CLASSES) <= fm_fields
    # library/blender/tripo/logging are library-wide only, by design (plan 05 §5 lists these seven)
    assert {"library", "blender", "tripo", "logging"}.isdisjoint(fm_fields)


def test_make_partial_produces_all_optional_fields() -> None:
    partial = make_partial(SECTION_MODELS["geometry"], name="P")
    instance = partial()
    assert all(getattr(instance, f) is None for f in partial.model_fields)
    assert partial.model_validate({"subdivision_levels": 2}).model_fields_set == {
        "subdivision_levels"
    }
