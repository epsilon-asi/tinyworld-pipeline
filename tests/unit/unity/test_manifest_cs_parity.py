"""06 §10: the C# JsonUtility DTOs must mirror the pydantic manifest models field for field."""

from __future__ import annotations

import re
import types
from pathlib import Path
from typing import Any, get_args, get_origin

import pytest
from pydantic import BaseModel

from grounded_pipeline.domain import manifest as manifest_module
from grounded_pipeline.domain.manifest import (
    Manifest,
    ManifestAnimation,
    ManifestColor,
    ManifestFloat,
    ManifestMaterial,
    ManifestProvenance,
    ManifestRig,
    ManifestTexture,
)

ROOT = Path(__file__).resolve().parents[3]
CS_FILE = ROOT / "unity" / "com.studio.grounded-pipeline" / "Editor" / "PipelineManifest.cs"

PY_TO_CS: dict[str, type[BaseModel]] = {
    "PipelineManifest": Manifest,
    "ManifestRig": ManifestRig,
    "ManifestMaterial": ManifestMaterial,
    "ManifestTexture": ManifestTexture,
    "ManifestFloat": ManifestFloat,
    "ManifestColor": ManifestColor,
    "ManifestAnimation": ManifestAnimation,
    "ManifestProvenance": ManifestProvenance,
}

CLASS_RE = re.compile(r"\[Serializable\]\s*public\s+(?:sealed\s+)?class\s+(\w+)\s*\{")
FIELD_RE = re.compile(
    r"^\s*public\s+(?!static\b|const\b)([\w<>\[\],\.]+)\s+(\w+)\s*;", re.MULTILINE
)
PRIMITIVES = {bool: "bool", float: "float", int: "int", str: "string"}


def class_bodies(source: str) -> dict[str, str]:
    """Body (between the outermost braces) of every ``[Serializable] public class``."""
    bodies: dict[str, str] = {}
    for match in CLASS_RE.finditer(source):
        depth = 1
        index = match.end()
        while depth and index < len(source):
            depth += {"{": 1, "}": -1}.get(source[index], 0)
            index += 1
        bodies[match.group(1)] = source[match.end() : index - 1]
    return bodies


def cs_fields(body: str) -> dict[str, str]:
    return {name: ctype for ctype, name in FIELD_RE.findall(body)}


def expected_cs_type(annotation: Any) -> set[str]:
    """Acceptable C# spellings for a pydantic annotation."""
    origin = get_origin(annotation)
    if origin is list:
        (item,) = get_args(annotation)
        inner = expected_cs_type(item)
        return {f"{t}[]" for t in inner} | {f"List<{t}>" for t in inner}
    if origin is types.UnionType:  # pragma: no cover - manifest declares no unions
        pytest.fail(f"manifest field uses a union {annotation}; JsonUtility cannot read it")
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return {annotation.__name__}
    return {PRIMITIVES[annotation]}


@pytest.fixture(scope="module")
def bodies() -> dict[str, str]:
    return class_bodies(CS_FILE.read_text(encoding="utf-8"))


def test_every_pydantic_manifest_model_has_a_cs_class(bodies: dict[str, str]) -> None:
    assert set(bodies) == set(PY_TO_CS)
    pydantic_models = {
        obj.__name__
        for obj in vars(manifest_module).values()
        if isinstance(obj, type)
        and issubclass(obj, BaseModel)
        and obj.__module__ == manifest_module.__name__
    }
    assert pydantic_models == {m.__name__ for m in PY_TO_CS.values()}


@pytest.mark.parametrize("cs_name", sorted(PY_TO_CS))
def test_field_names_and_types_match(cs_name: str, bodies: dict[str, str]) -> None:
    model = PY_TO_CS[cs_name]
    fields = cs_fields(bodies[cs_name])
    assert list(fields) == list(model.model_fields), f"{cs_name}: field names/order differ"
    for name, info in model.model_fields.items():
        allowed = expected_cs_type(info.annotation)
        assert fields[name] in allowed, f"{cs_name}.{name}: C# {fields[name]} not in {allowed}"


def test_cs_classes_use_public_fields_only_no_properties(bodies: dict[str, str]) -> None:
    for name, body in bodies.items():
        for line in body.splitlines():
            stripped = line.strip()
            if stripped.startswith("public ") and "{ get" in stripped:
                pytest.fail(f"{name}: JsonUtility ignores properties: {stripped}")
            assert (
                not stripped.startswith(("private ", "protected ", "internal ")) or "(" in stripped
            ), f"{name}: non-public field would be skipped by JsonUtility: {stripped}"


def test_supported_schema_version_matches_python() -> None:
    source = CS_FILE.read_text(encoding="utf-8")
    match = re.search(r"SupportedSchemaVersion\s*=\s*(\d+);", source)
    assert match is not None
    assert int(match.group(1)) == manifest_module.MANIFEST_SCHEMA_VERSION
    assert "JsonUtility.FromJson<PipelineManifest>" in source
