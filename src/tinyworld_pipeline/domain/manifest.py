"""``manifest.json`` — JsonUtility-compatible delivery manifest (plan 06 §10, appendix C §8).

Unity's ``JsonUtility`` cannot read dictionaries or nullable primitives, so every field here is a
primitive, a string, a nested object, or a list of objects; ``""`` stands in for "unknown".
"""

from __future__ import annotations

from typing import Any

from tinyworld_pipeline.domain._base import FrozenModel

MANIFEST_SCHEMA_VERSION = 1

TEXTURE_SLOTS = ("_BaseMap", "_BumpMap", "_MetallicGlossMap", "_OcclusionMap", "_EmissionMap")


class ManifestTexture(FrozenModel):
    slot: str  # one of TEXTURE_SLOTS
    file: str  # relative to manifest: "textures/<id>_basecolor.png"
    srgb: bool
    kind: str  # "default" | "normal"


class ManifestFloat(FrozenModel):
    name: str
    value: float


class ManifestColor(FrozenModel):
    name: str
    r: float
    g: float
    b: float
    a: float


class ManifestMaterial(FrozenModel):
    name: str
    shader: str
    renderers: list[str]  # mesh object names to assign this material to; [] = all
    textures: list[ManifestTexture]
    floats: list[ManifestFloat]
    colors: list[ManifestColor]
    keywords: list[str]


class ManifestRig(FrozenModel):
    animation_type: str  # "Generic"
    avatar_mode: str  # "copy_from_source" | "create_from_model"
    source_avatar_asset_path: str  # "" when unknown
    root_bone: str


class ManifestAnimation(FrozenModel):
    file: str
    clip_name: str


class ManifestProvenance(FrozenModel):
    source_fbx_sha256: str
    source_license: str
    subdivision_levels: int
    subdivision_type: str
    uv_decision: str
    texture_provider: str
    texture_task_id: str
    texture_route: str
    texture_seed: int
    prompt_sha256: str
    credits_consumed: float
    rig_fidelity: str  # "pass" | "pass_with_warnings"


class Manifest(FrozenModel):
    schema_version: int = MANIFEST_SCHEMA_VERSION
    pipeline_version: str
    generated_at: str
    model_id: str
    display_name: str
    fbx: str
    rig: ManifestRig
    materials: list[ManifestMaterial]
    animations: list[ManifestAnimation]
    provenance: ManifestProvenance


def json_utility_violations(value: Any, path: str = "$") -> list[str]:  # noqa: ANN401 - walks arbitrary JSON
    """Return human-readable paths where a JSON document breaks ``JsonUtility`` rules.

    Rules: no ``null`` anywhere; no dictionaries except as objects with fixed keys (which we
    cannot distinguish from maps, so the manifest models above never declare ``dict`` fields);
    arrays must be homogeneous in JSON type.
    """
    problems: list[str] = []
    if value is None:
        problems.append(f"{path}: null is not representable")
    elif isinstance(value, dict):
        for key, item in value.items():
            problems.extend(json_utility_violations(item, f"{path}.{key}"))
    elif isinstance(value, list):
        kinds = {type(item).__name__ for item in value}
        if len(kinds) > 1:
            problems.append(f"{path}: heterogeneous array {sorted(kinds)}")
        for index, item in enumerate(value):
            problems.extend(json_utility_violations(item, f"{path}[{index}]"))
    return problems


def manifest_json_schema() -> dict[str, Any]:
    """JSON schema of :class:`Manifest` (committed to ``docs/schemas/manifest.schema.json``)."""
    return Manifest.model_json_schema()


__all__ = [
    "MANIFEST_SCHEMA_VERSION",
    "TEXTURE_SLOTS",
    "Manifest",
    "ManifestAnimation",
    "ManifestColor",
    "ManifestFloat",
    "ManifestMaterial",
    "ManifestProvenance",
    "ManifestRig",
    "ManifestTexture",
    "json_utility_violations",
    "manifest_json_schema",
]
