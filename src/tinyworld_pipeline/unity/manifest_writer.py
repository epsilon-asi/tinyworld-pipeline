"""Build and write the Unity delivery ``manifest.json`` (plan 12 §2, appendix C §3-§4, §8).

The manifest is the only contract between the Python pipeline and the C# importer
(``unity/com.studio.tinyworld``). Everything here is pure except :func:`write_manifest`.

This module deliberately imports only from ``domain``: the handoff stage adapts
``textures.urp_pack.UrpTextureSet`` and ``blender.protocol.RigFingerprint`` into the plain
inputs below (:class:`UrpMaterialInputs`, :func:`first_root_bone`), so ``unity`` stays a leaf
sibling of those packages in the import-linter layer graph.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Protocol

from tinyworld_pipeline.domain.errors import HandoffError
from tinyworld_pipeline.domain.library_config import UnitySection
from tinyworld_pipeline.domain.manifest import (
    Manifest,
    ManifestAnimation,
    ManifestColor,
    ManifestFloat,
    ManifestMaterial,
    ManifestProvenance,
    ManifestRig,
    ManifestTexture,
    json_utility_violations,
)

ANIMATION_TYPE = "Generic"
WORKFLOW_MODE_METALLIC = 1.0
SURFACE_OPAQUE = 0.0
SMOOTHNESS_WITH_MAP = 1.0
SMOOTHNESS_CHANNEL_METALLIC_ALPHA = 0.0

# map name (as in ``UrpTextureSet.maps()``) -> (URP slot, srgb, kind, keyword)
_MAP_TABLE: dict[str, tuple[str, bool, str, str | None]] = {
    "basecolor": ("_BaseMap", True, "default", None),
    "normal": ("_BumpMap", False, "normal", "_NORMALMAP"),
    "metallic_smoothness": ("_MetallicGlossMap", False, "default", "_METALLICSPECGLOSSMAP"),
    "occlusion": ("_OcclusionMap", False, "default", "_OCCLUSIONMAP"),
    "emission": ("_EmissionMap", True, "default", "_EMISSION"),
}
MAP_NAMES: tuple[str, ...] = tuple(_MAP_TABLE)


class BoneLike(Protocol):
    """Structural view of ``blender.protocol.Bone`` (name + parent); see :func:`first_root_bone`."""

    @property
    def name(self) -> str: ...  # pragma: no cover - protocol

    @property
    def parent(self) -> str | None: ...  # pragma: no cover - protocol


class UrpScalarsLike(Protocol):
    """Structural view of ``textures.urp_pack.UrpTextureSet``'s scalar fields."""

    @property
    def bump_scale(self) -> float: ...  # pragma: no cover - protocol

    @property
    def occlusion_strength(self) -> float: ...  # pragma: no cover - protocol

    @property
    def emission_color(self) -> tuple[float, float, float]: ...  # pragma: no cover - protocol

    @property
    def metallic_fallback(self) -> float: ...  # pragma: no cover - protocol

    @property
    def smoothness_fallback(self) -> float: ...  # pragma: no cover - protocol


@dataclass(frozen=True)
class UrpMaterialInputs:
    """Scalar material properties the manifest needs besides the texture files.

    Mirrors the scalar half of ``UrpTextureSet`` without importing it.
    """

    bump_scale: float = 1.0
    occlusion_strength: float = 1.0
    emission_color: tuple[float, float, float] = (0.0, 0.0, 0.0)
    metallic_fallback: float = 0.0
    smoothness_fallback: float = 0.5

    @classmethod
    def from_texture_set(cls, urp: UrpScalarsLike) -> UrpMaterialInputs:
        """Copy the scalars out of a packed URP texture set (structural, no import)."""
        r, g, b = urp.emission_color
        return cls(
            bump_scale=urp.bump_scale,
            occlusion_strength=urp.occlusion_strength,
            emission_color=(r, g, b),
            metallic_fallback=urp.metallic_fallback,
            smoothness_fallback=urp.smoothness_fallback,
        )


def first_root_bone(bones: Iterable[BoneLike]) -> str:
    """Name of the first bone without a parent, in the fingerprint's depth-first order.

    Raises:
        HandoffError: The rig has no root bone.
    """
    for bone in bones:
        if bone.parent is None:
            return bone.name
    raise HandoffError("rig fingerprint has no root bone (every bone has a parent)")


def build_manifest(
    *,
    model_id: str,
    display_name: str,
    pipeline_version: str,
    generated_at: str,
    fbx_name: str,
    root_bone: str,
    unity: UnitySection,
    source_asset_path: str | None,
    material: UrpMaterialInputs,
    texture_files: Mapping[str, str],
    animations: Sequence[tuple[str, str]],
    provenance: ManifestProvenance,
) -> Manifest:
    """Assemble the manifest for one delivered model (plan 12 §2, appendix C §3).

    Args:
        model_id: Library folder name; also the material name prefix (``<model_id>_mat``).
        display_name: Human name from ``model.md``.
        pipeline_version: ``tinyworld_pipeline.__version__``.
        generated_at: ISO-8601 UTC timestamp.
        fbx_name: Delivered FBX file name relative to the manifest (``"<id>.fbx"``).
        root_bone: First root bone of the rig (see :func:`first_root_bone`).
        unity: Effective ``[unity]`` configuration (shader, avatar mode).
        source_asset_path: Unity asset path of the vendor FBX whose Avatar is copied, or
            ``None`` when unknown (serialised as ``""`` for JsonUtility).
        material: Scalar material properties.
        texture_files: Map name (``basecolor``, ``normal``, ``metallic_smoothness``,
            ``occlusion``, ``emission``) to file path relative to the manifest, for the maps
            that exist. Keywords, floats and colours are derived from which maps are present.
        animations: ``(file, clip_name)`` pairs, files relative to the manifest.
        provenance: Filled provenance block.

    Returns:
        A validated, JsonUtility-compatible :class:`Manifest`.

    Raises:
        HandoffError: Unknown map name, or a referenced path is absolute / escapes the folder.
    """
    present = {name: _check_relpath(name, path) for name, path in texture_files.items()}
    unknown = sorted(set(present) - set(_MAP_TABLE))
    if unknown:
        raise HandoffError(f"unknown texture map name(s) {unknown}; expected one of {MAP_NAMES}")
    textures = [
        ManifestTexture(
            slot=_MAP_TABLE[name][0],
            file=present[name],
            srgb=_MAP_TABLE[name][1],
            kind=_MAP_TABLE[name][2],
        )
        for name in MAP_NAMES
        if name in present
    ]
    keywords = [kw for name in MAP_NAMES if name in present and (kw := _MAP_TABLE[name][3])]
    mat = ManifestMaterial(
        name=f"{model_id}_mat",
        shader=unity.shader,
        renderers=[],
        textures=textures,
        floats=_floats(material, present),
        colors=_colors(material, present),
        keywords=keywords,
    )
    return Manifest(
        pipeline_version=pipeline_version,
        generated_at=generated_at,
        model_id=model_id,
        display_name=display_name,
        fbx=_check_relpath("fbx", fbx_name),
        rig=ManifestRig(
            animation_type=ANIMATION_TYPE,
            avatar_mode=unity.avatar_mode,
            source_avatar_asset_path=source_asset_path or "",
            root_bone=root_bone,
        ),
        materials=[mat],
        animations=[
            ManifestAnimation(file=_check_relpath("animation", file), clip_name=clip)
            for file, clip in animations
        ],
        provenance=provenance,
    )


def write_manifest(manifest: Manifest, path: Path) -> Path:
    """Write ``manifest`` as JSON atomically (temp sibling + ``os.replace``).

    Field order follows the schema (``sort_keys=False``) so the file reads top-down like plan
    06 §10; JsonUtility does not care about order.

    Raises:
        HandoffError: The document breaks a JsonUtility rule (``null``, mixed arrays).
    """
    doc = manifest.model_dump(mode="json")
    violations = json_utility_violations(doc)
    if violations:
        raise HandoffError(
            "manifest is not JsonUtility-compatible: " + "; ".join(violations),
            detail={"violations": violations},
        )
    text = json.dumps(doc, indent=2, sort_keys=False, ensure_ascii=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
    return path


def manifest_relpaths(manifest: Manifest) -> list[str]:
    """Every file the manifest references, relative to it, in document order without duplicates."""
    seen: dict[str, None] = {manifest.fbx: None}
    for mat in manifest.materials:
        for tex in mat.textures:
            seen.setdefault(tex.file, None)
    for anim in manifest.animations:
        seen.setdefault(anim.file, None)
    return list(seen)


# ------------------------------------------------------------------------------------ helpers
def _floats(material: UrpMaterialInputs, present: Mapping[str, str]) -> list[ManifestFloat]:
    floats = [
        ManifestFloat(name="_WorkflowMode", value=WORKFLOW_MODE_METALLIC),
        ManifestFloat(name="_Surface", value=SURFACE_OPAQUE),
    ]
    if "metallic_smoothness" in present:
        # The map's alpha carries smoothness; _Smoothness becomes a multiplier.
        floats.append(ManifestFloat(name="_Smoothness", value=SMOOTHNESS_WITH_MAP))
        floats.append(
            ManifestFloat(name="_SmoothnessTextureChannel", value=SMOOTHNESS_CHANNEL_METALLIC_ALPHA)
        )
    else:
        floats.append(ManifestFloat(name="_Metallic", value=material.metallic_fallback))
        floats.append(ManifestFloat(name="_Smoothness", value=material.smoothness_fallback))
    if "normal" in present:
        floats.append(ManifestFloat(name="_BumpScale", value=material.bump_scale))
    if "occlusion" in present:
        floats.append(ManifestFloat(name="_OcclusionStrength", value=material.occlusion_strength))
    return floats


def _colors(material: UrpMaterialInputs, present: Mapping[str, str]) -> list[ManifestColor]:
    colors = [ManifestColor(name="_BaseColor", r=1.0, g=1.0, b=1.0, a=1.0)]
    if "emission" in present:
        r, g, b = material.emission_color
        colors.append(ManifestColor(name="_EmissionColor", r=r, g=g, b=b, a=1.0))
    return colors


def _check_relpath(what: str, value: str) -> str:
    """Reject absolute, backslash or parent-escaping paths; manifests are relative + POSIX."""
    posix = PurePosixPath(value)
    if not value or "\\" in value or posix.is_absolute() or ".." in posix.parts:
        raise HandoffError(
            f"{what} path {value!r} must be a relative forward-slash path inside the output folder"
        )
    return value


__all__ = [
    "ANIMATION_TYPE",
    "MAP_NAMES",
    "BoneLike",
    "UrpMaterialInputs",
    "UrpScalarsLike",
    "build_manifest",
    "first_root_bone",
    "manifest_relpaths",
    "write_manifest",
]
