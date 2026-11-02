"""``roundtrip_animation`` job: pass a companion clip through the pipeline's FBX settings.

Implements plan 12 §1.3 (decision D-08): every companion animation file is re-imported with
``FBX_IMPORT_KWARGS`` and re-exported with the same ``fbx_export_kwargs`` as the model, so Unity
sees one consistent skeleton interpretation across files. Meshes that happen to be in the clip
file (some vendors ship a dummy mesh) are kept but never modified. Action names lose their import
prefixes and the pose is reset before the export (appendix B §8.1, §8.3) so the delivered take is
``Armature|walk`` and the armature nodes carry the rest pose.

The returned rig fingerprint is the *imported* vendor rig (what the Rig Fidelity Check compares
against the model's source rig); ``missing_bones`` / ``extra_bones`` are recorded against
``expected_bone_names`` for the orchestrator's report.
"""

from __future__ import annotations

from typing import Any

from ..errors import JobFailure
from ..fbx_settings import FBX_IMPORT_KWARGS, fbx_export_kwargs
from ..protocol import (
    E_BJ_BAD_PARAMS,
    E_BJ_IMPORT_FAILED,
    RigFingerprint,
    RoundtripAnimationData,
    RoundtripAnimationParams,
    from_dict,
    to_dict,
)
from ..scene_api import SceneApi
from .export_rigged import (
    assert_rig_unchanged,
    export_fbx_or_fail,
    fingerprint_rig,
    single_armature,
)


# --------------------------------------------------------------------------- entry points
def parse_params(raw: dict[str, Any]) -> RoundtripAnimationParams:
    """Validate the JSON params into the dataclass (``E_BJ_BAD_PARAMS`` on drift)."""
    try:
        params: RoundtripAnimationParams = from_dict(RoundtripAnimationParams, raw)
    except (KeyError, TypeError, ValueError) as exc:
        raise JobFailure(E_BJ_BAD_PARAMS, f"roundtrip_animation params invalid: {exc}") from exc
    return params


def run(params: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Registry entry point: run inside Blender against :class:`BpyScene`."""
    from ..bpy_scene import BpyScene  # noqa: PLC0415 - bpy only exists inside Blender

    data, warnings = run_with_scene(parse_params(params), BpyScene())
    result: dict[str, Any] = to_dict(data)
    return result, warnings


def run_with_scene(
    params: RoundtripAnimationParams, scene: SceneApi
) -> tuple[RoundtripAnimationData, list[str]]:
    """The normative algorithm (plan 12 §1.3) against any :class:`SceneApi`.

    Args:
        params: Validated job parameters.
        scene: Scene port.

    Returns:
        ``(data, warnings)``; warnings list missing/extra bones in words.

    Raises:
        JobFailure: ``E_BJ_IMPORT_FAILED``, ``E_BJ_NO_ARMATURE``, ``E_BJ_MULTI_ARMATURE``,
            ``E_BJ_EXPORT_FAILED``, ``E_BJ_RIG_MUTATED``.
    """
    warnings: list[str] = []
    scene.reset()
    import_source(scene, params.source_fbx)
    armature = single_armature(scene)
    meshes = scene.meshes()  # kept as-is; never subdivided or unwrapped
    scene.clean_action_names()
    scene.reset_pose(armature)
    rig = fingerprint_rig(scene, armature)
    missing, extra = compare_bone_names(rig, params.expected_bone_names)
    if missing:
        warnings.append(
            "bones expected from the model but absent in the clip: " + ", ".join(missing)
        )
    if extra:
        warnings.append("bones in the clip that the model does not have: " + ", ".join(extra))
    kwargs = fbx_export_kwargs(params.export.fbx_scale_option, params.export.embed_textures)
    export_fbx_or_fail(scene, [armature, *meshes], params.out_fbx, kwargs)
    assert_rig_unchanged(rig, fingerprint_rig(scene, armature))
    data = RoundtripAnimationData(
        rig=rig,
        actions_exported=scene.actions(),
        missing_bones=missing,
        extra_bones=extra,
    )
    return data, warnings


# --------------------------------------------------------------------------- steps
def import_source(scene: SceneApi, source_fbx: str) -> None:
    """Import the companion clip with the normative kwargs (``E_BJ_IMPORT_FAILED``)."""
    try:
        scene.import_fbx(source_fbx, **FBX_IMPORT_KWARGS)
    except (RuntimeError, OSError) as exc:
        raise JobFailure(
            E_BJ_IMPORT_FAILED, f"FBX import failed: {exc}", {"source_fbx": source_fbx}
        ) from exc


def compare_bone_names(rig: RigFingerprint, expected: list[str]) -> tuple[list[str], list[str]]:
    """``(missing, extra)`` bone names of ``rig`` relative to ``expected`` (sorted)."""
    present = {bone.name for bone in rig.bones}
    wanted = set(expected)
    return sorted(wanted - present), sorted(present - wanted)
