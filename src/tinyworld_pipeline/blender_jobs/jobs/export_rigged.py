"""``export_rigged`` job: bind URP textures, export the Unity-bound FBX, prove the rig is intact.

Implements plan 12 §1.2 against the :class:`SceneApi` port:

1. ``open_blend(prepared.blend)`` and fingerprint the rig;
2. when ``textures_dir`` is given, wire the URP PNGs that exist into one Principled material on
   every skinned mesh (appendix B §6; Unity ignores the node tree, this only yields sensible FBX
   material names and texture references);
3. strip the import prefixes off the action names (appendix B §8.1, so the delivered takes are
   ``Armature|idle`` like the vendor's and do not grow per round trip), reset the pose
   (appendix B §8.3: node transforms are written from the *current* pose) and export armature +
   skinned meshes (+ unskinned ones when ``include_unskinned``) with the normative
   ``fbx_export_kwargs``; every action is still written by ``bake_anim_use_all_actions``;
4. the fingerprint after export must equal the one from step 1 (``E_BJ_RIG_MUTATED``).

:func:`run_with_scene` is the testable core; :func:`run` is the registry entry point. The rig
helpers (:func:`find_rig`, :func:`fingerprint_rig`, :func:`export_fbx_or_fail`, …) are shared
with ``roundtrip_animation``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ..errors import JobFailure
from ..fbx_settings import fbx_export_kwargs
from ..protocol import (
    E_BJ_BAD_PARAMS,
    E_BJ_EXPORT_FAILED,
    E_BJ_IMPORT_FAILED,
    E_BJ_MULTI_ARMATURE,
    E_BJ_NO_ARMATURE,
    E_BJ_NO_SKINNED_MESH,
    E_BJ_RIG_MUTATED,
    ExportRiggedData,
    ExportRiggedParams,
    RigFingerprint,
    from_dict,
    to_dict,
)
from ..rig_fingerprint import rig_fingerprint
from ..scene_api import SceneApi

URP_TEXTURE_KEYS: tuple[str, ...] = ("basecolor", "normal", "metallic_smoothness", "emission")
"""URP maps wired into the Principled material (appendix B §6); occlusion has no such input."""

MATERIAL_SUFFIX = "_mat"


# --------------------------------------------------------------------------- entry points
def parse_params(raw: dict[str, Any]) -> ExportRiggedParams:
    """Validate the JSON params into the dataclass (``E_BJ_BAD_PARAMS`` on drift)."""
    try:
        params: ExportRiggedParams = from_dict(ExportRiggedParams, raw)
    except (KeyError, TypeError, ValueError) as exc:
        raise JobFailure(E_BJ_BAD_PARAMS, f"export_rigged params invalid: {exc}") from exc
    return params


def run(params: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Registry entry point: run inside Blender against :class:`BpyScene`."""
    from ..bpy_scene import BpyScene  # noqa: PLC0415 - bpy only exists inside Blender

    data, warnings = run_with_scene(parse_params(params), BpyScene())
    result: dict[str, Any] = to_dict(data)
    return result, warnings


def run_with_scene(
    params: ExportRiggedParams, scene: SceneApi
) -> tuple[ExportRiggedData, list[str]]:
    """The normative algorithm (plan 12 §1.2) against any :class:`SceneApi`.

    Args:
        params: Validated job parameters.
        scene: Scene port (``BpyScene`` in Blender, ``FakeScene`` in tests).

    Returns:
        ``(data, warnings)``; ``data.rig`` is also written to ``params.out_rig_fingerprint``.

    Raises:
        JobFailure: ``E_BJ_IMPORT_FAILED`` (blend missing/unreadable), ``E_BJ_NO_ARMATURE``,
            ``E_BJ_MULTI_ARMATURE``, ``E_BJ_NO_SKINNED_MESH``, ``E_BJ_EXPORT_FAILED``,
            ``E_BJ_RIG_MUTATED``.
    """
    warnings: list[str] = []
    open_prepared(scene, params.prepared_blend)
    armature, skinned, unskinned = find_rig(scene)
    rig_before = fingerprint_rig(scene, armature)
    if params.textures_dir is not None:
        textures = find_urp_textures(params.textures_dir, params.material_name)
        if textures:
            scene.bind_urp_material(skinned, params.material_name, textures)
        else:
            warnings.append(
                f"no URP textures found in {params.textures_dir}; exported without material binding"
            )
    objs = [armature, *skinned, *(unskinned if params.include_unskinned else [])]
    scene.clean_action_names()
    scene.reset_pose(armature)
    kwargs = fbx_export_kwargs(params.export.fbx_scale_option, params.export.embed_textures)
    export_fbx_or_fail(scene, objs, params.out_fbx, kwargs)
    rig_after = fingerprint_rig(scene, armature)
    assert_rig_unchanged(rig_before, rig_after)
    write_json(params.out_rig_fingerprint, to_dict(rig_after))
    data = ExportRiggedData(
        rig=rig_after,
        fbx_sha256=file_sha256(params.out_fbx),
        actions_exported=scene.actions(),
    )
    return data, warnings


# --------------------------------------------------------------------------- steps
def open_prepared(scene: SceneApi, path: str) -> None:
    """Open ``prepared.blend`` (``E_BJ_IMPORT_FAILED`` when missing or unreadable)."""
    if not Path(path).is_file():
        raise JobFailure(
            E_BJ_IMPORT_FAILED,
            f"prepared blend file not found: {path}",
            {"prepared_blend": path, "hint": "run the prepare stage first"},
        )
    try:
        scene.open_blend(path)
    except (RuntimeError, OSError) as exc:
        raise JobFailure(
            E_BJ_IMPORT_FAILED, f"cannot open {path}: {exc}", {"prepared_blend": path}
        ) from exc


def find_urp_textures(textures_dir: str, material_name: str) -> dict[str, str]:
    """URP PNGs present in ``textures_dir`` keyed by map name (:data:`URP_TEXTURE_KEYS`).

    ``<model_id>_<map>.png`` (the ``apply`` stage's naming, ``model_id`` derived from
    ``<model_id>_mat``) is preferred; otherwise the first ``*_<map>.png`` in name order is used.
    """
    folder = Path(textures_dir)
    if not folder.is_dir():
        return {}
    prefix = material_name.removesuffix(MATERIAL_SUFFIX)
    found: dict[str, str] = {}
    for key in URP_TEXTURE_KEYS:
        exact = folder / f"{prefix}_{key}.png"
        candidates = [exact] if exact.is_file() else sorted(folder.glob(f"*_{key}.png"))
        if candidates:
            found[key] = str(candidates[0])
    return found


def find_rig(scene: SceneApi) -> tuple[str, list[str], list[str]]:
    """``(armature, skinned meshes, unskinned meshes)``; exactly one armature with skinned meshes.

    Raises:
        JobFailure: ``E_BJ_NO_ARMATURE``, ``E_BJ_MULTI_ARMATURE``, ``E_BJ_NO_SKINNED_MESH``.
    """
    armature = single_armature(scene)
    skinned = scene.skinned_meshes(armature)
    if not skinned:
        raise JobFailure(
            E_BJ_NO_SKINNED_MESH,
            f"no mesh is bound to armature {armature!r}",
            {"armature": armature, "meshes": scene.meshes()},
        )
    unskinned = [m for m in scene.meshes() if m not in skinned]
    return armature, skinned, unskinned


def single_armature(scene: SceneApi) -> str:
    """The one armature object in the scene.

    Raises:
        JobFailure: ``E_BJ_NO_ARMATURE`` or ``E_BJ_MULTI_ARMATURE``.
    """
    armatures = scene.armatures()
    if not armatures:
        raise JobFailure(
            E_BJ_NO_ARMATURE,
            "the file contains no armature object",
            {"meshes": scene.meshes()},
        )
    if len(armatures) > 1:
        raise JobFailure(
            E_BJ_MULTI_ARMATURE,
            f"expected one armature, found {len(armatures)}: {', '.join(armatures)}",
            {"armatures": armatures},
        )
    return armatures[0]


def fingerprint_rig(scene: SceneApi, armature: str) -> RigFingerprint:
    """Rig fingerprint of ``armature`` with the scene's (prefix-stripped) object actions."""
    return rig_fingerprint(
        armature, scene.armature_data_name(armature), scene.bones(armature), scene.actions()
    )


def export_fbx_or_fail(scene: SceneApi, objs: list[str], path: str, kwargs: dict[str, Any]) -> None:
    """``export_fbx`` wrapped into ``E_BJ_EXPORT_FAILED`` (operator error or no file written)."""
    try:
        scene.export_fbx(objs, path, **kwargs)
    except RuntimeError as exc:
        raise JobFailure(E_BJ_EXPORT_FAILED, f"FBX export failed: {exc}", {"path": path}) from exc
    if not Path(path).is_file():
        raise JobFailure(E_BJ_EXPORT_FAILED, "FBX export wrote no file", {"path": path})


def assert_rig_unchanged(before: RigFingerprint, after: RigFingerprint) -> None:
    """Internal bug guard: export must never touch the armature (``E_BJ_RIG_MUTATED``)."""
    if after != before:
        raise JobFailure(
            E_BJ_RIG_MUTATED,
            "rig fingerprint changed during export (internal bug guard)",
            {"before": before.fingerprint_sha256, "after": after.fingerprint_sha256},
        )


# --------------------------------------------------------------------------- small helpers
def file_sha256(path: str) -> str:
    """Streamed SHA-256 of a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: str, data: Any) -> None:  # noqa: ANN401 - JSON payload
    """Atomic canonical JSON write (plan 19 §4)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_text(
        json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    tmp.replace(target)
