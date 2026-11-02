"""``prepare_model`` job: subdivide, (re)unwrap, fingerprint the rig, export the static GLB.

Implements the normative 12-step algorithm of plan 09 §2.2 against the :class:`SceneApi` port.
:func:`run_with_scene` is the testable core (``FakeScene`` in unit tests); :func:`run` is the
registry entry point and builds ``BpyScene`` lazily so this module imports under plain CPython.
"""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path
from typing import Any, Protocol

from ..errors import JobFailure
from ..fbx_settings import FBX_IMPORT_KWARGS, GLB_EXPORT_KWARGS
from ..protocol import (
    E_BJ_BAD_PARAMS,
    E_BJ_EXPORT_FAILED,
    E_BJ_FACE_LIMIT,
    E_BJ_IMPORT_FAILED,
    E_BJ_MULTI_ARMATURE,
    E_BJ_NO_ARMATURE,
    E_BJ_NO_SKINNED_MESH,
    E_BJ_RIG_MUTATED,
    E_BJ_SHAPEKEYS,
    E_BJ_UNWRAP_FAILED,
    MeshStats,
    PrepareModelData,
    PrepareModelParams,
    RigFingerprint,
    UvStats,
    from_dict,
    to_dict,
)
from ..rig_fingerprint import rig_fingerprint
from ..scene_api import SceneApi, Uv
from ..uv_heuristics import UvAnalysis, analyse, decide

SUBSURF_MODIFIER = "GP_Subdiv"
POSITION_QUANTUM = 1e-4
"""Same quantum as the orchestrator's ``gltf/fingerprint.py`` (``verify.position_quantum``)."""
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
_JPEG_FILL = 0xFF
_JPEG_STANDALONE = frozenset({0xFF, 0xD8, 0x01, *range(0xD0, 0xD8)})  # markers without a length
_JPEG_SOF_MARKERS = frozenset(range(0xC0, 0xD0)) - {0xC4, 0xC8, 0xCC}
_IMAGE_HEADER_BYTES = 1 << 20


# --------------------------------------------------------------------------- geometry strategy
class GeometryStrategy(Protocol):
    """Extension seam (plan 09 §2.9): how the base mesh gains detail."""

    def apply(self, scene: SceneApi, meshes: list[str], params: PrepareModelParams) -> list[str]:
        """Increase detail of ``meshes`` in place; return warnings.

        Raises:
            JobFailure: ``E_BJ_FACE_LIMIT`` when the projected result exceeds the budget.
        """
        ...


class SubdivideStrategy:
    """v1: Subdivision Surface modifier moved above the Armature modifier and applied."""

    def apply(self, scene: SceneApi, meshes: list[str], params: PrepareModelParams) -> list[str]:
        """Subdivide every mesh in ``meshes`` by ``geometry.subdivision_levels`` (0 → no-op)."""
        geo = params.geometry
        levels = geo.subdivision_levels
        if levels <= 0 or not meshes:
            return []
        projected = sum(scene.mesh_counts(m)[2] for m in meshes) * 4**levels
        if projected > geo.max_triangles:
            raise JobFailure(
                E_BJ_FACE_LIMIT,
                f"subdivision_levels={levels} would produce {projected} triangles "
                f"(> geometry.max_triangles={geo.max_triangles})",
                {
                    "projected_triangles": projected,
                    "max_triangles": geo.max_triangles,
                    "levels": levels,
                    "hint": "lower geometry.subdivision_levels or raise geometry.max_triangles",
                },
            )
        for mesh in meshes:
            scene.add_subsurf(
                mesh, SUBSURF_MODIFIER, levels, geo.subdivision_type, "PRESERVE_BOUNDARIES", False
            )
            scene.move_modifier_to_top(mesh, SUBSURF_MODIFIER)
            scene.apply_modifier(mesh, SUBSURF_MODIFIER)
        return []


STRATEGIES: dict[str, type[SubdivideStrategy]] = {"subdivide": SubdivideStrategy}


def select_strategy(name: str) -> GeometryStrategy:
    """Strategy for ``geometry.strategy`` (v1 accepts only ``subdivide``)."""
    cls = STRATEGIES.get(name)
    if cls is None:
        raise JobFailure(
            E_BJ_BAD_PARAMS, f"unknown geometry.strategy {name!r}", {"known": sorted(STRATEGIES)}
        )
    return cls()


# --------------------------------------------------------------------------- entry points
def parse_params(raw: dict[str, Any]) -> PrepareModelParams:
    """Validate the JSON params into the dataclass (``E_BJ_BAD_PARAMS`` on drift)."""
    try:
        params: PrepareModelParams = from_dict(PrepareModelParams, raw)
    except (KeyError, TypeError, ValueError) as exc:
        raise JobFailure(E_BJ_BAD_PARAMS, f"prepare_model params invalid: {exc}") from exc
    return params


def run(params: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Registry entry point: run inside Blender against :class:`BpyScene`."""
    from ..bpy_scene import BpyScene  # noqa: PLC0415 - bpy only exists inside Blender

    data, warnings = run_with_scene(parse_params(params), BpyScene())
    result: dict[str, Any] = to_dict(data)
    return result, warnings


def run_with_scene(
    params: PrepareModelParams, scene: SceneApi, *, strategy: GeometryStrategy | None = None
) -> tuple[PrepareModelData, list[str]]:
    """The normative algorithm (plan 09 §2.2) against any :class:`SceneApi`.

    Args:
        params: Validated job parameters.
        scene: Scene port (``BpyScene`` in Blender, ``FakeScene`` in tests).
        strategy: Geometry strategy override; defaults to ``params.geometry.strategy``.

    Returns:
        ``(data, warnings)``; ``data`` is also written to ``params.out_report``.

    Raises:
        JobFailure: Any ``E_BJ_*`` condition of plan 09 §2.8.
    """
    warnings: list[str] = []
    geo = params.geometry
    _import(scene, params.source_fbx)
    armature, skinned, unskinned = _find_rig(scene, warnings)
    rig_before = _fingerprint(scene, armature)
    slots_before = _slot_count(scene, skinned)
    if geo.merge_material_slots:
        scene.merge_material_slots(skinned, params.material_name)
    if geo.keep_only_one_uv_layer:
        for mesh in skinned:
            scene.keep_first_uv_layer(mesh)
    verts_before, faces_before, _ = _sum_counts(scene, skinned)
    uv_stats = _uv_step(scene, skinned, params)
    shapekey_meshes, policy_applied, subdividable = _shapekey_step(
        scene, skinned, geo.shapekeys, warnings
    )
    chosen = strategy if strategy is not None else select_strategy(geo.strategy)
    warnings.extend(chosen.apply(scene, subdividable, params))
    if geo.shade_smooth:
        for mesh in skinned:
            scene.shade_smooth(mesh)
    mesh_stats = _mesh_stats(
        scene, skinned, (verts_before, faces_before, slots_before), shapekey_meshes, policy_applied
    )
    scene.save_blend(params.out_blend)
    export_objs = [*skinned, *(unskinned if geo.include_unskinned else [])]
    _export_glb(scene, export_objs, params.out_static_glb)
    glb_triangles, unique_positions = _static_fingerprint(scene, export_objs)
    _assert_rig_unchanged(rig_before, _fingerprint(scene, armature))
    data = PrepareModelData(
        rig=rig_before,
        uv=uv_stats,
        mesh=mesh_stats,
        static_glb_sha256=_file_sha256(params.out_static_glb),
        static_glb_triangles=glb_triangles,
        static_glb_unique_positions=unique_positions,
        bbox_diagonal=scene.bbox_diagonal(export_objs),
    )
    _write_report(params.out_report, data)
    return data, warnings


# --------------------------------------------------------------------------- steps
def _mesh_stats(
    scene: SceneApi,
    meshes: list[str],
    before: tuple[int, int, int],
    shapekey_meshes: list[str],
    policy_applied: str,
) -> MeshStats:
    """Step 8: counts over the skinned meshes; ``before`` = (verts, faces, material slots)."""
    verts_after, faces_after, tris_after = _sum_counts(scene, meshes)
    return MeshStats(
        object_names=list(meshes),
        vertices_before=before[0],
        faces_before=before[1],
        vertices_after=verts_after,
        faces_after=faces_after,
        triangles_after=tris_after,
        shapekey_meshes=shapekey_meshes,
        shapekeys_policy_applied=policy_applied,
        material_slots_before=before[2],
        material_slots_after=_slot_count(scene, meshes),
    )


def _assert_rig_unchanged(before: RigFingerprint, after: RigFingerprint) -> None:
    """Step 12 invariant: steps 4-11 never touched the armature (``E_BJ_RIG_MUTATED``)."""
    if after != before:
        raise JobFailure(
            E_BJ_RIG_MUTATED,
            "rig fingerprint changed during prepare (internal bug guard)",
            {"before": before.fingerprint_sha256, "after": after.fingerprint_sha256},
        )


def _import(scene: SceneApi, source_fbx: str) -> None:
    try:
        scene.import_fbx(source_fbx, **FBX_IMPORT_KWARGS)
    except (RuntimeError, OSError) as exc:
        raise JobFailure(
            E_BJ_IMPORT_FAILED, f"FBX import failed: {exc}", {"source_fbx": source_fbx}
        ) from exc


def _find_rig(scene: SceneApi, warnings: list[str]) -> tuple[str, list[str], list[str]]:
    """Step 2: exactly one armature with at least one skinned mesh."""
    armatures = scene.armatures()
    if not armatures:
        raise JobFailure(
            E_BJ_NO_ARMATURE,
            "the file contains no armature object",
            {"meshes": scene.meshes(), "hint": "point source.primary at the rigged FBX"},
        )
    if len(armatures) > 1:
        raise JobFailure(
            E_BJ_MULTI_ARMATURE,
            f"expected one armature, found {len(armatures)}: {', '.join(armatures)}",
            {"armatures": armatures, "hint": "point source.primary at a single-rig FBX"},
        )
    armature = armatures[0]
    skinned = scene.skinned_meshes(armature)
    if not skinned:
        raise JobFailure(
            E_BJ_NO_SKINNED_MESH,
            f"no mesh is bound to armature {armature!r}",
            {"armature": armature, "meshes": scene.meshes()},
        )
    unskinned = [m for m in scene.meshes() if m not in skinned]
    if unskinned:
        warnings.append(
            "unskinned meshes are exported as-is and never subdivided: " + ", ".join(unskinned)
        )
    return armature, skinned, unskinned


def _fingerprint(scene: SceneApi, armature: str) -> RigFingerprint:
    return rig_fingerprint(
        armature, scene.armature_data_name(armature), scene.bones(armature), scene.actions()
    )


def _uv_step(scene: SceneApi, meshes: list[str], params: PrepareModelParams) -> UvStats:
    """Step 6: analyse the first UV layer of all meshes, decide, unwrap on the base mesh."""
    cfg = params.uv
    sizes = _texture_sizes(scene, meshes, params.palette_hint_textures)
    before = analyse(_uv_faces(scene, meshes), sizes)
    decision, reason = decide(before, cfg)
    if decision in ("unwrapped", "created"):
        scene.smart_unwrap(meshes, cfg.smart_project_angle_deg, cfg.island_margin, cfg.pack_margin)
        _check_unwrap(analyse(_uv_faces(scene, meshes), []), meshes)
    return UvStats(
        uv_layers=_layer_names(scene, meshes),
        overlap_ratio=before.overlap_ratio,
        island_count=before.island_count,
        source_texture_max_px=before.source_texture_max_px,
        decision=decision,
        reason=reason,
    )


def _check_unwrap(after: UvAnalysis, meshes: list[str]) -> None:
    if not after.has_uv or after.uv_bbox_coverage <= 0.0:
        raise JobFailure(
            E_BJ_UNWRAP_FAILED, "smart UV project produced zero UV area", {"meshes": meshes}
        )


def _uv_faces(scene: SceneApi, meshes: list[str]) -> list[list[Uv]]:
    """Per-face UV loops of every mesh's first layer; empty when any mesh lacks UVs."""
    faces: list[list[Uv]] = []
    for mesh in meshes:
        layers = scene.uv_layers(mesh)
        if not layers:
            return []
        loops = scene.uv_loops(mesh, layers[0])
        faces.extend(loops[start:end] for start, end in scene.face_loop_ranges(mesh))
    return faces


def _layer_names(scene: SceneApi, meshes: list[str]) -> list[str]:
    names: list[str] = []
    for mesh in meshes:
        names.extend(n for n in scene.uv_layers(mesh) if n not in names)
    return names


def _texture_sizes(scene: SceneApi, meshes: list[str], hints: list[str]) -> list[tuple[int, int]]:
    sizes: list[tuple[int, int]] = []
    for mesh in meshes:
        sizes.extend(scene.image_sizes(mesh))
    for hint in hints:
        dims = image_dimensions(hint)
        if dims is not None:
            sizes.append(dims)
    return sizes


def _shapekey_step(
    scene: SceneApi, meshes: list[str], policy: str, warnings: list[str]
) -> tuple[list[str], str, list[str]]:
    """Step 7a: ``(shapekey_meshes, policy_applied, meshes that may be subdivided)``."""
    with_keys = [m for m in meshes if scene.has_shapekeys(m)]
    if not with_keys:
        return [], "none", list(meshes)
    names = ", ".join(with_keys)
    if policy == "drop":
        for mesh in with_keys:
            scene.clear_shapekeys(mesh)
        warnings.append(f"shape keys dropped (geometry.shapekeys=drop): {names}")
        return with_keys, "dropped", list(meshes)
    if policy == "skip":
        warnings.append(
            f"meshes with shape keys left unsubdivided (geometry.shapekeys=skip): {names}"
        )
        return with_keys, "skipped", [m for m in meshes if m not in with_keys]
    raise JobFailure(
        E_BJ_SHAPEKEYS,
        f"shape keys found on {names}; they block subdivision",
        {"meshes": with_keys, "hint": "set geometry.shapekeys to drop (or skip) in model.md"},
    )


def _export_glb(scene: SceneApi, objs: list[str], path: str) -> None:
    try:
        scene.export_glb(objs, path, **GLB_EXPORT_KWARGS)
    except RuntimeError as exc:
        raise JobFailure(E_BJ_EXPORT_FAILED, f"GLB export failed: {exc}", {"path": path}) from exc
    if not Path(path).is_file():
        raise JobFailure(E_BJ_EXPORT_FAILED, "GLB export wrote no file", {"path": path})


def _static_fingerprint(scene: SceneApi, objs: list[str]) -> tuple[int, int]:
    """Step 11: triangles after triangulation and unique positions quantised at 1e-4."""
    triangles = sum(scene.mesh_counts(o)[2] for o in objs)
    unique = {
        tuple(round(c / POSITION_QUANTUM) for c in p)
        for o in objs
        for p in scene.world_vertex_positions(o)
    }
    return triangles, len(unique)


# --------------------------------------------------------------------------- small helpers
def _sum_counts(scene: SceneApi, meshes: list[str]) -> tuple[int, int, int]:
    verts = faces = tris = 0
    for mesh in meshes:
        v, f, t = scene.mesh_counts(mesh)
        verts, faces, tris = verts + v, faces + f, tris + t
    return verts, faces, tris


def _slot_count(scene: SceneApi, meshes: list[str]) -> int:
    return sum(len(scene.material_slots(m)) for m in meshes)


def _file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_report(path: str, data: PrepareModelData) -> None:
    """Atomic JSON write (plan 19 §4)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    text = json.dumps(to_dict(data), indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(target)


def image_dimensions(path: str) -> tuple[int, int] | None:
    """``(width, height)`` from a PNG or JPEG header, ``None`` if unreadable (stdlib only)."""
    try:
        with Path(path).open("rb") as handle:
            data = handle.read(_IMAGE_HEADER_BYTES)
    except OSError:
        return None
    if data.startswith(_PNG_MAGIC) and data[12:16] == b"IHDR":
        width, height = struct.unpack(">II", data[16:24])
        return int(width), int(height)
    if data[:2] == b"\xff\xd8":
        return _jpeg_dimensions(data)
    return None


def _jpeg_dimensions(data: bytes) -> tuple[int, int] | None:
    index = 2
    while index + 9 <= len(data):
        if data[index] != _JPEG_FILL:
            index += 1
            continue
        marker = data[index + 1]
        if marker in _JPEG_STANDALONE:
            index += 1 if marker == _JPEG_FILL else 2
            continue
        (length,) = struct.unpack(">H", data[index + 2 : index + 4])
        if marker in _JPEG_SOF_MARKERS:
            height, width = struct.unpack(">HH", data[index + 5 : index + 9])
            return int(width), int(height)
        index += 2 + length
    return None
