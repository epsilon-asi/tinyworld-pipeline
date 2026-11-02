"""``bake_textures`` job: bake the provider's textured result onto our prepared mesh.

Implements the normative seven-step algorithm of plan 11 §5.2 (appendix B §5) against the
:class:`SceneApi` port: open ``prepared.blend``, import ``result.glb`` as the *source* objects,
align their bounding box to our *target* meshes, then run one Cycles selected-to-active bake per
configured map. ``basecolor``/``metallic``/``occlusion`` use the EMIT trick (the input is routed
through an Emission shader on the source material and restored afterwards); ``emission`` bakes the
material's own emission natively, so ``KHR_materials_emissive_strength`` is honoured through the
Principled ``Emission Strength`` the importer sets. ``prepared.blend`` is never saved.

Deviation from the plan 11 §5.2 table (measured on Blender 4.5.12, WP-08 integration test): a
``DIFFUSE``/``COLOR`` bake of a Principled BSDF is scaled by ``(1 - metallic)`` and by the
dielectric specular layer's energy loss (~4 %), so a pure red texture with metallic 0.25 came back
as ``(221, 0, 0)``. Base colour therefore uses the EMIT trick, which reproduces the source texture
exactly (the acceptance bar is a mean error < 2/255).

:func:`run_with_scene` is the testable core (``FakeScene`` in unit tests); :func:`run` is the
registry entry point and builds ``BpyScene`` lazily so this module imports under plain CPython.
"""

from __future__ import annotations

import math
import os
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from ..errors import JobFailure
from ..protocol import (
    E_BJ_BAD_PARAMS,
    E_BJ_BAKE_EMPTY,
    E_BJ_BAKE_FAILED,
    E_BJ_BAKE_NO_SOURCE_MATERIAL,
    E_BJ_IMPORT_FAILED,
    E_BJ_NO_SKINNED_MESH,
    BakeTexturesData,
    BakeTexturesParams,
    from_dict,
    to_dict,
)
from ..scene_api import SceneApi, ShaderInput, Vec3

GLB_IMPORT_KWARGS: dict[str, Any] = {"merge_vertices": False, "import_shading": "NORMALS"}
"""Plan 11 §5.2 step 2: keep the provider's vertex split and its shading normals."""
EMIT_INPUT_BASECOLOR = "Base Color"
CYCLES_DEVICE_ENV = "GP_BLENDER_CYCLES_DEVICE"
CYCLES_DEVICES = frozenset({"CPU", "GPU"})
ALIGN_TOLERANCE = 1e-3
"""Relative bbox mismatch above which the source objects are transformed onto the target."""
EMIT_SAMPLES = 1
"""EMIT bakes cannot be noisy, so one sample suffices (plan 11 §5.2)."""
EMPTY_EPSILON = 1e-4
"""``image_stats`` mean and variance both below this → the bake produced an all-black image."""
Box = tuple[Vec3, Vec3]
Presence = Literal["linked", "constant", "absent"]


@dataclass(frozen=True)
class MapSpec:
    """How one configured map is baked (the type table of plan 11 §5.2 step 6).

    Attributes:
        bake_type: Cycles bake type.
        colorspace: Image colour space (``sRGB`` for colour data, ``Non-Color`` otherwise).
        emit_input: Source shader input rewired through an Emission shader (EMIT trick) or
            ``None`` for native bake types.
        input_name: Source shader input that must exist for the map to be baked at all.
    """

    bake_type: str
    colorspace: str
    emit_input: str | None
    input_name: str


MAP_SPECS: dict[str, MapSpec] = {
    "basecolor": MapSpec("EMIT", "sRGB", EMIT_INPUT_BASECOLOR, EMIT_INPUT_BASECOLOR),
    "normal": MapSpec("NORMAL", "Non-Color", None, "Normal"),
    "roughness": MapSpec("ROUGHNESS", "Non-Color", None, "Roughness"),
    "metallic": MapSpec("EMIT", "Non-Color", "Metallic", "Metallic"),
    "occlusion": MapSpec("EMIT", "Non-Color", "Occlusion", "Occlusion"),
    "emission": MapSpec("EMIT", "sRGB", None, "Emission Color"),
}
_LINK_ONLY_INPUTS = frozenset({"Normal", "Occlusion"})  # a constant carries no data here


# --------------------------------------------------------------------------- entry points
def parse_params(raw: dict[str, Any]) -> BakeTexturesParams:
    """Validate the JSON params into the dataclass (``E_BJ_BAD_PARAMS`` on drift)."""
    try:
        params: BakeTexturesParams = from_dict(BakeTexturesParams, raw)
    except (KeyError, TypeError, ValueError) as exc:
        raise JobFailure(E_BJ_BAD_PARAMS, f"bake_textures params invalid: {exc}") from exc
    return params


def run(params: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Registry entry point: run inside Blender against :class:`BpyScene`."""
    from ..bpy_scene import BpyScene  # noqa: PLC0415 - bpy only exists inside Blender

    data, warnings = run_with_scene(parse_params(params), BpyScene())
    result: dict[str, Any] = to_dict(data)
    return result, warnings


def run_with_scene(
    params: BakeTexturesParams, scene: SceneApi, *, env: Mapping[str, str] | None = None
) -> tuple[BakeTexturesData, list[str]]:
    """The normative algorithm (plan 11 §5.2) against any :class:`SceneApi`.

    Args:
        params: Validated job parameters.
        scene: Scene port (``BpyScene`` in Blender, ``FakeScene`` in tests).
        env: Environment for ``GP_BLENDER_CYCLES_DEVICE``; defaults to ``os.environ``.

    Returns:
        ``(data, warnings)``; skipped maps are reported in ``warnings``.

    Raises:
        JobFailure: ``E_BJ_IMPORT_FAILED``, ``E_BJ_NO_SKINNED_MESH``,
            ``E_BJ_BAKE_NO_SOURCE_MATERIAL``, ``E_BJ_BAKE_FAILED``, ``E_BJ_BAKE_EMPTY`` or
            ``E_BJ_BAD_PARAMS`` (unknown map name).
    """
    warnings: list[str] = []
    cfg = params.bake
    scene.open_blend(params.prepared_blend)
    armatures = scene.armatures()
    targets = _targets(scene, armatures)
    sources = _import_sources(scene, params.source_glb)
    _align(scene, sources, targets, warnings)
    inputs = _source_inputs(scene, sources)
    _ensure_single_material(scene, targets, warnings)
    for armature in armatures:
        scene.hide_from_render(armature)
        scene.set_pose_position(armature, True)
    device = cycles_device(os.environ if env is None else env, warnings)
    diagonal = params.bbox_diagonal
    scene.configure_bake(
        cfg.margin_px, cfg.cage_extrusion_ratio * diagonal, cfg.max_ray_distance_ratio * diagonal
    )
    try:
        maps, seconds = _bake_maps(scene, params, targets, sources, inputs, device, warnings)
    finally:
        for armature in armatures:
            scene.set_pose_position(armature, False)
    data = BakeTexturesData(
        maps=maps, resolution=cfg.resolution, samples=cfg.samples, bake_seconds=seconds
    )
    return data, warnings


# --------------------------------------------------------------------------- pure decisions
def alignment(source: Box, target: Box) -> tuple[float, Vec3] | None:
    """Uniform scale + translation mapping the source box onto the target box, or ``None``.

    ``None`` means the boxes already agree within :data:`ALIGN_TOLERANCE` (relative to the target
    diagonal) or one of them is degenerate. Plan 11 §5.2 step 3.
    """
    source_diag, target_diag = math.dist(*source), math.dist(*target)
    if source_diag <= 0.0 or target_diag <= 0.0:
        return None
    scale = target_diag / source_diag
    source_centre = _centre(source)
    target_centre = _centre(target)
    translation = (
        target_centre[0] - scale * source_centre[0],
        target_centre[1] - scale * source_centre[1],
        target_centre[2] - scale * source_centre[2],
    )
    offset = max(abs(t) for t in translation) / target_diag
    if abs(scale - 1.0) <= ALIGN_TOLERANCE and offset <= ALIGN_TOLERANCE:
        return None
    return scale, translation


def source_presence(inputs: list[ShaderInput], map_name: str) -> Presence:
    """Whether the source material can feed ``map_name``: texture-driven, constant, or absent.

    Constants count as data for ``Base Color`` and ``Roughness`` (always meaningful), for
    ``Metallic`` when positive and for ``Emission Color`` when the colour is non-black and the
    ``Emission Strength`` is positive. ``Normal`` and ``Occlusion`` need a texture link. A
    *linked* ``Emission Color`` is ``absent`` when ``Emission Strength`` is zero: that is how
    Blender's glTF importer represents an ``emissiveTexture`` with the default black
    ``emissiveFactor``, and baking it would (correctly) produce an all-black map.
    """
    spec = MAP_SPECS[map_name]
    sockets = [s for s in inputs if s.name == spec.input_name]
    if any(s.linked for s in sockets):
        if spec.input_name == "Emission Color" and not _emission_strength_positive(inputs):
            return "absent"
        return "linked"
    if spec.input_name in _LINK_ONLY_INPUTS or not sockets:
        return "absent"
    if spec.input_name == "Metallic":
        return "constant" if any(s.value and s.value[0] > 0.0 for s in sockets) else "absent"
    if spec.input_name == "Emission Color":
        return "constant" if _emission_constant(inputs, sockets) else "absent"
    return "constant"


def cycles_device(env: Mapping[str, str], warnings: list[str]) -> str:
    """``GP_BLENDER_CYCLES_DEVICE`` (``CPU`` | ``GPU``), defaulting to CPU on anything else."""
    raw = env.get(CYCLES_DEVICE_ENV, "").strip().upper()
    if not raw:
        return "CPU"
    if raw not in CYCLES_DEVICES:
        warnings.append(f"{CYCLES_DEVICE_ENV}={raw!r} is not CPU or GPU; using CPU")
        return "CPU"
    return raw


def _emission_strength_positive(inputs: list[ShaderInput]) -> bool:
    """``Emission Strength`` > 0 (an absent socket counts as the default strength of 1)."""
    strengths = [s.value[0] for s in inputs if s.name == "Emission Strength" and s.value]
    return (max(strengths) if strengths else 1.0) > 0.0


def _emission_constant(inputs: list[ShaderInput], colours: list[ShaderInput]) -> bool:
    return _emission_strength_positive(inputs) and any(
        any(c > 0.0 for c in s.value[:3]) for s in colours
    )


def _centre(box: Box) -> Vec3:
    low, high = box
    return ((low[0] + high[0]) / 2, (low[1] + high[1]) / 2, (low[2] + high[2]) / 2)


# --------------------------------------------------------------------------- steps
def _targets(scene: SceneApi, armatures: list[str]) -> list[str]:
    """Step 1: our skinned meshes (every mesh when the file has no armature)."""
    targets: list[str] = []
    for armature in armatures:
        targets.extend(m for m in scene.skinned_meshes(armature) if m not in targets)
    if not armatures:
        targets = scene.meshes()
    if not targets:
        raise JobFailure(
            E_BJ_NO_SKINNED_MESH,
            "prepared.blend contains no skinned mesh to bake onto",
            {"armatures": armatures, "meshes": scene.meshes()},
        )
    return sorted(targets)


def _import_sources(scene: SceneApi, source_glb: str) -> list[str]:
    """Step 2: import the provider result; the new mesh objects are the bake sources."""
    before = set(scene.meshes())
    try:
        scene.import_glb(source_glb, **GLB_IMPORT_KWARGS)
    except (RuntimeError, OSError) as exc:
        raise JobFailure(
            E_BJ_IMPORT_FAILED, f"GLB import failed: {exc}", {"source_glb": source_glb}
        ) from exc
    sources = sorted(set(scene.meshes()) - before)
    if not sources:
        raise JobFailure(
            E_BJ_IMPORT_FAILED, "GLB import produced no mesh objects", {"source_glb": source_glb}
        )
    return sources


def _align(scene: SceneApi, sources: list[str], targets: list[str], warnings: list[str]) -> None:
    """Step 3: move/scale the SOURCE objects onto the target box; the target is never touched."""
    fit = alignment(scene.object_bbox(sources), scene.object_bbox(targets))
    if fit is None:
        return
    scale, translation = fit
    scene.transform_objects(sources, scale, translation)
    warnings.append(
        f"source aligned to target bbox: scale={scale:.6g} translation="
        f"({translation[0]:.6g}, {translation[1]:.6g}, {translation[2]:.6g})"
    )


def _source_inputs(scene: SceneApi, sources: list[str]) -> list[ShaderInput]:
    inputs = [socket for source in sources for socket in scene.shader_inputs(source)]
    if not inputs:
        raise JobFailure(
            E_BJ_BAKE_NO_SOURCE_MATERIAL,
            "the provider result has no material with shader inputs to bake from",
            {"sources": sources, "hint": "inspect work/tripo/result.glb; re-run the texture stage"},
        )
    return inputs


def _ensure_single_material(scene: SceneApi, targets: list[str], warnings: list[str]) -> None:
    """Step 4: one shared material on every target (prepare guarantees it; guard anyway)."""
    slots = {t: [s for s in scene.material_slots(t) if s] for t in targets}
    names = {name for per_target in slots.values() for name in per_target}
    if len(names) == 1 and all(len(per_target) == 1 for per_target in slots.values()):
        return
    name = next(iter(sorted(names)), f"{targets[0]}_mat")
    scene.merge_material_slots(targets, name)
    warnings.append(f"target materials merged into {name!r} before baking")


def _bake_maps(  # noqa: PLR0917 - internal step with the job's working set
    scene: SceneApi,
    params: BakeTexturesParams,
    targets: list[str],
    sources: list[str],
    inputs: list[ShaderInput],
    device: str,
    warnings: list[str],
) -> tuple[dict[str, str], dict[str, float]]:
    """Step 6: one bake per configured map, in ``cfg.maps`` order; skips report a warning."""
    cfg = params.bake
    maps: dict[str, str] = {}
    seconds: dict[str, float] = {}
    for map_name in cfg.maps:
        spec = MAP_SPECS.get(map_name)
        if spec is None:
            raise JobFailure(
                E_BJ_BAD_PARAMS, f"unknown bake map {map_name!r}", {"known": sorted(MAP_SPECS)}
            )
        presence = source_presence(inputs, map_name)
        if presence == "absent":
            warnings.append(f"{map_name}: source material has no {spec.input_name} data; skipped")
            continue
        samples = EMIT_SAMPLES if spec.bake_type == "EMIT" else cfg.samples
        scene.set_render_engine_cycles(device, samples)
        out_path = str(Path(params.out_dir) / f"{map_name}.png")
        started = time.perf_counter()
        _bake_one(scene, targets, sources, map_name, spec, cfg.resolution, out_path)
        seconds[map_name] = round(time.perf_counter() - started, 3)
        if presence == "linked":
            _check_not_empty(scene, map_name, out_path)
        maps[map_name] = out_path
    return maps, seconds


def _bake_one(  # noqa: PLR0917 - internal step with the job's working set
    scene: SceneApi,
    targets: list[str],
    sources: list[str],
    map_name: str,
    spec: MapSpec,
    resolution: int,
    out_path: str,
) -> None:
    try:
        scene.bake_map(
            targets,
            sources,
            map_name=map_name,
            bake_type=spec.bake_type,
            resolution=resolution,
            out_path=out_path,
            colorspace=spec.colorspace,
            emit_input=spec.emit_input,
        )
    except RuntimeError as exc:
        raise JobFailure(
            E_BJ_BAKE_FAILED,
            f"bake of {map_name} failed: {exc}",
            {"map": map_name, "bake_type": spec.bake_type},
        ) from exc
    if not Path(out_path).is_file():
        raise JobFailure(E_BJ_BAKE_FAILED, f"bake of {map_name} wrote no file", {"map": map_name})


def _check_not_empty(scene: SceneApi, map_name: str, out_path: str) -> None:
    """``E_BJ_BAKE_EMPTY``: an all-black result where the source had texture data."""
    try:
        mean, variance = scene.image_stats(out_path)
    except RuntimeError as exc:
        raise JobFailure(
            E_BJ_BAKE_FAILED, f"cannot read back {map_name}: {exc}", {"map": map_name}
        ) from exc
    if mean < EMPTY_EPSILON and variance < EMPTY_EPSILON:
        raise JobFailure(
            E_BJ_BAKE_EMPTY,
            f"bake of {map_name} is all black although the source has texture data",
            {
                "map": map_name,
                "mean": mean,
                "variance": variance,
                "hint": "check bake.cage_extrusion_ratio / max_ray_distance_ratio and the UVs",
            },
        )
