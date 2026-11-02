"""Blender job protocol — Blender side (plan 06 §5).

Frozen dataclasses mirroring ``grounded_pipeline/blender/protocol.py`` field-for-field. This module
runs inside Blender's Python, so it uses the standard library only. ``to_dict``/``from_dict`` are
generic over the dataclass metadata; nested dataclasses, lists, dicts, tuples and ``None``
round-trip. Paths are plain strings on this side (JSON boundary).
"""

from __future__ import annotations

import dataclasses
import types
import typing
from dataclasses import dataclass, field
from typing import Any, Union, get_args, get_origin, get_type_hints

PROTOCOL_VERSION = 1
JOB_NAMES: tuple[str, ...] = (
    "probe",
    "prepare_model",
    "bake_textures",
    "export_rigged",
    "roundtrip_animation",
)

EXIT_OK = 0
EXIT_UNHANDLED = 2
EXIT_BAD_JOB_FILE = 3

# --------------------------------------------------------------------------- error codes (E_BJ_*)
E_BJ_IMPORT_FAILED = "E_BJ_IMPORT_FAILED"
E_BJ_NO_ARMATURE = "E_BJ_NO_ARMATURE"
E_BJ_MULTI_ARMATURE = "E_BJ_MULTI_ARMATURE"
E_BJ_NO_SKINNED_MESH = "E_BJ_NO_SKINNED_MESH"
E_BJ_SHAPEKEYS = "E_BJ_SHAPEKEYS"
E_BJ_NO_UV_KEEP = "E_BJ_NO_UV_KEEP"
E_BJ_UNWRAP_FAILED = "E_BJ_UNWRAP_FAILED"
E_BJ_FACE_LIMIT = "E_BJ_FACE_LIMIT"
E_BJ_RIG_MUTATED = "E_BJ_RIG_MUTATED"
E_BJ_EXPORT_FAILED = "E_BJ_EXPORT_FAILED"
E_BJ_BAKE_NO_SOURCE_MATERIAL = "E_BJ_BAKE_NO_SOURCE_MATERIAL"
E_BJ_BAKE_FAILED = "E_BJ_BAKE_FAILED"
E_BJ_BAKE_EMPTY = "E_BJ_BAKE_EMPTY"
E_BJ_UNKNOWN_JOB = "E_BJ_UNKNOWN_JOB"
E_BJ_BAD_PARAMS = "E_BJ_BAD_PARAMS"
E_BJ_UNHANDLED = "E_BJ_UNHANDLED"
E_BJ_PROTOCOL = "E_BJ_PROTOCOL"  # job.json protocol_version != PROTOCOL_VERSION

ALL_BJ_CODES: tuple[str, ...] = tuple(
    value for name, value in sorted(globals().items()) if name.startswith("E_BJ_")
)


# --------------------------------------------------------------------------- (de)serialisation
def to_dict(obj: Any) -> Any:  # noqa: ANN401 - generic JSON conversion
    """Recursively convert a dataclass instance (or container of them) to JSON-compatible data."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_dict(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, (list, tuple)):
        return [to_dict(item) for item in obj]
    if isinstance(obj, dict):
        return {str(key): to_dict(value) for key, value in obj.items()}
    return obj


def _strip_optional(tp: Any) -> tuple[Any, bool]:  # noqa: ANN401
    origin = get_origin(tp)
    if origin is Union or origin is types.UnionType:
        args = [a for a in get_args(tp) if a is not type(None)]
        if len(args) == 1:
            return args[0], True
    return tp, False


def _convert(value: Any, tp: Any) -> Any:  # noqa: ANN401, PLR0911 - type dispatch
    if value is None:
        return None
    tp, _ = _strip_optional(tp)
    if dataclasses.is_dataclass(tp) and isinstance(tp, type):
        return from_dict(tp, value)
    origin = get_origin(tp)
    args = get_args(tp)
    if origin in (list, typing.List):  # noqa: UP006 - get_origin may return either
        item_tp = args[0] if args else Any
        return [_convert(item, item_tp) for item in value]
    if origin in (tuple, typing.Tuple):  # noqa: UP006
        if args and args[-1] is Ellipsis:
            return tuple(_convert(item, args[0]) for item in value)
        if args:
            return tuple(_convert(item, tp_) for item, tp_ in zip(value, args, strict=True))
        return tuple(value)
    if origin in (dict, typing.Dict):  # noqa: UP006
        value_tp = args[1] if len(args) == 2 else Any  # noqa: PLR2004
        return {str(key): _convert(item, value_tp) for key, item in value.items()}
    return value


def from_dict(cls: type[Any], data: dict[str, Any]) -> Any:  # noqa: ANN401
    """Build dataclass ``cls`` from JSON data. Unknown keys raise ``KeyError`` (protocol drift)."""
    hints = get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - names
    if unknown:
        msg = f"{cls.__name__}: unknown keys {sorted(unknown)}"
        raise KeyError(msg)
    kwargs = {name: _convert(data[name], hints[name]) for name in names if name in data}
    return cls(**kwargs)


# --------------------------------------------------------------------------- config mirrors
@dataclass(frozen=True)
class GeometrySection:
    strategy: str = "subdivide"
    subdivision_levels: int = 1
    subdivision_type: str = "catmull"
    shade_smooth: bool = True
    shapekeys: str = "fail"
    merge_material_slots: bool = True
    keep_only_one_uv_layer: bool = True
    include_unskinned: bool = True
    max_triangles: int = 500_000


@dataclass(frozen=True)
class UvSection:
    mode: str = "auto"
    palette_texture_max_px: int = 128
    overlap_ratio_threshold: float = 0.20
    smart_project_angle_deg: float = 66.0
    island_margin: float = 0.02
    pack_margin: float = 0.01


@dataclass(frozen=True)
class BakeSection:
    resolution: int = 2048
    samples: int = 16
    margin_px: int = 16
    cage_extrusion_ratio: float = 0.01
    max_ray_distance_ratio: float = 0.05
    maps: list[str] = field(
        default_factory=lambda: [
            "basecolor",
            "normal",
            "metallic",
            "roughness",
            "occlusion",
            "emission",
        ]
    )


@dataclass(frozen=True)
class ExportSection:
    fbx_scale_option: str = "FBX_SCALE_ALL"
    roundtrip_companion_animations: bool = True
    embed_textures: bool = False


# --------------------------------------------------------------------------- envelope
@dataclass(frozen=True)
class JobError:
    code: str
    message: str
    detail: dict[str, Any] = field(default_factory=dict)
    traceback: str | None = None


@dataclass(frozen=True)
class JobRequest:
    job: str
    job_id: str
    params: dict[str, Any]
    protocol_version: int = PROTOCOL_VERSION
    model_id: str | None = None
    log_level: str = "INFO"


@dataclass(frozen=True)
class JobResult:
    job: str
    job_id: str
    ok: bool
    blender_version: str
    duration_s: float
    protocol_version: int = PROTOCOL_VERSION
    data: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    error: JobError | None = None


# --------------------------------------------------------------------------- probe
@dataclass(frozen=True)
class ProbeParams:
    pass


@dataclass(frozen=True)
class ProbeData:
    blender_version: str
    python_version: str
    has_cycles: bool
    fbx_io_available: bool
    gltf_io_available: bool


# --------------------------------------------------------------------------- prepare_model
@dataclass(frozen=True)
class PrepareModelParams:
    source_fbx: str
    out_blend: str
    out_static_glb: str
    out_report: str
    geometry: GeometrySection
    uv: UvSection
    material_name: str
    palette_hint_textures: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Bone:
    name: str
    parent: str | None
    head: tuple[float, float, float]
    tail: tuple[float, float, float]
    matrix_local: list[list[float]]
    transform_path: str


@dataclass(frozen=True)
class RigFingerprint:
    armature_object: str
    armature_data: str
    bone_count: int
    bones: list[Bone]
    actions: list[str]
    fingerprint_sha256: str


@dataclass(frozen=True)
class UvStats:
    uv_layers: list[str]
    overlap_ratio: float
    island_count: int
    source_texture_max_px: int | None
    decision: str
    reason: str


@dataclass(frozen=True)
class MeshStats:
    object_names: list[str]
    vertices_before: int
    faces_before: int
    vertices_after: int
    faces_after: int
    triangles_after: int
    shapekey_meshes: list[str]
    shapekeys_policy_applied: str
    material_slots_before: int
    material_slots_after: int


@dataclass(frozen=True)
class PrepareModelData:
    rig: RigFingerprint
    uv: UvStats
    mesh: MeshStats
    static_glb_sha256: str
    static_glb_triangles: int
    static_glb_unique_positions: int
    bbox_diagonal: float


# --------------------------------------------------------------------------- bake_textures
@dataclass(frozen=True)
class BakeTexturesParams:
    prepared_blend: str
    source_glb: str
    out_dir: str
    bake: BakeSection
    bbox_diagonal: float


@dataclass(frozen=True)
class BakeTexturesData:
    maps: dict[str, str]
    resolution: int
    samples: int
    bake_seconds: dict[str, float]


# --------------------------------------------------------------------------- export_rigged
@dataclass(frozen=True)
class ExportRiggedParams:
    prepared_blend: str
    textures_dir: str | None
    out_fbx: str
    out_rig_fingerprint: str
    export: ExportSection
    material_name: str
    include_unskinned: bool = True


@dataclass(frozen=True)
class ExportRiggedData:
    rig: RigFingerprint
    fbx_sha256: str
    actions_exported: list[str]


# --------------------------------------------------------------------------- roundtrip_animation
@dataclass(frozen=True)
class RoundtripAnimationParams:
    source_fbx: str
    out_fbx: str
    export: ExportSection
    expected_bone_names: list[str]


@dataclass(frozen=True)
class RoundtripAnimationData:
    rig: RigFingerprint
    actions_exported: list[str]
    missing_bones: list[str]
    extra_bones: list[str]


PARAMS_BY_JOB: dict[str, type[Any]] = {
    "probe": ProbeParams,
    "prepare_model": PrepareModelParams,
    "bake_textures": BakeTexturesParams,
    "export_rigged": ExportRiggedParams,
    "roundtrip_animation": RoundtripAnimationParams,
}
DATA_BY_JOB: dict[str, type[Any]] = {
    "probe": ProbeData,
    "prepare_model": PrepareModelData,
    "bake_textures": BakeTexturesData,
    "export_rigged": ExportRiggedData,
    "roundtrip_animation": RoundtripAnimationData,
}

MIRRORED_TYPES: tuple[str, ...] = (
    "GeometrySection",
    "UvSection",
    "BakeSection",
    "ExportSection",
    "JobError",
    "JobRequest",
    "JobResult",
    "ProbeParams",
    "ProbeData",
    "PrepareModelParams",
    "Bone",
    "RigFingerprint",
    "UvStats",
    "MeshStats",
    "PrepareModelData",
    "BakeTexturesParams",
    "BakeTexturesData",
    "ExportRiggedParams",
    "ExportRiggedData",
    "RoundtripAnimationParams",
    "RoundtripAnimationData",
)
"""Names that must exist with identical field names on both sides (parity test)."""
