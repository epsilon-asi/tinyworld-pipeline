"""Blender job protocol — orchestrator side (plan 06 §5).

Mirrored field-for-field by frozen dataclasses in ``blender_jobs/protocol.py`` (which cannot use
pydantic). ``tests/unit/blender/test_protocol_parity.py`` enforces the mirror.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from grounded_pipeline.domain._base import FrozenModel
from grounded_pipeline.domain.ids import ModelId, Sha256
from grounded_pipeline.domain.library_config import (
    BakeSection,
    ExportSection,
    GeometrySection,
    UvSection,
)

PROTOCOL_VERSION = 1
JobName = Literal["probe", "prepare_model", "bake_textures", "export_rigged", "roundtrip_animation"]
JOB_NAMES: tuple[JobName, ...] = (
    "probe",
    "prepare_model",
    "bake_textures",
    "export_rigged",
    "roundtrip_animation",
)

# entry.py exit codes
EXIT_OK = 0
EXIT_UNHANDLED = 2
EXIT_BAD_JOB_FILE = 3


class JobError(FrozenModel):
    code: str  # E_BJ_* (plan 09 §2.8, 11 §5.2, 12 §1)
    message: str
    detail: dict[str, Any] = Field(default_factory=dict)
    traceback: str | None = None


class JobRequest(FrozenModel):
    protocol_version: Literal[1] = 1
    job: JobName
    job_id: str  # ULID; appears in logs and result
    model_id: ModelId | None = None
    params: dict[str, Any]  # one of the *Params models below, dumped
    log_level: Literal["DEBUG", "INFO", "WARNING"] = "INFO"


class JobResult(FrozenModel):
    protocol_version: Literal[1] = 1
    job: JobName
    job_id: str
    ok: bool
    blender_version: str  # "4.5.12"
    duration_s: float
    data: dict[str, Any] = Field(default_factory=dict)  # one of the *Data models below, dumped
    warnings: list[str] = Field(default_factory=list)
    error: JobError | None = None


# --------------------------------------------------------------------------- probe
class ProbeParams(FrozenModel):
    pass


class ProbeData(FrozenModel):
    blender_version: str
    python_version: str
    has_cycles: bool
    fbx_io_available: bool
    gltf_io_available: bool


# --------------------------------------------------------------------------- prepare_model
class PrepareModelParams(FrozenModel):
    source_fbx: Path  # absolute
    out_blend: Path
    out_static_glb: Path
    out_report: Path  # prepare_report.json (also returned as data)
    geometry: GeometrySection
    uv: UvSection
    material_name: str  # "<model_id>_mat"
    palette_hint_textures: list[Path] = Field(default_factory=list)  # source/textures/*


class Bone(FrozenModel):
    name: str
    parent: str | None
    head: tuple[float, float, float]  # armature space, rest pose
    tail: tuple[float, float, float]
    matrix_local: list[list[float]]  # 4x4 row-major, rest pose, armature space
    transform_path: str  # "Armature/Root/Spine" (Unity-style)


class RigFingerprint(FrozenModel):
    armature_object: str
    armature_data: str
    bone_count: int
    bones: list[Bone]  # depth-first, stable order
    actions: list[str]  # action names found in the file
    fingerprint_sha256: Sha256  # over names+parents+quantized matrices


class UvStats(FrozenModel):
    uv_layers: list[str]
    overlap_ratio: float  # 0..1
    island_count: int
    source_texture_max_px: int | None
    decision: Literal["kept", "unwrapped", "created"]
    reason: str


class MeshStats(FrozenModel):
    object_names: list[str]
    vertices_before: int
    faces_before: int
    vertices_after: int
    faces_after: int
    triangles_after: int
    shapekey_meshes: list[str]
    shapekeys_policy_applied: Literal["none", "dropped", "skipped"]
    material_slots_before: int
    material_slots_after: int


class PrepareModelData(FrozenModel):
    rig: RigFingerprint
    uv: UvStats
    mesh: MeshStats
    static_glb_sha256: Sha256
    static_glb_triangles: int
    static_glb_unique_positions: int
    bbox_diagonal: float


# --------------------------------------------------------------------------- bake_textures
class BakeTexturesParams(FrozenModel):
    prepared_blend: Path
    source_glb: Path  # Tripo result.glb
    out_dir: Path  # work/textures/raw
    bake: BakeSection
    bbox_diagonal: float  # from PrepareModelData


class BakeTexturesData(FrozenModel):
    maps: dict[str, Path]  # "basecolor" → raw/basecolor.png …
    resolution: int
    samples: int
    bake_seconds: dict[str, float]


# --------------------------------------------------------------------------- export_rigged
class ExportRiggedParams(FrozenModel):
    prepared_blend: Path
    textures_dir: Path | None  # work/textures/urp ; None → no material binding
    out_fbx: Path
    out_rig_fingerprint: Path
    export: ExportSection
    material_name: str  # "<model_id>_mat"
    include_unskinned: bool = True


class ExportRiggedData(FrozenModel):
    rig: RigFingerprint
    fbx_sha256: Sha256
    actions_exported: list[str]


# --------------------------------------------------------------------------- roundtrip_animation
class RoundtripAnimationParams(FrozenModel):
    source_fbx: Path
    out_fbx: Path
    export: ExportSection
    expected_bone_names: list[str]  # from source rig fingerprint


class RoundtripAnimationData(FrozenModel):
    rig: RigFingerprint
    actions_exported: list[str]
    missing_bones: list[str]  # expected but absent
    extra_bones: list[str]


PARAMS_BY_JOB: dict[JobName, type[FrozenModel]] = {
    "probe": ProbeParams,
    "prepare_model": PrepareModelParams,
    "bake_textures": BakeTexturesParams,
    "export_rigged": ExportRiggedParams,
    "roundtrip_animation": RoundtripAnimationParams,
}
DATA_BY_JOB: dict[JobName, type[FrozenModel]] = {
    "probe": ProbeData,
    "prepare_model": PrepareModelData,
    "bake_textures": BakeTexturesData,
    "export_rigged": ExportRiggedData,
    "roundtrip_animation": RoundtripAnimationData,
}
