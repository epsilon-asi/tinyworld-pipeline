"""``FakeBlenderRunner``: scripted :class:`BlenderRunner` for stage tests (plan 15 §3.2).

``FakeBlenderRunner({"prepare_model": prepare_model_response()})`` records every request,
counts calls per job (used to assert skipping) and lets a response callable either return a
:class:`JobResult` or an exception instance to raise (``BlenderTimeoutError`` etc.).
:func:`prepare_model_response` writes the declared output files (``static.glb`` with bytes whose
sha matches the returned data, ``prepared.blend``, ``prepare_report.json``) so stage output and
cross-check assertions pass. :func:`export_rigged_response` / :func:`roundtrip_animation_response`
do the same for the export stage (FBX bytes + ``rig_fingerprint_out.json``), and
:func:`bake_textures_response` writes one small solid PNG per requested bake map.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from PIL import Image as PilImage

from grounded_pipeline.blender.protocol import (
    BakeTexturesData,
    BakeTexturesParams,
    Bone,
    ExportRiggedData,
    ExportRiggedParams,
    JobName,
    JobRequest,
    JobResult,
    MeshStats,
    PrepareModelData,
    PrepareModelParams,
    RigFingerprint,
    RoundtripAnimationData,
    RoundtripAnimationParams,
    UvStats,
)
from grounded_pipeline.domain.errors import BlenderJobError, CancelledError
from grounded_pipeline.domain.ports import CancelToken

Response = Callable[[JobRequest], JobResult | Exception]
DEFAULT_STATIC_GLB = b"FAKE_GLB\nCube"
DEFAULT_EXPORT_FBX = b"FAKE_FBX\nArmature\nCube"
DEFAULT_CLIP_FBX = b"FAKE_FBX\nArmature"
DEFAULT_BLENDER_VERSION = "4.5.12"


class FakeBlenderRunner:
    """Scripted Blender runner; see the module docstring."""

    def __init__(self, responses: Mapping[JobName, Response] | None = None) -> None:
        self.responses: dict[JobName, Response] = dict(responses or {})
        self.requests: list[JobRequest] = []
        self.calls: Counter[str] = Counter()
        self.timeouts: list[int] = []
        self.log_paths: list[Path] = []

    def run(
        self, req: JobRequest, *, timeout_s: int, log_path: Path, cancel: CancelToken
    ) -> JobResult:
        """Record the request, write a log line, then return/raise the scripted response."""
        self.requests.append(req)
        self.calls[req.job] += 1
        self.timeouts.append(timeout_s)
        self.log_paths.append(log_path)
        if cancel.is_set():
            raise CancelledError(f"Cancelled before Blender job {req.job} started.")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"[fake-runner] job={req.job} job_id={req.job_id}\n")
        handler = self.responses.get(req.job)
        if handler is None:
            raise BlenderJobError("E_BJ_UNKNOWN_JOB", f"no scripted response for {req.job!r}")
        outcome = handler(req)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def call_count(self, job: JobName) -> int:
        """How many times ``job`` was requested."""
        return self.calls[job]


# --------------------------------------------------------------------------- scripted responses
def sample_rig(armature: str = "Armature") -> RigFingerprint:
    """The fixture rig (``Root → Spine → Head``) as the orchestrator-side contract value."""
    names = ("Root", "Spine", "Head")
    bones: list[Bone] = []
    path = armature
    for i, name in enumerate(names):
        z = i * 0.5
        path = f"{path}/{name}"
        bones.append(
            Bone(
                name=name,
                parent=names[i - 1] if i else None,
                head=(0.0, 0.0, z),
                tail=(0.0, 0.0, z + 0.5),
                matrix_local=[
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, z],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                transform_path=path,
            )
        )
    return RigFingerprint(
        armature_object=armature,
        armature_data=armature,
        bone_count=3,
        bones=bones,
        actions=["idle"],
        fingerprint_sha256=hashlib.sha256(b"fake-rig").hexdigest(),
    )


def sample_prepare_data(*, static_glb_sha256: str) -> PrepareModelData:
    """A plausible ``PrepareModelData`` for the fixture cube subdivided once."""
    return PrepareModelData(
        rig=sample_rig(),
        uv=UvStats(
            uv_layers=["UVMap"],
            overlap_ratio=0.0,
            island_count=1,
            source_texture_max_px=256,
            decision="kept",
            reason="uvs usable",
        ),
        mesh=MeshStats(
            object_names=["Cube"],
            vertices_before=8,
            faces_before=6,
            vertices_after=26,
            faces_after=24,
            triangles_after=48,
            shapekey_meshes=[],
            shapekeys_policy_applied="none",
            material_slots_before=1,
            material_slots_after=1,
        ),
        static_glb_sha256=static_glb_sha256,
        static_glb_triangles=48,
        static_glb_unique_positions=26,
        bbox_diagonal=1.7320508,
    )


def prepare_model_response(
    *,
    static_glb_bytes: bytes = DEFAULT_STATIC_GLB,
    data_overrides: Mapping[str, Any] | None = None,
    warnings: tuple[str, ...] = (),
    write_outputs: bool = True,
    blender_version: str = DEFAULT_BLENDER_VERSION,
) -> Response:
    """Response for ``prepare_model`` that writes the declared outputs and returns matching data.

    Args:
        static_glb_bytes: Bytes written to ``params.out_static_glb``; the returned
            ``static_glb_sha256`` is their sha (unless overridden).
        data_overrides: Top-level keys merged into the dumped data (e.g. a wrong sha, or
            ``{"bogus": 1}`` to provoke protocol drift).
        warnings: Returned in ``JobResult.warnings``.
        write_outputs: Set False to simulate a job that reports success without files.
        blender_version: Reported Blender version.
    """

    def respond(req: JobRequest) -> JobResult:
        params = PrepareModelParams.model_validate(req.params)
        data = sample_prepare_data(static_glb_sha256=hashlib.sha256(static_glb_bytes).hexdigest())
        payload: dict[str, Any] = data.model_dump(mode="json")
        payload.update(dict(data_overrides or {}))
        if write_outputs:
            for path in (params.out_blend, params.out_static_glb, params.out_report):
                path.parent.mkdir(parents=True, exist_ok=True)
            params.out_static_glb.write_bytes(static_glb_bytes)
            params.out_blend.write_bytes(b"FAKE_BLEND\n")
            params.out_report.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return JobResult(
            job=req.job,
            job_id=req.job_id,
            ok=True,
            blender_version=blender_version,
            duration_s=1.0,
            data=payload,
            warnings=list(warnings),
        )

    return respond


def _ok(req: JobRequest, payload: dict[str, Any], warnings: tuple[str, ...]) -> JobResult:
    return JobResult(
        job=req.job,
        job_id=req.job_id,
        ok=True,
        blender_version=DEFAULT_BLENDER_VERSION,
        duration_s=1.0,
        data=payload,
        warnings=list(warnings),
    )


def export_rigged_response(
    *,
    rig: RigFingerprint | None = None,
    actions: tuple[str, ...] = ("idle",),
    fbx_bytes: bytes = DEFAULT_EXPORT_FBX,
    warnings: tuple[str, ...] = (),
    write_outputs: bool = True,
    data_overrides: Mapping[str, Any] | None = None,
) -> Response:
    """Response for ``export_rigged``: writes ``out_fbx`` + ``out_rig_fingerprint``, returns data.

    Args:
        rig: Rig reported back (default: the fixture rig, i.e. a perfect RFC pass).
        actions: ``actions_exported``.
        fbx_bytes: Bytes written to ``params.out_fbx``; ``fbx_sha256`` is their sha.
        warnings: Returned in ``JobResult.warnings``.
        write_outputs: Set False to simulate a job that reports success without files.
        data_overrides: Top-level keys merged into the dumped data (drift / wrong sha).
    """

    def respond(req: JobRequest) -> JobResult:
        params = ExportRiggedParams.model_validate(req.params)
        data = ExportRiggedData(
            rig=rig or sample_rig(),
            fbx_sha256=hashlib.sha256(fbx_bytes).hexdigest(),
            actions_exported=list(actions),
        )
        payload: dict[str, Any] = data.model_dump(mode="json")
        payload.update(dict(data_overrides or {}))
        if write_outputs:
            params.out_fbx.parent.mkdir(parents=True, exist_ok=True)
            params.out_fbx.write_bytes(fbx_bytes)
            params.out_rig_fingerprint.parent.mkdir(parents=True, exist_ok=True)
            params.out_rig_fingerprint.write_text(
                json.dumps(payload["rig"], indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        return _ok(req, payload, warnings)

    return respond


def roundtrip_animation_response(
    *,
    rig: RigFingerprint | None = None,
    actions: tuple[str, ...] = ("walk",),
    fbx_bytes: bytes = DEFAULT_CLIP_FBX,
    warnings: tuple[str, ...] = (),
    write_outputs: bool = True,
    rig_by_source: Mapping[str, RigFingerprint] | None = None,
) -> Response:
    """Response for ``roundtrip_animation``: writes ``out_fbx`` and returns the clip's rig.

    ``missing_bones`` / ``extra_bones`` are derived from ``rig`` vs ``expected_bone_names`` the
    way the real job does. ``rig_by_source`` picks a rig per source file name (falls back to
    ``rig``); the default rig is the fixture rig with ``actions`` as its action list.
    """

    def respond(req: JobRequest) -> JobResult:
        params = RoundtripAnimationParams.model_validate(req.params)
        chosen = (rig_by_source or {}).get(params.source_fbx.name) or rig
        if chosen is None:
            chosen = sample_rig().model_copy(update={"actions": list(actions)})
        names = {b.name for b in chosen.bones}
        expected = set(params.expected_bone_names)
        data = RoundtripAnimationData(
            rig=chosen,
            actions_exported=list(actions),
            missing_bones=sorted(expected - names),
            extra_bones=sorted(names - expected),
        )
        if write_outputs:
            params.out_fbx.parent.mkdir(parents=True, exist_ok=True)
            params.out_fbx.write_bytes(fbx_bytes + params.source_fbx.name.encode("utf-8"))
        return _ok(req, data.model_dump(mode="json"), warnings)

    return respond


DEFAULT_BAKE_PIXELS: dict[str, tuple[int, int, int, int]] = {
    "basecolor": (200, 100, 50, 255),
    "normal": (128, 128, 255, 255),
    "roughness": (128, 128, 128, 255),
    "metallic": (64, 64, 64, 255),
    "occlusion": (180, 180, 180, 255),
    "emission": (10, 20, 30, 255),
}
"""Solid colours written per baked map (greyscale maps carry their value in every channel)."""


def bake_textures_response(
    *,
    pixels: Mapping[str, tuple[int, int, int, int]] | None = None,
    skip: tuple[str, ...] = (),
    size: int = 2,
    data_overrides: Mapping[str, Any] | None = None,
    warnings: tuple[str, ...] = (),
    write_outputs: bool = True,
    blender_version: str = DEFAULT_BLENDER_VERSION,
) -> Response:
    """Response for ``bake_textures`` that writes ``<out_dir>/<map>.png`` for each requested map.

    Args:
        pixels: RGBA value per map name; defaults to :data:`DEFAULT_BAKE_PIXELS`. Maps requested
            in ``params.bake.maps`` but absent here are not produced.
        skip: Requested maps to leave out (as the job does when the source lacks the input);
            each adds a ``"<map>: ... skipped"`` warning.
        size: Square size of every written PNG.
        data_overrides: Top-level keys merged into the dumped data (protocol-drift tests).
        warnings: Extra warnings returned in ``JobResult.warnings``.
        write_outputs: Set False to report maps without writing their files.
        blender_version: Reported Blender version.
    """

    def respond(req: JobRequest) -> JobResult:
        params = BakeTexturesParams.model_validate(req.params)
        colours = dict(DEFAULT_BAKE_PIXELS if pixels is None else pixels)
        produced = {
            name: params.out_dir / f"{name}.png"
            for name in params.bake.maps
            if name in colours and name not in skip
        }
        if write_outputs:
            params.out_dir.mkdir(parents=True, exist_ok=True)
            for name, path in produced.items():
                PilImage.new("RGBA", (size, size), colours[name]).save(path)
        data = BakeTexturesData(
            maps=produced,
            resolution=params.bake.resolution,
            samples=params.bake.samples,
            bake_seconds=dict.fromkeys(produced, 0.5),
        )
        payload: dict[str, Any] = data.model_dump(mode="json")
        payload.update(dict(data_overrides or {}))
        skipped = [f"{name}: source material has no data; skipped" for name in skip]
        return JobResult(
            job=req.job,
            job_id=req.job_id,
            ok=True,
            blender_version=blender_version,
            duration_s=2.0,
            data=payload,
            warnings=[*skipped, *warnings],
        )

    return respond


def job_error_response(code: str, message: str, **detail: Any) -> Response:
    """Response that raises :class:`BlenderJobError` (what the real runner does on ``ok=False``)."""

    def respond(req: JobRequest) -> Exception:
        return BlenderJobError(code, message, detail={**detail, "job_id": req.job_id})

    return respond
