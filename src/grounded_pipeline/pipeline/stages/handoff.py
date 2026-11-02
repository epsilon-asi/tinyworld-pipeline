"""Stage ``handoff`` (plan 12 §2-§3): assemble ``output/<id>/`` and sync it into Unity.

Inputs are the reports of the earlier stages — ``work/prepare_report.json`` (uv decision),
``work/apply_report.json`` (URP maps + material scalars; tolerated when absent → no textures),
``work/verify_report.json`` (route), ``work/tripo/task.json`` (task id, credits) and the export
report (clips, rig fidelity, model FBX) — plus the resolved model (display name, licence, prompt
sha, ``[unity]``).

Layout written (plan 12 §3), staged under ``output/<id>.tmp-<run_id>`` and swapped into
``output/<id>`` atomically so a failure mid-copy leaves the previous delivery untouched::

    output/<id>/<id>.fbx
    output/<id>/manifest.json
    output/<id>/textures/<id>_*.png
    output/<id>/animations/<id>@*.fbx

With ``[library].unity_import_dir`` set the folder is then copied with ``unity.sync.copy_model``
(same swap; ``.meta`` files preserved so Unity GUIDs survive).
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import ValidationError

from grounded_pipeline.blender.protocol import RigFingerprint
from grounded_pipeline.domain.errors import HandoffError, StagePrereqError
from grounded_pipeline.domain.manifest import Manifest, ManifestProvenance
from grounded_pipeline.domain.state import StageName, StageOutcome
from grounded_pipeline.pipeline.atomic import atomic_replace_dir
from grounded_pipeline.pipeline.hashing import cfg_dump, file_sha, sha_many
from grounded_pipeline.pipeline.stages.apply import APPLY_REPORT
from grounded_pipeline.pipeline.stages.export import (
    RIG_FINGERPRINT_OUT,
    ExportReport,
    export_outputs,
    read_export_report,
)
from grounded_pipeline.pipeline.stages.texture import TASK_JSON
from grounded_pipeline.pipeline.stages.verify import PREPARE_REPORT, VERIFY_REPORT
from grounded_pipeline.unity.manifest_writer import (
    UrpMaterialInputs,
    build_manifest,
    first_root_bone,
    write_manifest,
)
from grounded_pipeline.unity.sync import copy_model

if TYPE_CHECKING:
    from grounded_pipeline.pipeline.context import ModelContext, PipelineContext

NowFn = Callable[[], datetime]
MANIFEST_JSON = "manifest.json"
TEXTURES_SUBDIR = "textures"
ANIMATIONS_SUBDIR = "animations"
SUBDIVISION_TYPE_NAMES = {"catmull": "catmull_clark", "simple": "simple"}
"""Config value → Blender subdivision type name recorded in the manifest provenance."""
W_NO_TEXTURES = "W_HANDOFF_NO_TEXTURES"


def utc_now() -> datetime:
    """Current UTC time (default ``now`` of :class:`HandoffStage`)."""
    return datetime.now(UTC)


def iso_utc(moment: datetime) -> str:
    """``2026-11-02T12:00:00Z`` (seconds precision, always UTC)."""
    return moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------- report readers
def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def read_uv_decision(folder: Path) -> str:
    """``prepare_report.uv.decision``.

    Raises:
        StagePrereqError: The prepare report is missing or lacks the field.
    """
    data = _read_json(folder / PREPARE_REPORT)
    uv = data.get("uv") if data else None
    decision = uv.get("decision") if isinstance(uv, dict) else None
    if not isinstance(decision, str):
        raise StagePrereqError(
            f"{PREPARE_REPORT.as_posix()} is missing or has no uv decision; run prepare first",
            detail={"path": str(folder / PREPARE_REPORT)},
        )
    return decision


def read_apply_material(folder: Path) -> tuple[dict[str, Path], UrpMaterialInputs] | None:
    """Absolute URP map paths + material scalars from ``apply_report.json``; ``None`` if absent."""
    data = _read_json(folder / APPLY_REPORT)
    if data is None:
        return None
    urp = data.get("urp")
    maps = {
        str(name): folder / str(rel)
        for name, rel in (urp.items() if isinstance(urp, dict) else [])
        if isinstance(rel, str)
    }
    scalars = data.get("material")
    scalars = scalars if isinstance(scalars, dict) else {}
    defaults = UrpMaterialInputs()
    emission = scalars.get("emission_color", list(defaults.emission_color))
    material = UrpMaterialInputs(
        bump_scale=float(scalars.get("bump_scale", defaults.bump_scale)),
        occlusion_strength=float(scalars.get("occlusion_strength", defaults.occlusion_strength)),
        emission_color=(float(emission[0]), float(emission[1]), float(emission[2])),
        metallic_fallback=float(scalars.get("metallic_fallback", defaults.metallic_fallback)),
        smoothness_fallback=float(scalars.get("smoothness_fallback", defaults.smoothness_fallback)),
    )
    return maps, material


def read_route(folder: Path) -> str:
    """``verify_report.route`` or ``""`` when unknown (tolerant)."""
    data = _read_json(folder / VERIFY_REPORT)
    route = data.get("route") if data else None
    return route if isinstance(route, str) else ""


def read_texture_task(folder: Path) -> tuple[str, float]:
    """``(task_id, credits_consumed)`` from ``work/tripo/task.json`` (``("", 0.0)`` if absent)."""
    data = _read_json(folder / Path(TASK_JSON))
    if data is None:
        return "", 0.0
    task_id = data.get("task_id")
    spent = data.get("credits_consumed")
    return (
        task_id if isinstance(task_id, str) else "",
        float(spent) if isinstance(spent, int | float) and not isinstance(spent, bool) else 0.0,
    )


def read_output_rig(folder: Path) -> RigFingerprint:
    """The exported rig (``work/export/rig_fingerprint_out.json``).

    Raises:
        StagePrereqError: Missing or invalid.
    """
    path = folder / RIG_FINGERPRINT_OUT
    try:
        return RigFingerprint.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError) as exc:
        raise StagePrereqError(
            f"{RIG_FINGERPRINT_OUT.as_posix()} is missing or invalid; run the export stage first",
            detail={"path": str(path)},
            cause=exc,
        ) from exc


# --------------------------------------------------------------------------- the stage
class HandoffStage:
    """``handoff``: see the module docstring.

    Args:
        now: Wall-clock source for ``manifest.generated_at`` (injected in tests).
    """

    name: ClassVar[StageName] = "handoff"

    def __init__(self, now: NowFn = utc_now) -> None:
        self.now = now

    def input_fingerprint(self, model: ModelContext, ctx: PipelineContext) -> str:
        """``[unity]``, every export output (bytes), the Unity import dir.

        Raises:
            StagePrereqError: The export report is missing.
        """
        del ctx
        cfg = model.config
        return sha_many(
            cfg_dump(cfg.unity),
            *(f"{model.relative(p)}:{file_sha(p)}" for p in export_outputs(model.paths.folder)),
            str(cfg.library.unity_import_dir or ""),
        )

    def run(self, model: ModelContext, ctx: PipelineContext) -> StageOutcome:
        """Build the manifest, stage ``output/<id>``, swap it in, sync to Unity.

        Raises:
            StagePrereqError: Prepare/export outputs are missing.
            HandoffError: A referenced file is missing, the manifest is invalid, or the sync
                failed (the previous ``output/<id>`` is left intact).
        """
        folder = model.paths.folder
        export = read_export_report(folder)
        apply = read_apply_material(folder)
        warnings: list[str] = []
        if apply is None:
            warnings.append(f"{APPLY_REPORT.as_posix()} not found; delivering without textures")
            model.log.warning(W_NO_TEXTURES, stage=self.name)
        maps, material = apply if apply is not None else ({}, UrpMaterialInputs())
        manifest = self.build(model, ctx.pipeline_version, export, maps, material)
        final = model.paths.output / model.model_id
        staging = model.paths.output / f"{model.model_id}.tmp-{ctx.run_id}"
        self._stage(staging, folder, export, maps, manifest)
        atomic_replace_dir(staging, final)
        synced = False
        unity_dir = model.config.library.unity_import_dir
        if unity_dir is not None:
            report = copy_model(final, unity_dir, model.model_id, run_token=ctx.run_id)
            synced = True
            model.log.info("handoff.synced", dest=str(unity_dir / model.model_id), **vars(report))
        files = sorted(p for p in final.rglob("*") if p.is_file())
        model.log.info("handoff.done", files=len(files), clips=len(manifest.animations))
        return StageOutcome(
            status="succeeded",
            outputs=[model.relative(p) for p in files],
            extra={
                "output_bytes": sum(p.stat().st_size for p in files),
                "synced": synced,
                "manifest": model.relative(final / MANIFEST_JSON),
            },
            warnings=warnings,
        )

    def build(
        self,
        model: ModelContext,
        pipeline_version: str,
        export: ExportReport,
        maps: dict[str, Path],
        material: UrpMaterialInputs,
    ) -> Manifest:
        """Assemble the manifest (plan 06 §10) from the reports and the resolved model."""
        folder = model.paths.folder
        front = model.resolved.doc.front
        cfg = model.config
        task_id, credits_used = read_texture_task(folder)
        unity_override = front.unity
        return build_manifest(
            model_id=model.model_id,
            display_name=front.display_name,
            pipeline_version=pipeline_version,
            generated_at=iso_utc(self.now()),
            fbx_name=f"{model.model_id}.fbx",
            root_bone=first_root_bone(read_output_rig(folder).bones),
            unity=cfg.unity,
            source_asset_path=unity_override.source_asset_path if unity_override else None,
            material=material,
            texture_files={name: f"{TEXTURES_SUBDIR}/{path.name}" for name, path in maps.items()},
            animations=[
                (f"{ANIMATIONS_SUBDIR}/{Path(clip.file).name}", clip.clip_name)
                for clip in export.clips
            ],
            provenance=ManifestProvenance(
                source_fbx_sha256=file_sha(model.primary_fbx),
                source_license=front.source.license or "",
                subdivision_levels=cfg.geometry.subdivision_levels,
                subdivision_type=SUBDIVISION_TYPE_NAMES[cfg.geometry.subdivision_type],
                uv_decision=read_uv_decision(folder),
                texture_provider=cfg.texture.provider,
                texture_task_id=task_id,
                texture_route=read_route(folder),
                texture_seed=cfg.texture.seed,
                prompt_sha256=model.composed_prompt.sha256,
                credits_consumed=credits_used,
                rig_fidelity=export.overall,
            ),
        )

    # ------------------------------------------------------------------ internals
    @staticmethod
    def _stage(
        staging: Path,
        folder: Path,
        export: ExportReport,
        maps: dict[str, Path],
        manifest: Manifest,
    ) -> None:
        """Copy every delivered file into ``staging`` and write the manifest; clean up on error."""
        shutil.rmtree(staging, ignore_errors=True)
        try:
            staging.mkdir(parents=True)
            _copy(folder / export.fbx, staging / manifest.fbx)
            for path in maps.values():  # the manifest's texture files are these, renamed
                _copy(path, staging / TEXTURES_SUBDIR / path.name)
            for clip, anim in zip(export.clips, manifest.animations, strict=True):
                _copy(folder / clip.file, staging / anim.file)
            write_manifest(manifest, staging / MANIFEST_JSON)
        except (OSError, HandoffError):
            shutil.rmtree(staging, ignore_errors=True)
            raise


def _copy(src: Path, dest: Path) -> None:
    if not src.is_file():
        raise HandoffError(
            f"delivered file is missing: {src}",
            detail={"path": str(src)},
            hint="re-run the export/apply stages",
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)


__all__ = [
    "ANIMATIONS_SUBDIR",
    "MANIFEST_JSON",
    "SUBDIVISION_TYPE_NAMES",
    "TEXTURES_SUBDIR",
    "W_NO_TEXTURES",
    "HandoffStage",
    "iso_utc",
    "read_apply_material",
    "read_output_rig",
    "read_route",
    "read_texture_task",
    "read_uv_decision",
    "utc_now",
]
