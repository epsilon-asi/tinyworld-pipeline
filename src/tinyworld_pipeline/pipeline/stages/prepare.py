"""Stage ``prepare`` (plan 09 §2.1, NORMATIVE): drive the ``prepare_model`` Blender job.

Builds :class:`PrepareModelParams` with absolute paths under ``work/``, runs the job through the
:class:`BlenderRunner` port, validates the returned data against the pydantic contract (drift →
``ProtocolDriftError``), cross-checks the ``static.glb`` sha and writes ``prepare_report.json``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import ValidationError
from ulid import ULID

from tinyworld_pipeline.blender.protocol import (
    JobRequest,
    JobResult,
    PrepareModelData,
    PrepareModelParams,
)
from tinyworld_pipeline.domain.errors import ProtocolDriftError
from tinyworld_pipeline.domain.state import StageName, StageOutcome
from tinyworld_pipeline.pipeline.atomic import atomic_write_json
from tinyworld_pipeline.pipeline.hashing import blender_jobs_code_sha, cfg_dump, file_sha, sha_many

if TYPE_CHECKING:
    from tinyworld_pipeline.pipeline.context import ModelContext, PipelineContext

PREPARED_BLEND = "prepared.blend"
STATIC_GLB = "static.glb"
PREPARE_REPORT = "prepare_report.json"
OUTPUTS = [f"work/{PREPARED_BLEND}", f"work/{STATIC_GLB}", f"work/{PREPARE_REPORT}"]
_HINT_SUFFIXES = frozenset({".png", ".jpg", ".jpeg"})


def palette_hint_textures(textures_dir: Path) -> list[Path]:
    """``source/textures/*.png|jpg|jpeg`` sorted by name (empty when the folder is absent)."""
    if not textures_dir.is_dir():
        return []
    return sorted(
        p for p in textures_dir.iterdir() if p.is_file() and p.suffix.lower() in _HINT_SUFFIXES
    )


def build_params(model: ModelContext) -> PrepareModelParams:
    """Job parameters for ``model`` (absolute paths, effective config sections)."""
    work = model.paths.work
    return PrepareModelParams(
        source_fbx=model.primary_fbx,
        out_blend=work / PREPARED_BLEND,
        out_static_glb=work / STATIC_GLB,
        out_report=work / PREPARE_REPORT,
        geometry=model.config.geometry,
        uv=model.config.uv,
        material_name=model.material_name,
        palette_hint_textures=palette_hint_textures(model.paths.textures),
    )


class PrepareStage:
    """``prepare``: ``work/prepared.blend`` + ``work/static.glb`` + ``work/prepare_report.json``."""

    name: ClassVar[StageName] = "prepare"

    def input_fingerprint(self, model: ModelContext, ctx: PipelineContext) -> str:
        """Primary FBX bytes, ``[geometry]``, ``[uv]``, ``export.fbx_scale_option``, job code."""
        del ctx
        cfg = model.config
        return sha_many(
            file_sha(model.primary_fbx),
            cfg_dump(cfg.geometry),
            cfg_dump(cfg.uv),
            cfg.export.fbx_scale_option,
            blender_jobs_code_sha(),
        )

    def run(self, model: ModelContext, ctx: PipelineContext) -> StageOutcome:
        """Run the Blender job and validate its outputs (see the module docstring).

        Raises:
            ProtocolDriftError: The job data does not match :class:`PrepareModelData` or the
                written ``static.glb`` does not hash to ``data.static_glb_sha256``.
            BlenderJobError: Propagated from the runner when the job reports ``ok=False``.
        """
        params = build_params(model)
        model.paths.work.mkdir(parents=True, exist_ok=True)
        request = JobRequest(
            job="prepare_model",
            job_id=str(ULID()),
            model_id=model.model_id,
            params=params.model_dump(mode="json"),
        )
        model.log.info("prepare.job.start", job_id=request.job_id, source=str(model.primary_fbx))
        result = ctx.blender.run(
            request,
            timeout_s=model.config.blender.job_timeout_s,
            log_path=model.paths.logs / f"prepare-{ctx.run_id}.log",
            cancel=ctx.cancel,
        )
        data = _validate_data(result)
        _cross_check_glb(params.out_static_glb, data)
        atomic_write_json(params.out_report, data)
        model.log.info(
            "prepare.job.done",
            job_id=request.job_id,
            triangles_after=data.mesh.triangles_after,
            uv_decision=data.uv.decision,
        )
        return StageOutcome(
            status="succeeded",
            outputs=list(OUTPUTS),
            extra={
                "triangles_after": data.mesh.triangles_after,
                "uv_decision": data.uv.decision,
                "bone_count": data.rig.bone_count,
            },
            warnings=list(result.warnings),
        )


def _validate_data(result: JobResult) -> PrepareModelData:
    try:
        return PrepareModelData.model_validate(result.data)
    except ValidationError as exc:
        detail: dict[str, Any] = {"job_id": result.job_id, "errors": exc.errors(include_url=False)}
        raise ProtocolDriftError(
            "prepare_model result data does not match PrepareModelData.",
            detail=detail,
            hint="Orchestrator and blender_jobs must come from the same install.",
            cause=exc,
        ) from exc


def _cross_check_glb(static_glb: Path, data: PrepareModelData) -> None:
    if not static_glb.is_file():
        raise ProtocolDriftError(
            "prepare_model reported success but wrote no static.glb.",
            detail={"path": str(static_glb)},
        )
    actual = file_sha(static_glb)
    if actual != data.static_glb_sha256:
        raise ProtocolDriftError(
            "static.glb on disk does not match the sha reported by prepare_model.",
            detail={"path": str(static_glb), "expected": data.static_glb_sha256, "actual": actual},
        )
