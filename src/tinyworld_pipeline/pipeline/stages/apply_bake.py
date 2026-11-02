"""Bake route of stage ``apply`` (plan 11 §4, §5.2): drive the Blender ``bake_textures`` job.

:func:`run_bake_route` is the production ``bake_route`` of :class:`~.apply.ApplyStage`. It builds
:class:`BakeTexturesParams` from ``work/prepared.blend``, ``work/tripo/result.glb`` and the
``bbox_diagonal`` recorded by ``prepare``, runs the job through the :class:`BlenderRunner` port,
validates the returned :class:`BakeTexturesData` and reshapes the baked maps into the
:class:`RawTextureSet` the URP packer expects:

* ``metallic.png`` + ``roughness.png`` → ``metallic_roughness.png`` (roughness → G, metallic → B)
  via :func:`merge_metallic_roughness`. When only one of them was baked the other channel is
  neutral (roughness missing → fully rough, metallic missing → dielectric).
* ``emission.png`` → ``emissive.png`` (the glTF/``RawTextureSet`` field name).
* Factors are identity: the bake folds the provider's factors into the pixels.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from pydantic import ValidationError
from ulid import ULID

from tinyworld_pipeline.blender.protocol import (
    BakeTexturesData,
    BakeTexturesParams,
    JobRequest,
    JobResult,
)
from tinyworld_pipeline.domain.errors import ProtocolDriftError, StagePrereqError
from tinyworld_pipeline.domain.textures import RawTextureSet, TextureFactors
from tinyworld_pipeline.pipeline.stages.verify import PREPARE_REPORT, RESULT_GLB
from tinyworld_pipeline.textures.image_io import image_size, write_png
from tinyworld_pipeline.textures.urp_pack import merge_metallic_roughness

if TYPE_CHECKING:
    from tinyworld_pipeline.pipeline.context import ModelContext, PipelineContext

PREPARED_BLEND = Path("work/prepared.blend")
RAW_DIR = Path("work/textures/raw")
MERGED_MAP = "metallic_roughness.png"
EMISSIVE_MAP = "emissive.png"
NEUTRAL_ROUGHNESS = 1.0  # fully rough when no roughness map was baked
NEUTRAL_METALLIC = 0.0  # dielectric when no metallic map was baked
EVENT_START = "apply.bake.start"
EVENT_DONE = "apply.bake.done"
EVENT_WARNING = "apply.bake.warning"


def read_bbox_diagonal(folder: Path) -> float:
    """``bbox_diagonal`` from ``work/prepare_report.json``.

    Raises:
        StagePrereqError: The report is missing, unreadable or lacks the value.
    """
    path = folder / PREPARE_REPORT
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StagePrereqError(
            f"{PREPARE_REPORT.as_posix()} is missing or unreadable; run the prepare stage first",
            detail={"path": str(path)},
            hint="gp run --model <id> --from-stage prepare",
            cause=exc,
        ) from exc
    value = data.get("bbox_diagonal") if isinstance(data, dict) else None
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        raise StagePrereqError(
            f"{PREPARE_REPORT.as_posix()} has no valid bbox_diagonal (got {value!r})",
            detail={"path": str(path)},
            hint="re-run the prepare stage",
        )
    return float(value)


def build_params(model: ModelContext) -> BakeTexturesParams:
    """Job parameters for ``model`` (absolute paths, effective ``[bake]`` section)."""
    folder = model.paths.folder
    return BakeTexturesParams(
        prepared_blend=folder / PREPARED_BLEND,
        source_glb=folder / RESULT_GLB,
        out_dir=folder / RAW_DIR,
        bake=model.config.bake,
        bbox_diagonal=read_bbox_diagonal(folder),
    )


def run_bake_route(model: ModelContext, ctx: PipelineContext) -> RawTextureSet:
    """Run ``bake_textures`` and return the raw texture set for URP packing.

    Raises:
        StagePrereqError: ``prepare_report.json`` is unusable.
        BlenderJobError: Propagated from the runner when the job reports ``ok=False``.
        ProtocolDriftError: The job data does not match :class:`BakeTexturesData` or a declared
            map file is missing.
    """
    params = build_params(model)
    params.out_dir.mkdir(parents=True, exist_ok=True)
    request = JobRequest(
        job="bake_textures",
        job_id=str(ULID()),
        model_id=model.model_id,
        params=params.model_dump(mode="json"),
    )
    model.log.info(EVENT_START, job_id=request.job_id, maps=list(params.bake.maps))
    result = ctx.blender.run(
        request,
        timeout_s=model.config.blender.job_timeout_s,
        log_path=model.paths.logs / f"apply-bake-{ctx.run_id}.log",
        cancel=ctx.cancel,
    )
    data = _validate_data(result)
    for warning in result.warnings:
        model.log.warning(EVENT_WARNING, job_id=request.job_id, warning=warning)
    raw = shape_raw_set(data.maps, params.out_dir)
    model.log.info(
        EVENT_DONE,
        job_id=request.job_id,
        baked=sorted(data.maps),
        bake_seconds=dict(data.bake_seconds),
        resolution=data.resolution,
    )
    return raw


def shape_raw_set(maps: dict[str, Path], raw_dir: Path) -> RawTextureSet:
    """Turn the job's per-map PNGs into the :class:`RawTextureSet` shape (see module docstring).

    Raises:
        ProtocolDriftError: A declared map file does not exist.
    """
    missing = sorted(name for name, path in maps.items() if not path.is_file())
    if missing:
        raise ProtocolDriftError(
            "bake_textures reported maps that do not exist on disk.",
            detail={"missing": missing, "maps": {k: str(v) for k, v in maps.items()}},
        )
    emissive = maps.get("emission")
    if emissive is not None and emissive.name != EMISSIVE_MAP:
        emissive = emissive.replace(raw_dir / EMISSIVE_MAP)
    return RawTextureSet(
        basecolor=maps.get("basecolor"),
        metallic_roughness=merge_bake_maps(maps.get("metallic"), maps.get("roughness"), raw_dir),
        normal=maps.get("normal"),
        occlusion=maps.get("occlusion"),
        emissive=emissive,
        factors=TextureFactors(),
        source="bake",
    )


def merge_bake_maps(metallic: Path | None, roughness: Path | None, raw_dir: Path) -> Path | None:
    """``metallic_roughness.png`` from the baked greyscale maps; ``None`` when neither exists.

    A missing partner is replaced by a neutral constant of the other map's size
    (:data:`NEUTRAL_ROUGHNESS` / :data:`NEUTRAL_METALLIC`).
    """
    if metallic is None and roughness is None:
        return None
    temp: Path | None = None
    if metallic is None:
        assert roughness is not None  # guarded by the early return above
        metallic = temp = _neutral_png(
            raw_dir / "_neutral_metallic.png", roughness, NEUTRAL_METALLIC
        )
    elif roughness is None:
        roughness = temp = _neutral_png(
            raw_dir / "_neutral_roughness.png", metallic, NEUTRAL_ROUGHNESS
        )
    try:
        return merge_metallic_roughness(metallic, roughness, raw_dir / MERGED_MAP)
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)


def _neutral_png(path: Path, like: Path, value: float) -> Path:
    width, height = image_size(like)
    pixels = np.full((height, width, 4), np.float32(value), dtype=np.float32)
    pixels[..., 3] = np.float32(1.0)
    return write_png(path, pixels, channels=4)


def _validate_data(result: JobResult) -> BakeTexturesData:
    try:
        return BakeTexturesData.model_validate(result.data)
    except ValidationError as exc:
        detail: dict[str, Any] = {"job_id": result.job_id, "errors": exc.errors(include_url=False)}
        raise ProtocolDriftError(
            "bake_textures result data does not match BakeTexturesData.",
            detail=detail,
            hint="Orchestrator and blender_jobs must come from the same install.",
            cause=exc,
        ) from exc


__all__ = [
    "EMISSIVE_MAP",
    "MERGED_MAP",
    "PREPARED_BLEND",
    "RAW_DIR",
    "build_params",
    "merge_bake_maps",
    "read_bbox_diagonal",
    "run_bake_route",
    "shape_raw_set",
]
