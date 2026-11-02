"""Stage ``verify`` (plan 11 §3): decide between the extract and bake routes.

Inputs: ``work/static.glb`` (ours), ``work/tripo/result.glb`` (theirs), the ``[verify]`` section
with ``--route`` applied, plus two optional helpers that are tolerated when absent:
``work/prepare_report.json`` (self-check counts) and ``work/tripo/task.json`` (the provider's
``normalize_scale`` / ``normalize_translation``). Output: ``work/verify_report.json``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tinyworld_pipeline.domain.library_config import VerifySection
from tinyworld_pipeline.domain.state import StageName, StageOutcome
from tinyworld_pipeline.gltf.fingerprint import Float3
from tinyworld_pipeline.gltf.verify import verify_surfaces
from tinyworld_pipeline.pipeline.atomic import atomic_write_json
from tinyworld_pipeline.pipeline.context import ModelContext, PipelineContext
from tinyworld_pipeline.pipeline.hashing import cfg_dump, file_sha, sha_many

STATIC_GLB = Path("work/static.glb")
RESULT_GLB = Path("work/tripo/result.glb")
PREPARE_REPORT = Path("work/prepare_report.json")
TASK_JSON = Path("work/tripo/task.json")
VERIFY_REPORT = Path("work/verify_report.json")
_VEC3 = 3


def effective_verify_section(model: ModelContext, ctx: PipelineContext) -> VerifySection:
    """``model.config.verify`` with ``--route`` (``options.route_override``) applied."""
    cfg = model.config.verify
    override = ctx.options.route_override
    if override is not None and override != cfg.route:
        cfg = cfg.model_copy(update={"route": override})
    return cfg


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _int_or_none(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def read_prepare_counts(path: Path) -> tuple[int | None, int | None]:
    """``(static_glb_triangles, static_glb_unique_positions)`` from the prepare report, if any."""
    data = _read_json(path)
    if data is None:
        return None, None
    return (
        _int_or_none(data.get("static_glb_triangles")),
        _int_or_none(data.get("static_glb_unique_positions")),
    )


def read_normalisation(path: Path) -> tuple[float | None, Float3 | None]:
    """Provider ``normalize_scale`` / ``normalize_translation`` from ``task.json`` (tolerant)."""
    data = _read_json(path)
    if data is None:
        return None, None
    raw = data.get("raw_output")
    output = raw.get("output") if isinstance(raw, dict) else None
    if not isinstance(output, dict):
        return None, None
    scale_raw = output.get("normalize_scale")
    scale = float(scale_raw) if isinstance(scale_raw, int | float) else None
    trans_raw = output.get("normalize_translation")
    translation: Float3 | None = None
    if (
        isinstance(trans_raw, list | tuple)
        and len(trans_raw) == _VEC3
        and all(isinstance(v, int | float) for v in trans_raw)
    ):
        translation = (float(trans_raw[0]), float(trans_raw[1]), float(trans_raw[2]))
    return scale, translation


class VerifyStage:
    """Fingerprint = sha(static.glb) + sha(result.glb) + ``cfg_dump(verify)``."""

    name: StageName = "verify"

    def input_fingerprint(self, model: ModelContext, ctx: PipelineContext) -> str:
        folder = model.paths.folder
        return sha_many(
            file_sha(folder / STATIC_GLB),
            file_sha(folder / RESULT_GLB),
            cfg_dump(effective_verify_section(model, ctx)),
        )

    def run(self, model: ModelContext, ctx: PipelineContext) -> StageOutcome:
        """Compare both GLBs, write ``work/verify_report.json`` and report the route."""
        folder = model.paths.folder
        cfg = effective_verify_section(model, ctx)
        triangles, unique_positions = read_prepare_counts(folder / PREPARE_REPORT)
        scale, translation = read_normalisation(folder / TASK_JSON)
        report = verify_surfaces(
            folder / STATIC_GLB,
            folder / RESULT_GLB,
            cfg,
            expected_triangles=triangles,
            expected_unique_positions=unique_positions,
            normalize_scale=scale,
            normalize_translation=translation,
        )
        atomic_write_json(folder / VERIFY_REPORT, report)
        model.log.info(
            "verify_decided",
            route=report.route,
            forced=report.forced,
            position_match=report.position_match,
            uv_match=report.uv_match,
            reason=report.reason,
        )
        return StageOutcome(
            status="succeeded",
            outputs=[VERIFY_REPORT.as_posix()],
            extra={
                "route": report.route,
                "position_match": report.position_match,
                "uv_match": report.uv_match,
                "topology_match": report.topology_match,
            },
        )


__all__ = [
    "PREPARE_REPORT",
    "RESULT_GLB",
    "STATIC_GLB",
    "TASK_JSON",
    "VERIFY_REPORT",
    "VerifyStage",
    "effective_verify_section",
    "read_normalisation",
    "read_prepare_counts",
]
