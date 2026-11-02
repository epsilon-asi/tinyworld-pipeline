"""Stage ``apply`` (plan 11 §4): turn the provider result into URP-packed textures.

* ``extract`` route — :func:`extract_textures` on ``work/tripo/result.glb`` → ``work/textures/raw``.
  A :class:`MultiMaterialResultError` is not a failure: the stage logs ``W_EXTRACT_FALLBACK_BAKE``
  and takes the bake route (``apply_report.fallback_to_bake = true``).
* ``bake`` route — delegated to a constructor-injected ``bake_route(model, ctx) -> RawTextureSet``;
  the default is ``apply_bake.run_bake_route`` (Blender job ``bake_textures``, plan 11 §5.2).

Both routes end in :func:`pack_urp` → ``work/textures/urp/<id>_*.png`` and
``work/apply_report.json`` (plan 11 §7).
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

from grounded_pipeline.domain.errors import MultiMaterialResultError, StagePrereqError
from grounded_pipeline.domain.state import StageName, StageOutcome
from grounded_pipeline.domain.textures import RawTextureSet
from grounded_pipeline.gltf.extract import extract_textures
from grounded_pipeline.gltf.loader import load_glb
from grounded_pipeline.pipeline.atomic import atomic_write_json
from grounded_pipeline.pipeline.context import ModelContext, PipelineContext
from grounded_pipeline.pipeline.hashing import blender_jobs_code_sha, cfg_dump, file_sha, sha_many
from grounded_pipeline.pipeline.stages.apply_bake import run_bake_route
from grounded_pipeline.pipeline.stages.verify import RESULT_GLB, VERIFY_REPORT
from grounded_pipeline.textures.urp_pack import UrpTextureSet, pack_urp

Route = Literal["extract", "bake"]
BakeRoute = Callable[[ModelContext, PipelineContext], RawTextureSet]

PREPARED_BLEND = Path("work/prepared.blend")
RAW_DIR = Path("work/textures/raw")
URP_DIR = Path("work/textures/urp")
APPLY_REPORT = Path("work/apply_report.json")
W_EXTRACT_FALLBACK_BAKE = "W_EXTRACT_FALLBACK_BAKE"


def bake_route_not_implemented(model: ModelContext, ctx: PipelineContext) -> RawTextureSet:
    """Placeholder bake route kept for tests that must prove the seam is used."""
    del model, ctx
    raise NotImplementedError("bake route lands in WP-08")


def read_route(folder: Path) -> Route:
    """The route decided by ``verify`` (``work/verify_report.json``).

    Raises:
        StagePrereqError: The report is missing or has no valid ``route``.
    """
    path = folder / VERIFY_REPORT
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StagePrereqError(
            f"{VERIFY_REPORT.as_posix()} is missing or unreadable; run the verify stage first",
            detail={"path": str(path)},
            hint="gp run --model <id> --from-stage verify --to-stage verify",
            cause=exc,
        ) from exc
    route = data.get("route") if isinstance(data, dict) else None
    if route not in ("extract", "bake"):
        raise StagePrereqError(
            f"{VERIFY_REPORT.as_posix()} has no valid route (got {route!r})",
            detail={"path": str(path)},
            hint="re-run the verify stage",
        )
    return "extract" if route == "extract" else "bake"


def _reset_dir(path: Path) -> Path:
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _rel_maps(model: ModelContext, maps: dict[str, Path | None]) -> dict[str, str]:
    return {k: model.relative(v) for k, v in maps.items() if v is not None}


class ApplyStage:
    """Fingerprint = route + sha(result.glb) [+ ``cfg_dump(bake)`` + sha(prepared.blend) for bake].

    Args:
        bake_route: Produces the raw texture set for the bake route (WP-08's ``run_bake_route``).
    """

    name: StageName = "apply"

    def __init__(self, bake_route: BakeRoute | None = None) -> None:
        self.bake_route: BakeRoute = bake_route or run_bake_route

    def input_fingerprint(self, model: ModelContext, ctx: PipelineContext) -> str:
        del ctx
        folder = model.paths.folder
        route = read_route(folder)
        parts = [f"route={route}", file_sha(folder / RESULT_GLB)]
        if route == "bake":
            parts.extend(
                [
                    cfg_dump(model.config.bake),
                    file_sha(folder / PREPARED_BLEND),
                    blender_jobs_code_sha(),  # bake_textures.py changes must re-run the bake
                ]
            )
        return sha_many(*parts)

    def run(self, model: ModelContext, ctx: PipelineContext) -> StageOutcome:
        """Extract or bake, pack for URP, write ``work/apply_report.json``."""
        folder = model.paths.folder
        route = read_route(folder)
        raw_dir = _reset_dir(folder / RAW_DIR)
        fallback = False
        raw: RawTextureSet | None = None
        if route == "extract":
            try:
                raw = extract_textures(load_glb(folder / RESULT_GLB), raw_dir)
            except MultiMaterialResultError as exc:
                model.log.warning(
                    W_EXTRACT_FALLBACK_BAKE, error=exc.message, detail=exc.detail, stage=self.name
                )
                route, fallback = "bake", True
                _reset_dir(raw_dir)
        if raw is None:
            raw = self.bake_route(model, ctx)
        urp = pack_urp(raw, _reset_dir(folder / URP_DIR), model.model_id)
        report = self._report(model, route, raw, urp, fallback=fallback)
        atomic_write_json(folder / APPLY_REPORT, report)
        outputs = [*sorted(report["urp"].values()), APPLY_REPORT.as_posix()]
        model.log.info(
            "apply_packed", route=route, fallback_to_bake=fallback, maps=sorted(urp.maps())
        )
        return StageOutcome(
            status="succeeded",
            outputs=outputs,
            extra={
                "route": route,
                "fallback_to_bake": fallback,
                "resolution": list(urp.resolution) if urp.resolution else None,
            },
        )

    @staticmethod
    def _report(
        model: ModelContext,
        route: Route,
        raw: RawTextureSet,
        urp: UrpTextureSet,
        *,
        fallback: bool,
    ) -> dict[str, Any]:
        raw_maps: dict[str, Path | None] = {
            "basecolor": raw.basecolor,
            "metallic_roughness": raw.metallic_roughness,
            "normal": raw.normal,
            "occlusion": raw.occlusion,
            "emissive": raw.emissive,
        }
        urp_maps: dict[str, Path | None] = dict(urp.maps())
        return {
            "route": route,
            "raw": _rel_maps(model, raw_maps),
            "urp": _rel_maps(model, urp_maps),
            "factors": asdict(raw.factors),
            "resolution": list(urp.resolution) if urp.resolution else None,
            "fallback_to_bake": fallback,
            "source": raw.source,
            "material": {
                "bump_scale": urp.bump_scale,
                "occlusion_strength": urp.occlusion_strength,
                "emission_color": list(urp.emission_color),
                "metallic_fallback": urp.metallic_fallback,
                "smoothness_fallback": urp.smoothness_fallback,
            },
        }


__all__ = [
    "APPLY_REPORT",
    "PREPARED_BLEND",
    "RAW_DIR",
    "URP_DIR",
    "W_EXTRACT_FALLBACK_BAKE",
    "ApplyStage",
    "BakeRoute",
    "bake_route_not_implemented",
    "read_route",
]
