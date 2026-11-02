"""Stage ``export`` (plan 12 §1.1, NORMATIVE): Unity-bound FBX, companion round-trips, RFC.

1. ``export_rigged`` job: ``work/prepared.blend`` (+ ``work/textures/urp``) →
   ``work/export/<id>.fbx`` and ``work/export/rig_fingerprint_out.json``;
2. per companion clip (``cfg.export.roundtrip_companion_animations``): ``roundtrip_animation``
   job → ``work/export/animations/<id>@<clip>.fbx`` (clip naming: :func:`clip_name`);
3. :func:`rig_fidelity_check` in ``model`` mode for the export and ``animation`` mode per clip;
4. ``work/export/rig_fidelity_report.json`` (:class:`ExportReport`); any ``fail`` raises
   :class:`RigFidelityError` *after* the report is written.

Fingerprint hashes inputs only (appendix B §8.5: exported FBX bytes are never deterministic):
``prepared.blend``, every URP texture, every companion FBX (name + bytes), ``[export]``,
``geometry.include_unskinned`` and the Blender-side code.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import Field, ValidationError
from ulid import ULID

from grounded_pipeline.blender.protocol import (
    ExportRiggedData,
    ExportRiggedParams,
    JobName,
    JobRequest,
    JobResult,
    RigFingerprint,
    RoundtripAnimationData,
    RoundtripAnimationParams,
)
from grounded_pipeline.domain._base import FrozenModel
from grounded_pipeline.domain.errors import (
    AnimNameCollisionError,
    ProtocolDriftError,
    RigFidelityError,
    StagePrereqError,
)
from grounded_pipeline.domain.state import StageName, StageOutcome
from grounded_pipeline.pipeline.atomic import atomic_write_json
from grounded_pipeline.pipeline.hashing import blender_jobs_code_sha, cfg_dump, file_sha, sha_many
from grounded_pipeline.pipeline.rig_fidelity import (
    Overall,
    RfcTolerances,
    RigFidelityReport,
    rig_fidelity_check,
)
from grounded_pipeline.pipeline.stages.apply import URP_DIR
from grounded_pipeline.pipeline.stages.prepare import PREPARE_REPORT, PREPARED_BLEND

if TYPE_CHECKING:
    from grounded_pipeline.pipeline.context import ModelContext, PipelineContext

EXPORT_DIR = Path("work/export")
ANIMATIONS_DIR = EXPORT_DIR / "animations"
RIG_FINGERPRINT_OUT = EXPORT_DIR / "rig_fingerprint_out.json"
RIG_FIDELITY_REPORT = EXPORT_DIR / "rig_fidelity_report.json"
_CLIP_SEPARATORS = "@_- ."
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


# --------------------------------------------------------------------------- report document
class ClipRecord(FrozenModel):
    """One round-tripped companion clip in ``rig_fidelity_report.json``."""

    clip_name: str
    file: str  # relative to the model folder: "work/export/animations/<id>@<clip>.fbx"
    source: str  # relative to the model folder (file name only if it lives elsewhere)
    actions_exported: list[str]
    missing_bones: list[str]
    extra_bones: list[str]
    report: RigFidelityReport


class ExportReport(FrozenModel):
    """``work/export/rig_fidelity_report.json`` — what ``handoff`` reads back."""

    overall: Overall
    fbx: str  # relative to the model folder: "work/export/<id>.fbx"
    fbx_sha256: str
    actions_exported: list[str]
    tolerances: dict[str, Any] = Field(default_factory=dict)
    model: RigFidelityReport
    clips: list[ClipRecord] = Field(default_factory=list)


# --------------------------------------------------------------------------- pure helpers
def clip_name(stem: str, prefixes: tuple[str, ...]) -> str:
    """``<clip>`` for a companion file stem (plan 12 §1.1).

    The longest matching model-name prefix (case-insensitive, followed by an optional
    separator) is stripped, the rest is lower-cased and every run of non-alphanumerics becomes
    one ``_``: ``Ant_Walk`` → ``walk``, ``cube_rigged@walk`` → ``walk``, ``Walk`` → ``walk``.
    A stem that is nothing but the prefix keeps its full sanitised name.
    """
    lowered = stem.lower()
    rest = stem
    for prefix in sorted({p.lower() for p in prefixes if p}, key=len, reverse=True):
        if lowered.startswith(prefix) and len(lowered) > len(prefix):
            rest = stem[len(prefix) :].lstrip(_CLIP_SEPARATORS)
            break
    name = _NON_ALNUM.sub("_", rest.lower()).strip("_")
    return name or _NON_ALNUM.sub("_", lowered).strip("_") or "clip"


def clip_names(files: list[Path], prefixes: tuple[str, ...]) -> list[tuple[Path, str]]:
    """``(file, clip)`` pairs in input order.

    Raises:
        AnimNameCollisionError: Two files map to the same clip name (``E_ANIM_NAME_COLLISION``).
    """
    pairs = [(path, clip_name(path.stem, prefixes)) for path in files]
    by_name: dict[str, list[str]] = {}
    for path, name in pairs:
        by_name.setdefault(name, []).append(path.name)
    collisions = {name: names for name, names in by_name.items() if len(names) > 1}
    if collisions:
        first = next(iter(collisions))
        raise AnimNameCollisionError(
            f"companion animations {collisions[first]} all map to clip name {first!r}",
            detail={"collisions": collisions},
            hint="rename the companion files so their stems differ after the model prefix",
        )
    return pairs


def urp_textures(folder: Path) -> list[Path]:
    """PNGs in ``work/textures/urp`` sorted by name (empty when the folder is absent)."""
    urp = folder / URP_DIR
    if not urp.is_dir():
        return []
    return sorted(p for p in urp.iterdir() if p.is_file() and p.suffix.lower() == ".png")


def companion_files(model: ModelContext) -> list[Path]:
    """Companion clips to round-trip (none when the config switches round-tripping off)."""
    if not model.config.export.roundtrip_companion_animations:
        return []
    return sorted(model.animation_fbx, key=lambda p: p.name)


def source_label(model: ModelContext, path: Path) -> str:
    """``path`` relative to the model folder, or just its name when it lives elsewhere."""
    try:
        return model.relative(path)
    except ValueError:
        return path.name


def read_source_rig(folder: Path) -> RigFingerprint:
    """``prepare_report.rig`` (the vendor rig as imported).

    Raises:
        StagePrereqError: The report is missing or carries no valid rig.
    """
    path = folder / "work" / PREPARE_REPORT
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return RigFingerprint.model_validate(data["rig"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValidationError) as exc:
        raise StagePrereqError(
            f"work/{PREPARE_REPORT} is missing or has no valid rig; run the prepare stage first",
            detail={"path": str(path)},
            hint="gp run --model <id> --from-stage prepare --to-stage prepare",
            cause=exc,
        ) from exc


def export_outputs(folder: Path) -> list[Path]:
    """Every file the export stage produced, read from its report (for ``handoff``).

    Raises:
        StagePrereqError: The report is missing or unreadable.
    """
    report = read_export_report(folder)
    files = [folder / report.fbx, *(folder / clip.file for clip in report.clips)]
    return [*files, folder / RIG_FINGERPRINT_OUT, folder / RIG_FIDELITY_REPORT]


def read_export_report(folder: Path) -> ExportReport:
    """Parse ``work/export/rig_fidelity_report.json``.

    Raises:
        StagePrereqError: Missing or invalid.
    """
    path = folder / RIG_FIDELITY_REPORT
    try:
        return ExportReport.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError) as exc:
        raise StagePrereqError(
            f"{RIG_FIDELITY_REPORT.as_posix()} is missing or invalid; run the export stage first",
            detail={"path": str(path)},
            hint="gp run --model <id> --from-stage export --to-stage export",
            cause=exc,
        ) from exc


# --------------------------------------------------------------------------- the stage
class ExportStage:
    """``export``: see the module docstring.

    Args:
        tolerances: RFC tolerances (normative defaults).
    """

    name: ClassVar[StageName] = "export"

    def __init__(self, tolerances: RfcTolerances | None = None) -> None:
        self.tolerances = tolerances or RfcTolerances()

    def input_fingerprint(self, model: ModelContext, ctx: PipelineContext) -> str:
        """Inputs only; see the module docstring.

        Raises:
            StagePrereqError: ``work/prepared.blend`` does not exist.
        """
        del ctx
        folder = model.paths.folder
        blend = model.paths.work / PREPARED_BLEND
        if not blend.is_file():
            raise StagePrereqError(
                f"work/{PREPARED_BLEND} missing for {model.model_id}; run the prepare stage first",
                detail={"path": str(blend)},
            )
        return sha_many(
            file_sha(blend),
            *(f"{p.name}:{file_sha(p)}" for p in urp_textures(folder)),
            *(f"{source_label(model, p)}:{file_sha(p)}" for p in companion_files(model)),
            cfg_dump(model.config.export),
            f"include_unskinned={model.config.geometry.include_unskinned}",
            blender_jobs_code_sha(),
        )

    def run(self, model: ModelContext, ctx: PipelineContext) -> StageOutcome:
        """Export, round-trip, check, report (see the module docstring).

        Raises:
            StagePrereqError: Prepare outputs are missing.
            AnimNameCollisionError: Two companion files share a clip name.
            RigFidelityError: Any RFC check failed (the report is written first).
            ProtocolDriftError: A job result does not match its contract or its file on disk.
            BlenderJobError: Propagated from the runner.
        """
        folder = model.paths.folder
        source_rig = read_source_rig(folder)
        clips = clip_names(companion_files(model), (model.model_id, model.primary_fbx.stem))
        export_dir = folder / EXPORT_DIR
        shutil.rmtree(export_dir, ignore_errors=True)
        export_dir.mkdir(parents=True, exist_ok=True)
        log_path = model.paths.logs / f"export-{ctx.run_id}.log"

        params = self.export_params(model)
        data = self._export_model(model, ctx, params, log_path)
        atomic_write_json(params.out_rig_fingerprint, data.rig)
        model_report = rig_fidelity_check(source_rig, data.rig, self.tolerances, "model")
        records = [
            self._roundtrip(model, ctx, source_rig, src, clip, log_path) for src, clip in clips
        ]
        report = ExportReport(
            overall=_rollup([model_report, *(r.report for r in records)]),
            fbx=model.relative(params.out_fbx),
            fbx_sha256=data.fbx_sha256,
            actions_exported=list(data.actions_exported),
            tolerances=vars(self.tolerances),
            model=model_report,
            clips=records,
        )
        atomic_write_json(folder / RIG_FIDELITY_REPORT, report)
        return self._outcome(model, report)

    def export_params(self, model: ModelContext) -> ExportRiggedParams:
        """Job parameters for ``export_rigged`` (absolute paths under ``work/``)."""
        folder = model.paths.folder
        return ExportRiggedParams(
            prepared_blend=model.paths.work / PREPARED_BLEND,
            textures_dir=(folder / URP_DIR) if urp_textures(folder) else None,
            out_fbx=folder / EXPORT_DIR / f"{model.model_id}.fbx",
            out_rig_fingerprint=folder / RIG_FINGERPRINT_OUT,
            export=model.config.export,
            material_name=model.material_name,
            include_unskinned=model.config.geometry.include_unskinned,
        )

    # ------------------------------------------------------------------ internals
    def _export_model(
        self, model: ModelContext, ctx: PipelineContext, params: ExportRiggedParams, log: Path
    ) -> ExportRiggedData:
        result = _run_job(model, ctx, "export_rigged", params.model_dump(mode="json"), log)
        data = _validate(result, ExportRiggedData)
        _cross_check_fbx(params.out_fbx, data.fbx_sha256)
        model.log.info(
            "export.model.done",
            job_id=result.job_id,
            actions=data.actions_exported,
            warnings=result.warnings,
        )
        return data

    def _roundtrip(  # noqa: PLR0917 - internal step with explicit inputs
        self,
        model: ModelContext,
        ctx: PipelineContext,
        source_rig: RigFingerprint,
        source: Path,
        clip: str,
        log: Path,
    ) -> ClipRecord:
        out = model.paths.folder / ANIMATIONS_DIR / f"{model.model_id}@{clip}.fbx"
        params = RoundtripAnimationParams(
            source_fbx=source,
            out_fbx=out,
            export=model.config.export,
            expected_bone_names=[b.name for b in source_rig.bones],
        )
        result = _run_job(model, ctx, "roundtrip_animation", params.model_dump(mode="json"), log)
        data = _validate(result, RoundtripAnimationData)
        if not out.is_file():
            raise ProtocolDriftError(
                "roundtrip_animation reported success but wrote no FBX.", detail={"path": str(out)}
            )
        report = rig_fidelity_check(source_rig, data.rig, self.tolerances, "animation")
        model.log.info("export.clip.done", clip=clip, source=source.name, rfc=report.overall)
        return ClipRecord(
            clip_name=clip,
            file=model.relative(out),
            source=source_label(model, source),
            actions_exported=list(data.actions_exported),
            missing_bones=list(data.missing_bones),
            extra_bones=list(data.extra_bones),
            report=report,
        )

    def _outcome(self, model: ModelContext, report: ExportReport) -> StageOutcome:
        failures = [f"model: {m}" for m in report.model.messages("fail")]
        warnings = [f"model: {m}" for m in report.model.messages("warn")]
        for clip in report.clips:
            failures.extend(f"{clip.clip_name}: {m}" for m in clip.report.messages("fail"))
            warnings.extend(f"{clip.clip_name}: {m}" for m in clip.report.messages("warn"))
        if failures:
            raise RigFidelityError(
                f"rig fidelity check failed for {model.model_id}: {failures[0]}",
                detail={"failures": failures, "report": RIG_FIDELITY_REPORT.as_posix()},
                hint="compare work/export/rig_fidelity_report.json with work/prepare_report.json",
            )
        for warning in warnings:
            model.log.warning("export.rfc.warning", stage=self.name, detail=warning)
        outputs = [
            report.fbx,
            *(clip.file for clip in report.clips),
            RIG_FINGERPRINT_OUT.as_posix(),
            RIG_FIDELITY_REPORT.as_posix(),
        ]
        return StageOutcome(
            status="succeeded",
            outputs=outputs,
            extra={
                "rig_fidelity": report.overall,
                "clips": [clip.clip_name for clip in report.clips],
                "actions_exported": list(report.actions_exported),
            },
            warnings=warnings,
        )


# --------------------------------------------------------------------------- job plumbing
def _run_job(
    model: ModelContext, ctx: PipelineContext, job: JobName, params: dict[str, Any], log: Path
) -> JobResult:
    request = JobRequest(job=job, job_id=str(ULID()), model_id=model.model_id, params=params)
    model.log.info("export.job.start", job=job, job_id=request.job_id)
    return ctx.blender.run(
        request, timeout_s=model.config.blender.job_timeout_s, log_path=log, cancel=ctx.cancel
    )


def _validate[T: FrozenModel](result: JobResult, cls: type[T]) -> T:
    try:
        return cls.model_validate(result.data)
    except ValidationError as exc:
        raise ProtocolDriftError(
            f"{result.job} result data does not match {cls.__name__}.",
            detail={"job_id": result.job_id, "errors": exc.errors(include_url=False)},
            hint="Orchestrator and blender_jobs must come from the same install.",
            cause=exc,
        ) from exc


def _cross_check_fbx(path: Path, expected_sha: str) -> None:
    if not path.is_file():
        raise ProtocolDriftError(
            "export_rigged reported success but wrote no FBX.", detail={"path": str(path)}
        )
    actual = file_sha(path)
    if actual != expected_sha:
        raise ProtocolDriftError(
            "exported FBX on disk does not match the sha reported by export_rigged.",
            detail={"path": str(path), "expected": expected_sha, "actual": actual},
        )


def _rollup(reports: list[RigFidelityReport]) -> Overall:
    verdicts = {r.overall for r in reports}
    if "fail" in verdicts:
        return "fail"
    if "pass_with_warnings" in verdicts:
        return "pass_with_warnings"
    return "pass"


__all__ = [
    "ANIMATIONS_DIR",
    "EXPORT_DIR",
    "RIG_FIDELITY_REPORT",
    "RIG_FINGERPRINT_OUT",
    "ClipRecord",
    "ExportReport",
    "ExportStage",
    "clip_name",
    "clip_names",
    "companion_files",
    "export_outputs",
    "read_export_report",
    "read_source_rig",
    "source_label",
    "urp_textures",
]
