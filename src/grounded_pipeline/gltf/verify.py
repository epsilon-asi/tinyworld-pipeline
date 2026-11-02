"""Route decision between texture extraction and re-baking (plan 06 §6, 11 §2-§3).

:func:`decide_route` is the pure decision function; :func:`verify_surfaces` is the file-reading
orchestration used by the ``verify`` stage: load both GLBs, bring the provider's result back into
our frame, fingerprint, self-check against the ``prepare`` report and decide.
"""

from __future__ import annotations

from pathlib import Path

from grounded_pipeline.domain.errors import GeometryChangedError, VerifySelfMismatchError
from grounded_pipeline.domain.fingerprint import SurfaceFingerprint, VerifyReport
from grounded_pipeline.domain.library_config import VerifySection
from grounded_pipeline.gltf.fingerprint import (
    Float3,
    fit_affine,
    normalise_positions,
    surface_fingerprint,
)
from grounded_pipeline.gltf.loader import GlbDocument, load_glb


def decide_route(
    ours: SurfaceFingerprint, theirs: SurfaceFingerprint, cfg: VerifySection
) -> VerifyReport:
    """Decide whether provider textures can be extracted as-is or must be baked (plan 06 §6).

    Args:
        ours: Fingerprint of ``work/static.glb``.
        theirs: Fingerprint of the provider result, already normalised into our frame.
        cfg: ``[verify]`` section.

    Returns:
        The report; ``forced`` is set when ``cfg.route`` is not ``"auto"``.

    Raises:
        GeometryChangedError: Neither positions nor topology match and
            ``cfg.allow_geometry_change`` is false.
    """
    topology_match = (
        ours.triangle_count == theirs.triangle_count
        and ours.unique_position_count == theirs.unique_position_count
    )
    position_match = ours.position_set_sha256 == theirs.position_set_sha256
    uv_match = ours.surface_sha256 == theirs.surface_sha256
    forced = False
    if cfg.route != "auto":
        route, forced = cfg.route, True
        reason = f"route forced by config (verify.route = {cfg.route!r})"
    elif uv_match and position_match:
        route, reason = "extract", "positions and UVs identical within quantum"
    elif position_match:
        route, reason = "bake", "positions identical but UVs differ (re-unwrapped)"
    elif topology_match:
        route, reason = "bake", "topology identical but positions differ"
    elif cfg.allow_geometry_change:
        route, reason = "bake", "geometry changed; baking allowed by verify.allow_geometry_change"
    else:
        raise GeometryChangedError(
            "provider result does not match the uploaded mesh: "
            f"ours {ours.triangle_count} tris / {ours.unique_position_count} unique positions, "
            f"theirs {theirs.triangle_count} tris / {theirs.unique_position_count}",
            detail={"ours": ours.model_dump(mode="json"), "theirs": theirs.model_dump(mode="json")},
            hint=(
                "set [verify] allow_geometry_change = true to bake the result onto the original "
                'mesh, or route = "bake" to force baking'
            ),
        )
    return VerifyReport(
        ours=ours,
        theirs=theirs,
        topology_match=topology_match,
        position_match=position_match,
        uv_match=uv_match,
        route=route,
        reason=reason,
        forced=forced,
    )


def verify_surfaces(
    static_glb: Path,
    result_glb: Path,
    cfg: VerifySection,
    *,
    expected_triangles: int | None,
    expected_unique_positions: int | None,
    normalize_scale: float | None,
    normalize_translation: Float3 | None,
) -> VerifyReport:
    """Load, normalise, fingerprint, self-check and decide (plan 11 §3).

    Args:
        static_glb: Our uploaded mesh (``work/static.glb``).
        result_glb: The provider's textured result.
        cfg: ``[verify]`` section.
        expected_triangles: ``prepare_report.static_glb_triangles`` or ``None`` to skip the check.
        expected_unique_positions: ``prepare_report.static_glb_unique_positions`` or ``None``.
        normalize_scale: Provider-reported ``normalize_scale`` (``None`` → bounding-box fit).
        normalize_translation: Provider-reported ``normalize_translation`` or ``None``.

    Returns:
        The verify report; ``reason`` ends with ``"; normalised scale=… source=task|bbox|none"``.

    Raises:
        VerifySelfMismatchError: Our own fingerprint disagrees with the ``prepare`` report.
        GeometryChangedError: See :func:`decide_route`.
        UnsupportedGlbError: Either file cannot be loaded.
    """
    ours_doc = load_glb(static_glb)
    theirs_doc, note = _normalise(
        ours_doc, load_glb(result_glb), normalize_scale, normalize_translation
    )
    ours = surface_fingerprint(ours_doc, q_pos=cfg.position_quantum, q_uv=cfg.uv_quantum)
    theirs = surface_fingerprint(theirs_doc, q_pos=cfg.position_quantum, q_uv=cfg.uv_quantum)
    _self_check(ours, expected_triangles, expected_unique_positions, static_glb)
    report = decide_route(ours, theirs, cfg)
    return report.model_copy(update={"reason": report.reason + note})


def _normalise(
    ours: GlbDocument,
    theirs: GlbDocument,
    scale: float | None,
    translation: Float3 | None,
) -> tuple[GlbDocument, str]:
    if scale is not None and scale > 0:
        source: str = "task"
    elif translation is not None and scale is None:
        scale, source = 1.0, "task"
    else:
        fit = fit_affine(ours, theirs)
        if fit is None:
            return theirs, "; normalised scale=1 source=none"
        scale, translation = fit
        source = "bbox"
    t = translation or (0.0, 0.0, 0.0)
    normalised = normalise_positions(theirs, scale=scale, translation=t)
    note = (
        f"; normalised scale={scale:.6g} translation=({t[0]:.6g},{t[1]:.6g},{t[2]:.6g}) "
        f"source={source}"
    )
    return normalised, note


def _self_check(
    ours: SurfaceFingerprint,
    expected_triangles: int | None,
    expected_unique_positions: int | None,
    static_glb: Path,
) -> None:
    mismatches: dict[str, tuple[int, int]] = {}
    if expected_triangles is not None and expected_triangles != ours.triangle_count:
        mismatches["triangles"] = (expected_triangles, ours.triangle_count)
    if expected_unique_positions is not None and (
        expected_unique_positions != ours.unique_position_count
    ):
        mismatches["unique_positions"] = (expected_unique_positions, ours.unique_position_count)
    if mismatches:
        rendered = ", ".join(f"{k}: prepare={a} verify={b}" for k, (a, b) in mismatches.items())
        raise VerifySelfMismatchError(
            f"{static_glb.name} does not match the prepare report ({rendered})",
            detail={k: {"prepare": a, "verify": b} for k, (a, b) in mismatches.items()},
            hint="re-run the prepare stage; work/static.glb was modified or comes from another run",
        )
