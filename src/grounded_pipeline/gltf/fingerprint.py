"""Order-independent surface fingerprints and frame normalisation (plan 11 §2).

The fingerprint describes a mesh as *sets* of quantized rows, so it is invariant to everything a
texturing provider or a second exporter may legitimately do without changing the surface: vertex
reordering, primitive merging/splitting, re-splitting vertices along UV seams, triangle rewinding.
It is sensitive to any real change of positions or UVs larger than the quantum.
"""

from __future__ import annotations

import hashlib
from dataclasses import replace

import numpy as np
import numpy.typing as npt

from grounded_pipeline.domain.fingerprint import SurfaceFingerprint
from grounded_pipeline.gltf.loader import GlbDocument

Float3 = tuple[float, float, float]


def surface_fingerprint(doc: GlbDocument, *, q_pos: float, q_uv: float) -> SurfaceFingerprint:
    """Compute the quantized surface fingerprint of every primitive in ``doc`` (plan 11 §2).

    Args:
        doc: Loaded document; positions must already be in the frame to compare in.
        q_pos: Position quantum (world units); ``1e-4`` absorbs float32 export noise.
        q_uv: UV quantum.

    Returns:
        The fingerprint. ``material_count`` counts the distinct materials referenced by primitives.
    """
    quantized = _quantize(_concat_positions(doc), q_pos)
    unique_positions = np.unique(quantized, axis=0)
    has_uv = bool(doc.meshes) and all(m.uvs is not None for m in doc.meshes)
    if has_uv:
        uvs = np.concatenate([m.uvs for m in doc.meshes if m.uvs is not None], axis=0)
        rows = np.concatenate([quantized, _quantize(uvs, q_uv)], axis=1)
        surface_sha = _sha256_rows(np.unique(rows, axis=0))
    else:
        surface_sha = hashlib.sha256(b"").hexdigest()
    referenced = {m.material_index for m in doc.meshes if m.material_index is not None}
    return SurfaceFingerprint(
        triangle_count=sum(len(m.indices) for m in doc.meshes),
        unique_position_count=len(unique_positions),
        position_set_sha256=_sha256_rows(unique_positions),
        surface_sha256=surface_sha,
        has_uv=has_uv,
        primitive_count=len(doc.meshes),
        material_count=len(referenced),
        quantum_position=q_pos,
        quantum_uv=q_uv,
    )


def normalise_positions(
    doc: GlbDocument, *, scale: float | None, translation: Float3 | None
) -> GlbDocument:
    """Map positions from a provider's normalised frame back into ours: ``(p - t) / s``.

    Tripo documents ``normalized = original x normalize_scale (+ normalize_translation)``; this is
    the exact inverse. ``None`` means "not applied" (scale 1, zero translation).

    Args:
        doc: Document whose positions are in the normalised frame.
        scale: ``normalize_scale`` (must be positive) or ``None``.
        translation: ``normalize_translation`` or ``None``.

    Returns:
        A new document with transformed positions; meshes without positions change are shared.

    Raises:
        ValueError: ``scale`` is zero or negative.
    """
    s = 1.0 if scale is None else float(scale)
    if s <= 0:
        raise ValueError(f"normalisation scale must be positive, got {s}")
    t = np.zeros(3) if translation is None else np.asarray(translation, dtype=np.float64)
    if s == 1.0 and not t.any():
        return doc
    meshes = [
        replace(m, positions=((m.positions.astype(np.float64) - t) / s).astype(np.float32))
        for m in doc.meshes
    ]
    return replace(doc, meshes=meshes)


def fit_affine(
    ours: GlbDocument, theirs: GlbDocument, *, rel_tol: float = 1e-3
) -> tuple[float, Float3] | None:
    """Fit ``theirs ~= scale x ours + translation`` from the two axis-aligned bounding boxes.

    Used when the provider did not report its normalisation. The fit is accepted only when a
    single uniform scale explains every non-degenerate axis within ``rel_tol`` (relative to the
    box extents), i.e. the boxes genuinely agree up to scale and translation.

    Args:
        ours: Reference document (our frame).
        theirs: Document in the unknown frame.
        rel_tol: Relative tolerance on per-axis scale agreement and on the fitted box corners.

    Returns:
        ``(scale, translation)`` such that ``normalise_positions(theirs, scale, translation)``
        maps it into our frame, or ``None`` when no uniform-scale fit exists (or a box is empty).
    """
    ours_pts, theirs_pts = _concat_positions(ours), _concat_positions(theirs)
    if not len(ours_pts) or not len(theirs_pts):
        return None
    o_min, o_max = ours_pts.min(axis=0).astype(np.float64), ours_pts.max(axis=0).astype(np.float64)
    t_min, t_max = (
        theirs_pts.min(axis=0).astype(np.float64),
        theirs_pts.max(axis=0).astype(np.float64),
    )
    o_ext, t_ext = o_max - o_min, t_max - t_min
    o_diag = float(np.linalg.norm(o_ext))
    if o_diag == 0.0:
        return None
    live = o_ext > rel_tol * o_diag
    if not live.any():
        return None
    ratios = t_ext[live] / o_ext[live]
    scale = float(ratios.mean())
    if scale <= 0 or np.any(np.abs(ratios - scale) > rel_tol * scale):
        return None
    translation = (t_min + t_max) / 2 - scale * (o_min + o_max) / 2
    fitted_min, fitted_max = scale * o_min + translation, scale * o_max + translation
    tolerance = rel_tol * max(float(np.linalg.norm(t_ext)), scale * o_diag)
    if np.any(np.abs(fitted_min - t_min) > tolerance) or np.any(
        np.abs(fitted_max - t_max) > tolerance
    ):
        return None
    return scale, (float(translation[0]), float(translation[1]), float(translation[2]))


# ------------------------------------------------------------------------------------ helpers
def _concat_positions(doc: GlbDocument) -> npt.NDArray[np.float32]:
    if not doc.meshes:
        return np.zeros((0, 3), dtype=np.float32)
    return np.concatenate([m.positions for m in doc.meshes], axis=0)


def _quantize(values: npt.NDArray[np.float32], quantum: float) -> npt.NDArray[np.int64]:
    return np.round(values.astype(np.float64) / quantum).astype(np.int64)


def _sha256_rows(rows: npt.NDArray[np.int64]) -> str:
    # Fixed little-endian int64 so the hash is identical on every platform.
    return hashlib.sha256(np.ascontiguousarray(rows).astype("<i8").tobytes()).hexdigest()
