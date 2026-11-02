"""UV analysis and the keep/unwrap/create decision (plan 09 §2.4). Pure.

``numpy`` is the one third-party module allowed Blender-side besides ``bpy``/``bmesh``/``mathutils``
(Blender bundles it); the module runs unchanged under CPython for unit tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from .errors import JobFailure
from .protocol import E_BJ_NO_UV_KEEP, UvSection

UvCfg = UvSection
"""The ``[uv]`` config section as the Blender side sees it."""

UvDecision = Literal["kept", "unwrapped", "created"]
Uv = tuple[float, float]
DEFAULT_GRID = 512
_ISLAND_ROUND = 6  # UV coordinates equal to 1e-6 are "the same vertex"
_EDGE_EPS = 1e-9  # strict inside test: cell centres on a shared edge belong to neither triangle


@dataclass(frozen=True)
class UvAnalysis:
    """What the heuristics know about a mesh group's first UV layer.

    Attributes:
        has_uv: At least one face has UV loops.
        overlap_ratio: Fraction of covered coverage-grid cells that are covered by >= 2 faces.
        island_count: Connected components of faces sharing (rounded) UV coordinates.
        source_texture_max_px: Largest side of any referenced texture, or ``None``.
        uv_bbox_coverage: Area of the UV bounding box (1.0 = the whole unit square).
    """

    has_uv: bool
    overlap_ratio: float
    island_count: int
    source_texture_max_px: int | None
    uv_bbox_coverage: float


def analyse(
    uv_loops_per_face: list[list[Uv]],
    texture_sizes: list[tuple[int, int]],
    grid: int = DEFAULT_GRID,
) -> UvAnalysis:
    """Rasterise the UV faces into a ``grid x grid`` coverage grid and summarise them.

    Args:
        uv_loops_per_face: One list of ``(u, v)`` per face, in loop order; n-gons are
            fan-triangulated. Faces with fewer than three loops are ignored.
        texture_sizes: ``(width, height)`` of the textures the materials reference.
        grid: Coverage grid resolution.

    Returns:
        The analysis. UV sets lying entirely outside the unit square are shifted by whole tiles
        before rasterising so tiled layouts are still measured.
    """
    faces = [face for face in uv_loops_per_face if len(face) >= 3]  # noqa: PLR2004 - triangle
    max_px = max((max(w, h) for w, h in texture_sizes), default=None)
    if not faces:
        return UvAnalysis(False, 0.0, 0, max_px, 0.0)
    points = np.asarray([uv for face in faces for uv in face], dtype=np.float64)
    low, high = points.min(axis=0), points.max(axis=0)
    offset = np.floor(low)
    return UvAnalysis(
        has_uv=True,
        overlap_ratio=_overlap_ratio(faces, offset, grid),
        island_count=_island_count(faces),
        source_texture_max_px=max_px,
        uv_bbox_coverage=float((high[0] - low[0]) * (high[1] - low[1])),
    )


def decide(a: UvAnalysis, cfg: UvCfg) -> tuple[UvDecision, str]:
    """Choose between keeping, re-unwrapping or creating UVs (plan 09 §2.4, normative order).

    Args:
        a: Result of :func:`analyse`.
        cfg: The ``[uv]`` section.

    Returns:
        ``(decision, reason)``.

    Raises:
        JobFailure: ``E_BJ_NO_UV_KEEP`` when ``cfg.mode == "keep"`` and there is no UV layer.
    """
    if cfg.mode == "keep":
        if not a.has_uv:
            raise JobFailure(
                E_BJ_NO_UV_KEEP,
                "uv.mode=keep but the mesh has no UV layer",
                {"hint": "set uv.mode to auto or unwrap"},
            )
        return "kept", "mode=keep"
    if cfg.mode == "unwrap":
        return ("unwrapped" if a.has_uv else "created"), "mode=unwrap"
    if not a.has_uv:
        return "created", "no uv layer"
    px = a.source_texture_max_px
    if px is not None and px <= cfg.palette_texture_max_px:
        return "unwrapped", f"palette texture {px}px"
    if a.overlap_ratio > cfg.overlap_ratio_threshold:
        return "unwrapped", f"overlap {a.overlap_ratio:.2f}"
    return "kept", "uvs usable"


# --------------------------------------------------------------------------- rasteriser
def _overlap_ratio(faces: list[list[Uv]], offset: npt.NDArray[np.float64], grid: int) -> float:
    cover = np.zeros((grid, grid), dtype=np.int32)
    for face in faces:
        pts = (np.asarray(face, dtype=np.float64) - offset) * grid
        for i in range(1, len(pts) - 1):  # fan triangulation
            _rasterise_triangle(cover, pts[0], pts[i], pts[i + 1])
    covered = int(np.count_nonzero(cover >= 1))
    if covered == 0:
        return 0.0
    return float(np.count_nonzero(cover >= 2)) / covered  # noqa: PLR2004 - "covered twice"


def _rasterise_triangle(
    cover: npt.NDArray[np.int32],
    a: npt.NDArray[np.float64],
    b: npt.NDArray[np.float64],
    c: npt.NDArray[np.float64],
) -> None:
    """Increment every grid cell whose centre lies strictly inside triangle ``abc``."""
    area = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    if abs(area) < _EDGE_EPS:
        return
    grid = cover.shape[0]
    xs, ys = (a[0], b[0], c[0]), (a[1], b[1], c[1])
    x0, x1 = max(int(np.floor(min(xs))), 0), min(int(np.ceil(max(xs))), grid)
    y0, y1 = max(int(np.floor(min(ys))), 0), min(int(np.ceil(max(ys))), grid)
    if x1 <= x0 or y1 <= y0:
        return
    gx, gy = np.meshgrid(np.arange(x0, x1) + 0.5, np.arange(y0, y1) + 0.5)
    sign = 1.0 if area > 0 else -1.0
    w0 = sign * _edge(b, c, gx, gy)
    w1 = sign * _edge(c, a, gx, gy)
    w2 = sign * _edge(a, b, gx, gy)
    inside = (w0 > _EDGE_EPS) & (w1 > _EDGE_EPS) & (w2 > _EDGE_EPS)
    cover[y0:y1, x0:x1][inside] += 1


def _edge(
    p: npt.NDArray[np.float64],
    q: npt.NDArray[np.float64],
    gx: npt.NDArray[np.float64],
    gy: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    result: npt.NDArray[np.float64] = (q[0] - p[0]) * (gy - p[1]) - (q[1] - p[1]) * (gx - p[0])
    return result


# --------------------------------------------------------------------------- islands
def _island_count(faces: list[list[Uv]]) -> int:
    """Union-find over faces that share a UV coordinate rounded to 1e-6."""
    parent = list(range(len(faces)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    first_face: dict[tuple[float, float], int] = {}
    for index, face in enumerate(faces):
        for u, v in face:
            key = (round(u, _ISLAND_ROUND) + 0.0, round(v, _ISLAND_ROUND) + 0.0)
            other = first_face.setdefault(key, index)
            if other != index:
                parent[find(other)] = find(index)
    return sum(1 for i in range(len(faces)) if find(i) == i)
