"""Surface fingerprint and verification report (plan 06 §6)."""

from __future__ import annotations

from typing import Literal

from grounded_pipeline.domain._base import FrozenModel
from grounded_pipeline.domain.ids import Sha256


class SurfaceFingerprint(FrozenModel):
    """Order-independent description of a triangle mesh's quantized surface."""

    triangle_count: int
    unique_position_count: int
    position_set_sha256: Sha256  # sorted unique quantized (x,y,z)
    surface_sha256: Sha256  # sorted unique quantized (x,y,z,u,v)
    has_uv: bool
    primitive_count: int
    material_count: int
    quantum_position: float
    quantum_uv: float


class VerifyReport(FrozenModel):
    """Outcome of comparing our static mesh with the provider's result."""

    ours: SurfaceFingerprint
    theirs: SurfaceFingerprint
    topology_match: bool  # triangle_count & unique_position_count equal
    position_match: bool  # position_set_sha256 equal
    uv_match: bool  # surface_sha256 equal
    route: Literal["extract", "bake"]
    reason: str
    forced: bool  # route forced by config
