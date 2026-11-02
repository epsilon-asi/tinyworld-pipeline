from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from fakes.glb_builder import cube, cube_mesh, from_mesh, quad, write_glb
from tinyworld_pipeline.domain.errors import GeometryChangedError, VerifySelfMismatchError
from tinyworld_pipeline.domain.fingerprint import SurfaceFingerprint, VerifyReport
from tinyworld_pipeline.domain.library_config import VerifySection
from tinyworld_pipeline.gltf.verify import decide_route, verify_surfaces

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64


def _fp(
    *, tris: int = 12, uniq: int = 8, pos: str = SHA_A, surf: str = SHA_B
) -> SurfaceFingerprint:
    return SurfaceFingerprint(
        triangle_count=tris,
        unique_position_count=uniq,
        position_set_sha256=pos,
        surface_sha256=surf,
        has_uv=True,
        primitive_count=1,
        material_count=1,
        quantum_position=1e-4,
        quantum_uv=1e-4,
    )


OURS = _fp()
SAME = _fp()
UV_CHANGED = _fp(surf=SHA_C)
POS_CHANGED_SAME_TOPOLOGY = _fp(pos=SHA_C, surf=SHA_D)
REMESHED = _fp(tris=40, uniq=30, pos=SHA_C, surf=SHA_D)
TRIS_ONLY_DIFFER = _fp(tris=13, pos=SHA_C, surf=SHA_D)


@pytest.mark.parametrize(
    ("theirs", "cfg", "route", "forced", "matches"),
    [
        (SAME, VerifySection(), "extract", False, (True, True, True)),
        (UV_CHANGED, VerifySection(), "bake", False, (True, True, False)),
        (POS_CHANGED_SAME_TOPOLOGY, VerifySection(), "bake", False, (True, False, False)),
        (REMESHED, VerifySection(allow_geometry_change=True), "bake", False, (False, False, False)),
        (
            TRIS_ONLY_DIFFER,
            VerifySection(allow_geometry_change=True),
            "bake",
            False,
            (False, False, False),
        ),
        (REMESHED, VerifySection(route="extract"), "extract", True, (False, False, False)),
        (SAME, VerifySection(route="bake"), "bake", True, (True, True, True)),
        (UV_CHANGED, VerifySection(route="extract"), "extract", True, (True, True, False)),
    ],
    ids=[
        "identical->extract",
        "uv-changed->bake",
        "positions-changed-topology-same->bake",
        "remeshed-allowed->bake",
        "triangle-count-differs-allowed->bake",
        "forced-extract",
        "forced-bake",
        "forced-extract-despite-uv-change",
    ],
)
def test_decide_route_table(
    theirs: SurfaceFingerprint,
    cfg: VerifySection,
    route: str,
    forced: bool,
    matches: tuple[bool, bool, bool],
) -> None:
    report = decide_route(OURS, theirs, cfg)
    assert report.route == route
    assert report.forced is forced
    assert (report.topology_match, report.position_match, report.uv_match) == matches
    assert report.ours == OURS
    assert report.theirs == theirs
    assert report.reason


@pytest.mark.parametrize("theirs", [REMESHED, TRIS_ONLY_DIFFER], ids=["remeshed", "tris-differ"])
def test_decide_route_raises_when_geometry_changed_and_not_allowed(
    theirs: SurfaceFingerprint,
) -> None:
    with pytest.raises(GeometryChangedError) as info:
        decide_route(OURS, theirs, VerifySection())
    err = info.value
    assert err.code == "E_GEOMETRY_CHANGED"
    assert "allow_geometry_change" in (err.hint or "")
    assert err.detail["ours"]["triangle_count"] == 12
    assert err.detail["theirs"]["triangle_count"] == theirs.triangle_count


def test_forced_route_reason_names_the_config_key() -> None:
    report = decide_route(OURS, SAME, VerifySection(route="bake"))
    assert "verify.route" in report.reason


# ------------------------------------------------------------------------------ verify_surfaces
@pytest.fixture
def ours_glb(tmp_path: Path) -> Path:
    return write_glb(tmp_path / "static.glb", cube())


def _verify(
    ours: Path, theirs: Path, cfg: VerifySection | None = None, **kwargs: Any
) -> VerifyReport:
    defaults: dict[str, Any] = {
        "expected_triangles": 12,
        "expected_unique_positions": 8,
        "normalize_scale": None,
        "normalize_translation": None,
    }
    return verify_surfaces(ours, theirs, cfg or VerifySection(), **{**defaults, **kwargs})


def test_verify_surfaces_with_task_normalisation(ours_glb: Path, tmp_path: Path) -> None:
    theirs = write_glb(tmp_path / "r.glb", cube(scale=2.0, translate=(0.5, 0.0, 0.0), shuffle=True))
    report = _verify(ours_glb, theirs, normalize_scale=2.0, normalize_translation=(0.5, 0.0, 0.0))
    assert report.route == "extract"
    assert report.reason.endswith("; normalised scale=2 translation=(0.5,0,0) source=task")


def test_verify_surfaces_scale_without_translation(ours_glb: Path, tmp_path: Path) -> None:
    theirs = write_glb(tmp_path / "r.glb", cube(scale=4.0))
    report = _verify(ours_glb, theirs, normalize_scale=4.0)
    assert report.route == "extract"
    assert "source=task" in report.reason


def test_verify_surfaces_translation_only_counts_as_task(ours_glb: Path, tmp_path: Path) -> None:
    theirs = write_glb(tmp_path / "r.glb", cube(translate=(0.0, 1.0, 0.0)))
    report = _verify(ours_glb, theirs, normalize_translation=(0.0, 1.0, 0.0))
    assert report.route == "extract"
    assert "scale=1 translation=(0,1,0) source=task" in report.reason


def test_verify_surfaces_falls_back_to_bbox_fit(ours_glb: Path, tmp_path: Path) -> None:
    theirs = write_glb(tmp_path / "r.glb", cube(scale=2.0, translate=(0.5, 0.0, 0.0), primitives=2))
    report = _verify(ours_glb, theirs)
    assert report.route == "extract"
    assert report.position_match
    assert report.uv_match
    assert "scale=2 translation=(0.5,0,0) source=bbox" in report.reason


def test_verify_surfaces_non_positive_task_scale_falls_back_to_bbox(
    ours_glb: Path, tmp_path: Path
) -> None:
    theirs = write_glb(tmp_path / "r.glb", cube(scale=2.0))
    report = _verify(ours_glb, theirs, normalize_scale=0.0)
    assert "source=bbox" in report.reason


def test_verify_surfaces_records_source_none_when_no_fit_exists(
    ours_glb: Path, tmp_path: Path
) -> None:
    mesh = cube_mesh()
    stretched = mesh.positions * np.array([1.0, 3.0, 1.0], dtype=np.float32)
    theirs = write_glb(tmp_path / "r.glb", from_mesh(stretched, mesh.uvs, mesh.indices))
    report = _verify(ours_glb, theirs, VerifySection(allow_geometry_change=False))
    # Same topology, positions differ -> bake without needing allow_geometry_change.
    assert report.route == "bake"
    assert report.topology_match is True
    assert report.position_match is False
    assert report.reason.endswith("; normalised scale=1 source=none")


def test_verify_surfaces_self_mismatch(ours_glb: Path, tmp_path: Path) -> None:
    theirs = write_glb(tmp_path / "r.glb", cube())
    with pytest.raises(VerifySelfMismatchError) as info:
        _verify(ours_glb, theirs, expected_triangles=10, expected_unique_positions=9)
    err = info.value
    assert err.code == "E_VERIFY_SELF_MISMATCH"
    assert err.detail == {
        "triangles": {"prepare": 10, "verify": 12},
        "unique_positions": {"prepare": 9, "verify": 8},
    }
    assert "prepare" in (err.hint or "")


def test_verify_surfaces_self_check_skipped_when_counts_unknown(
    ours_glb: Path, tmp_path: Path
) -> None:
    theirs = write_glb(tmp_path / "r.glb", cube())
    report = _verify(ours_glb, theirs, expected_triangles=None, expected_unique_positions=None)
    assert report.route == "extract"


def test_verify_surfaces_only_triangle_mismatch(ours_glb: Path, tmp_path: Path) -> None:
    theirs = write_glb(tmp_path / "r.glb", cube())
    with pytest.raises(VerifySelfMismatchError, match="triangles"):
        _verify(ours_glb, theirs, expected_triangles=11)


def test_verify_surfaces_propagates_geometry_changed(ours_glb: Path, tmp_path: Path) -> None:
    theirs = write_glb(tmp_path / "r.glb", quad())
    with pytest.raises(GeometryChangedError):
        _verify(ours_glb, theirs)
    report = _verify(ours_glb, theirs, VerifySection(allow_geometry_change=True))
    assert report.route == "bake"
    assert "allow_geometry_change" in report.reason
