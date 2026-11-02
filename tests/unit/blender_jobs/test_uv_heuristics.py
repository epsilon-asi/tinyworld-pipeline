"""``uv_heuristics``: rasteriser properties and the full ``decide`` table (plan 09 §2.4, §2.10)."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from fakes.fake_scene import cube_uv_atlas, cube_uv_palette
from grounded_pipeline.blender_jobs.errors import JobFailure
from grounded_pipeline.blender_jobs.protocol import E_BJ_NO_UV_KEEP, UvSection
from grounded_pipeline.blender_jobs.uv_heuristics import UvAnalysis, analyse, decide

Uv = tuple[float, float]


def quad(u0: float, v0: float, u1: float, v1: float) -> list[Uv]:
    return [(u0, v0), (u1, v0), (u1, v1), (u0, v1)]


# --------------------------------------------------------------------------- analyse
def test_perfect_atlas_has_zero_overlap_and_six_islands() -> None:
    a = analyse(cube_uv_atlas(), [(256, 256), (64, 128)])
    assert a.has_uv is True
    assert a.overlap_ratio == 0.0
    assert a.island_count == 6
    assert a.source_texture_max_px == 256
    assert a.uv_bbox_coverage == pytest.approx((1 - 0.04) * (1 - 0.04))


def test_fully_overlapping_quads_have_overlap_one_and_one_island() -> None:
    a = analyse(cube_uv_palette(), [(16, 4)])
    assert a.overlap_ratio == 1.0
    assert a.island_count == 1
    assert a.source_texture_max_px == 16


def test_no_faces_means_no_uv() -> None:
    a = analyse([], [(4, 4)])
    assert a == UvAnalysis(False, 0.0, 0, 4, 0.0)
    assert analyse([[(0.0, 0.0), (1.0, 1.0)]], []).has_uv is False  # degenerate face ignored
    assert analyse([], []).source_texture_max_px is None


def test_half_overlapping_quads_are_one_third_overlapped() -> None:
    # Union area 1.5 A, doubly covered 0.5 A → 1/3 (up to grid discretisation).
    a = analyse([quad(0.0, 0.0, 0.5, 0.5), quad(0.25, 0.0, 0.75, 0.5)], [])
    assert a.overlap_ratio == pytest.approx(1 / 3, abs=0.01)
    assert a.island_count == 2


def test_adjacent_quads_sharing_an_edge_do_not_overlap_and_form_one_island() -> None:
    a = analyse([quad(0.0, 0.0, 0.5, 1.0), quad(0.5, 0.0, 1.0, 1.0)], [])
    assert a.overlap_ratio == 0.0
    assert a.island_count == 1


def test_ngon_fan_triangulation_does_not_self_overlap() -> None:
    hexagon = [(0.5, 0.1), (0.85, 0.3), (0.85, 0.7), (0.5, 0.9), (0.15, 0.7), (0.15, 0.3)]
    assert analyse([hexagon], []).overlap_ratio == 0.0


def test_tiled_uvs_are_shifted_into_the_unit_square() -> None:
    shifted = [[(u + 2.0, v - 1.0) for u, v in face] for face in cube_uv_atlas()]
    assert analyse(shifted, []).overlap_ratio == 0.0
    assert analyse(shifted, []).island_count == 6


def test_clockwise_and_counter_clockwise_faces_rasterise_alike() -> None:
    ccw = quad(0.1, 0.1, 0.9, 0.9)
    cw = list(reversed(ccw))
    # Fan diagonals differ between windings; cells on a diagonal belong to neither triangle.
    assert analyse([ccw, cw], []).overlap_ratio == pytest.approx(1.0, abs=0.01)


def test_island_rounding_merges_coordinates_within_1e_6() -> None:
    a = [(0.0, 0.0), (0.4, 0.0), (0.4, 0.4), (0.0, 0.4)]
    b = [(0.4 + 1e-9, 0.0), (0.8, 0.0), (0.8, 0.4), (0.4, 0.4)]
    assert analyse([a, b], []).island_count == 1


finite = st.floats(min_value=-2.0, max_value=3.0, allow_nan=False, allow_infinity=False)
faces_strategy = st.lists(
    st.lists(st.tuples(finite, finite), min_size=3, max_size=6), min_size=1, max_size=10
)


@settings(max_examples=60, deadline=None)
@given(faces=faces_strategy)
def test_overlap_ratio_is_in_unit_interval(faces: list[list[Uv]]) -> None:
    a = analyse(faces, [], grid=32)
    assert 0.0 <= a.overlap_ratio <= 1.0
    assert 1 <= a.island_count <= len(faces)
    assert a.uv_bbox_coverage >= 0.0


# --------------------------------------------------------------------------- decide
def analysis(**overrides: object) -> UvAnalysis:
    base: dict[str, object] = {
        "has_uv": True,
        "overlap_ratio": 0.0,
        "island_count": 1,
        "source_texture_max_px": 1024,
        "uv_bbox_coverage": 0.8,
    }
    base.update(overrides)
    return UvAnalysis(**base)  # type: ignore[arg-type]


def test_decide_keep_mode() -> None:
    assert decide(analysis(), UvSection(mode="keep")) == ("kept", "mode=keep")
    assert decide(analysis(overlap_ratio=1.0, source_texture_max_px=4), UvSection(mode="keep")) == (
        "kept",
        "mode=keep",
    )


def test_decide_keep_mode_without_uv_fails() -> None:
    with pytest.raises(JobFailure) as info:
        decide(analysis(has_uv=False), UvSection(mode="keep"))
    assert info.value.code == E_BJ_NO_UV_KEEP


def test_decide_unwrap_mode() -> None:
    assert decide(analysis(), UvSection(mode="unwrap")) == ("unwrapped", "mode=unwrap")
    assert decide(analysis(has_uv=False), UvSection(mode="unwrap")) == ("created", "mode=unwrap")


def test_decide_auto_no_uv_creates() -> None:
    assert decide(analysis(has_uv=False), UvSection()) == ("created", "no uv layer")


def test_decide_auto_palette_texture_unwraps_at_and_below_threshold() -> None:
    cfg = UvSection(palette_texture_max_px=128)
    assert decide(analysis(source_texture_max_px=64), cfg) == ("unwrapped", "palette texture 64px")
    assert decide(analysis(source_texture_max_px=128), cfg) == (
        "unwrapped",
        "palette texture 128px",
    )
    assert decide(analysis(source_texture_max_px=129), cfg) == ("kept", "uvs usable")


def test_decide_auto_overlap_unwraps_strictly_above_threshold() -> None:
    cfg = UvSection(overlap_ratio_threshold=0.2)
    assert decide(analysis(overlap_ratio=0.5), cfg) == ("unwrapped", "overlap 0.50")
    assert decide(analysis(overlap_ratio=0.2), cfg) == ("kept", "uvs usable")


def test_decide_auto_without_textures_keeps_usable_uvs() -> None:
    assert decide(analysis(source_texture_max_px=None), UvSection()) == ("kept", "uvs usable")


def test_decide_palette_takes_precedence_over_overlap() -> None:
    a = analysis(source_texture_max_px=16, overlap_ratio=1.0)
    assert decide(a, UvSection())[1] == "palette texture 16px"
