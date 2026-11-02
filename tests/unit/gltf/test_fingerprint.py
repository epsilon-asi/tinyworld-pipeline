from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from fakes.glb_builder import MaterialSpec, NodeSpec, cube, cube_mesh, from_mesh, quad, write_glb
from tinyworld_pipeline.gltf.fingerprint import fit_affine, normalise_positions, surface_fingerprint
from tinyworld_pipeline.gltf.loader import GlbDocument, GlbMesh, load_glb

Q = {"q_pos": 1e-4, "q_uv": 1e-4}


GlbWriter = Callable[[bytes], Path]


@pytest.fixture
def glb(tmp_path: Path) -> GlbWriter:
    counter = iter(range(1000))

    def _write(data: bytes) -> Path:
        return write_glb(tmp_path / f"m{next(counter)}.glb", data)

    return _write


def _doc(positions: np.ndarray, uvs: np.ndarray | None, indices: np.ndarray) -> GlbDocument:
    mesh = GlbMesh(
        positions=np.asarray(positions, dtype=np.float32),
        uvs=None if uvs is None else np.asarray(uvs, dtype=np.float32),
        indices=np.asarray(indices, dtype=np.uint32).reshape(-1, 3),
        material_index=None,
    )
    return GlbDocument(meshes=[mesh], materials=[], images=[])


def _cube_doc(**kwargs: Any) -> GlbDocument:
    mesh = cube_mesh()
    assert mesh.uvs is not None
    positions = mesh.positions if "positions" not in kwargs else kwargs.pop("positions")
    uvs = mesh.uvs if "uvs" not in kwargs else kwargs.pop("uvs")
    return _doc(positions, uvs, mesh.indices)


# ------------------------------------------------------------------------------ basic shape
def test_cube_fingerprint_counts(glb: GlbWriter) -> None:
    fp = surface_fingerprint(load_glb(glb(cube())), **Q)
    assert fp.triangle_count == 12
    assert fp.unique_position_count == 8
    assert fp.has_uv is True
    assert fp.primitive_count == 1
    assert fp.material_count == 0
    assert (fp.quantum_position, fp.quantum_uv) == (1e-4, 1e-4)
    assert fp.position_set_sha256 != fp.surface_sha256


def test_material_count_counts_distinct_referenced_materials(glb: GlbWriter) -> None:
    mats = [MaterialSpec(name="a"), MaterialSpec(name="b")]
    fp = surface_fingerprint(load_glb(glb(cube(primitives=3, materials=mats))), **Q)
    assert fp.primitive_count == 3
    assert fp.material_count == 2


def test_empty_document() -> None:
    fp = surface_fingerprint(GlbDocument(meshes=[], materials=[], images=[]), **Q)
    assert fp.triangle_count == 0
    assert fp.unique_position_count == 0
    assert fp.has_uv is False
    assert fp.surface_sha256 == hashlib.sha256(b"").hexdigest()


def test_missing_uvs_on_any_primitive_disables_surface_hash(glb: GlbWriter) -> None:
    mesh = cube_mesh()
    fp = surface_fingerprint(load_glb(glb(from_mesh(mesh.positions, None, mesh.indices))), **Q)
    assert fp.has_uv is False
    assert fp.surface_sha256 == hashlib.sha256(b"").hexdigest()
    assert fp.unique_position_count == 8


# ------------------------------------------------------------------------------ invariance
@pytest.mark.parametrize(
    "variant",
    [
        {"shuffle": True, "seed": 1},
        {"shuffle": True, "seed": 7},
        {"split_seams": True},
        {"primitives": 3},
        {"primitives": 4, "shuffle": True, "split_seams": True},
        {"interleave": True},
        {"indexed": False},
        {"node_transform": [NodeSpec(), NodeSpec(rotation=(0.0, 0.0, 0.0, 1.0))]},
    ],
    ids=[
        "shuffle1",
        "shuffle7",
        "seams",
        "split3",
        "split-shuffle-seams",
        "interleaved",
        "unindexed",
        "nested-identity",
    ],
)
def test_fingerprint_is_invariant_to_exporter_choices(
    glb: GlbWriter, variant: dict[str, Any]
) -> None:
    reference = surface_fingerprint(load_glb(glb(cube())), **Q)
    other = surface_fingerprint(load_glb(glb(cube(**variant))), **Q)
    assert other.position_set_sha256 == reference.position_set_sha256
    assert other.surface_sha256 == reference.surface_sha256
    assert other.unique_position_count == reference.unique_position_count
    assert other.triangle_count == reference.triangle_count


def test_sensitive_to_uv_change_above_quantum_and_insensitive_below() -> None:
    reference = surface_fingerprint(_cube_doc(), **Q)
    mesh = cube_mesh()
    assert mesh.uvs is not None
    nudged = mesh.uvs.copy()
    nudged[0, 0] += 3e-4
    changed = surface_fingerprint(_cube_doc(uvs=nudged), **Q)
    assert changed.position_set_sha256 == reference.position_set_sha256
    assert changed.surface_sha256 != reference.surface_sha256
    tiny = mesh.uvs.copy()
    tiny[0, 0] += 2e-5
    same = surface_fingerprint(_cube_doc(uvs=tiny), **Q)
    assert same.surface_sha256 == reference.surface_sha256


def test_sensitive_to_position_change_above_quantum() -> None:
    reference = surface_fingerprint(_cube_doc(), **Q)
    moved = cube_mesh().positions.copy()
    moved[0] += np.float32(1e-3)
    changed = surface_fingerprint(_cube_doc(positions=moved), **Q)
    assert changed.position_set_sha256 != reference.position_set_sha256
    assert changed.surface_sha256 != reference.surface_sha256
    assert changed.unique_position_count == 9  # one corner split away from its twins


def test_quantum_is_part_of_the_fingerprint() -> None:
    doc = _cube_doc()
    a = surface_fingerprint(doc, q_pos=1e-4, q_uv=1e-4)
    b = surface_fingerprint(doc, q_pos=1e-3, q_uv=1e-4)
    assert a.position_set_sha256 != b.position_set_sha256
    assert a.quantum_position != b.quantum_position


@settings(max_examples=40, deadline=None)
@given(seed=st.integers(min_value=0, max_value=2**31 - 1))
def test_invariant_under_random_vertex_and_triangle_permutation(seed: int) -> None:
    mesh = cube_mesh()
    assert mesh.uvs is not None
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(mesh.positions))
    inverse = np.empty_like(perm)
    inverse[perm] = np.arange(len(perm))
    tri = inverse[mesh.indices][rng.permutation(len(mesh.indices))]
    tri = np.roll(tri, shift=int(rng.integers(0, 3)), axis=1)
    reference = surface_fingerprint(_doc(mesh.positions, mesh.uvs, mesh.indices), **Q)
    shuffled = surface_fingerprint(_doc(mesh.positions[perm], mesh.uvs[perm], tri), **Q)
    assert shuffled == reference


# ------------------------------------------------------------------------------ normalisation
def test_normalise_positions_with_task_values(glb: GlbWriter) -> None:
    reference = surface_fingerprint(load_glb(glb(cube())), **Q)
    scaled = load_glb(glb(cube(scale=2.5, translate=(0.1, -0.2, 0.3), shuffle=True)))
    assert surface_fingerprint(scaled, **Q).position_set_sha256 != reference.position_set_sha256
    restored = normalise_positions(scaled, scale=2.5, translation=(0.1, -0.2, 0.3))
    fp = surface_fingerprint(restored, **Q)
    assert fp.position_set_sha256 == reference.position_set_sha256
    assert fp.surface_sha256 == reference.surface_sha256
    assert restored.meshes[0].positions.dtype == np.float32


def test_normalise_positions_identity_returns_same_document() -> None:
    doc = _cube_doc()
    assert normalise_positions(doc, scale=None, translation=None) is doc
    assert normalise_positions(doc, scale=1.0, translation=(0.0, 0.0, 0.0)) is doc
    shifted = normalise_positions(doc, scale=None, translation=(1.0, 0.0, 0.0))
    assert shifted is not doc
    np.testing.assert_allclose(shifted.meshes[0].positions[:, 0], doc.meshes[0].positions[:, 0] - 1)


@pytest.mark.parametrize("scale", [0.0, -2.0])
def test_normalise_positions_rejects_non_positive_scale(scale: float) -> None:
    with pytest.raises(ValueError, match="positive"):
        normalise_positions(_cube_doc(), scale=scale, translation=None)


def test_fit_affine_recovers_uniform_scale_and_translation(glb: GlbWriter) -> None:
    ours = load_glb(glb(cube()))
    theirs = load_glb(glb(cube(scale=3.0, translate=(1.0, 2.0, -4.0), shuffle=True, primitives=2)))
    fit = fit_affine(ours, theirs)
    assert fit is not None
    scale, translation = fit
    assert scale == pytest.approx(3.0)
    assert translation == pytest.approx((1.0, 2.0, -4.0))
    restored = normalise_positions(theirs, scale=scale, translation=translation)
    assert (
        surface_fingerprint(restored, **Q).position_set_sha256
        == surface_fingerprint(ours, **Q).position_set_sha256
    )


def test_fit_affine_tolerates_float_noise() -> None:
    ours = _cube_doc()
    noisy = cube_mesh().positions * np.float32(2.0) + np.float32(1e-5)
    fit = fit_affine(ours, _cube_doc(positions=noisy))
    assert fit is not None
    assert fit[0] == pytest.approx(2.0, rel=1e-4)


def test_fit_affine_none_when_scale_is_not_uniform() -> None:
    ours = _cube_doc()
    stretched = cube_mesh().positions * np.array([1.0, 2.0, 1.0], dtype=np.float32)
    assert fit_affine(ours, _cube_doc(positions=stretched)) is None


def test_fit_affine_none_when_a_degenerate_axis_gains_extent() -> None:
    flat = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float32)
    lifted = flat.copy()
    lifted[2, 2] = 1.0
    tri = np.array([[0, 1, 2], [0, 2, 3]])
    assert fit_affine(_doc(flat, None, tri), _doc(lifted, None, tri)) is None
    assert fit_affine(_doc(flat, None, tri), _doc(flat * 2, None, tri)) == (2.0, (0.0, 0.0, 0.0))


def test_fit_affine_none_for_empty_or_degenerate_inputs() -> None:
    empty = GlbDocument(meshes=[], materials=[], images=[])
    point = _doc(np.zeros((3, 3)), None, np.array([[0, 1, 2]]))
    assert fit_affine(empty, _cube_doc()) is None
    assert fit_affine(_cube_doc(), empty) is None
    assert fit_affine(point, _cube_doc()) is None


def test_fit_affine_none_when_boxes_disagree_in_shape() -> None:
    ours = _cube_doc()
    # Same centre and same x/y extent but a vertex pushed out along z: ratios disagree.
    moved = cube_mesh().positions.copy()
    moved[0, 2] = 2.0
    assert fit_affine(ours, _cube_doc(positions=moved)) is None


def test_quad_builder_used_for_fit_matches_quad_mesh(glb: GlbWriter) -> None:
    doc = load_glb(glb(quad(scale=2.0)))
    fit = fit_affine(load_glb(glb(quad())), doc)
    assert fit is not None
    assert fit[0] == pytest.approx(2.0)
