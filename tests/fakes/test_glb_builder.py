"""Self-tests for the GLB builder fake: a fake with a bug poisons every suite that uses it."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pygltflib
import pytest

from fakes.glb_builder import (
    ImageSpec,
    MaterialSpec,
    NodeSpec,
    cube,
    cube_mesh,
    encode_image,
    from_mesh,
    gradient_image,
    quad,
    quad_mesh,
    rewrite,
    solid_image,
    textured_set,
    write_glb,
)


def _load(data: bytes) -> pygltflib.GLTF2:
    gltf = pygltflib.GLTF2.load_from_bytes(data)
    assert gltf is not None
    return gltf


def _positions(gltf: pygltflib.GLTF2, primitive: int = 0) -> np.ndarray:
    acc = gltf.accessors[gltf.meshes[0].primitives[primitive].attributes.POSITION]
    view = gltf.bufferViews[acc.bufferView]
    stride = view.byteStride or 12
    raw = gltf.binary_blob()
    offset = view.byteOffset + acc.byteOffset
    out = np.ndarray((acc.count, 3), dtype="<f4", buffer=raw, offset=offset, strides=(stride, 4))
    return np.array(out)


def test_quad_and_cube_are_valid_glbs() -> None:
    for data, vertices, triangles in ((quad(), 4, 2), (cube(), 24, 12)):
        assert data[:4] == b"glTF"
        gltf = _load(data)
        prim = gltf.meshes[0].primitives[0]
        assert gltf.accessors[prim.attributes.POSITION].count == vertices
        assert gltf.accessors[prim.attributes.TEXCOORD_0].count == vertices
        assert gltf.accessors[prim.indices].count == triangles * 3
        assert gltf.scenes[gltf.scene].nodes == [0]
        assert gltf.nodes[0].mesh == 0


def test_cube_mesh_is_a_unit_cube_with_per_face_uvs() -> None:
    mesh = cube_mesh()
    assert mesh.uvs is not None
    assert mesh.positions.min() == -0.5
    assert mesh.positions.max() == 0.5
    assert len(np.unique(mesh.positions, axis=0)) == 8
    assert mesh.indices.max() == 23
    assert set(map(tuple, mesh.uvs.tolist())) == {(0, 0), (1, 0), (1, 1), (0, 1)}


def test_shuffle_permutes_vertices_but_keeps_the_surface() -> None:
    plain, shuffled = _load(cube()), _load(cube(shuffle=True, seed=3))
    a, b = _positions(plain), _positions(shuffled)
    assert not np.array_equal(a, b)
    assert np.array_equal(np.unique(a, axis=0), np.unique(b, axis=0))


def test_split_seams_gives_each_triangle_its_own_vertices() -> None:
    gltf = _load(quad(split_seams=True))
    prim = gltf.meshes[0].primitives[0]
    assert gltf.accessors[prim.attributes.POSITION].count == 6
    assert gltf.accessors[prim.indices].count == 6


def test_scale_and_translate_are_applied_to_positions() -> None:
    gltf = _load(quad(scale=2.0, translate=(1.0, 0.0, -1.0)))
    expected = quad_mesh().positions * 2 + np.array([1.0, 0.0, -1.0], dtype=np.float32)
    np.testing.assert_array_equal(_positions(gltf), expected)


def test_primitives_split_triangles_and_compact_vertices() -> None:
    gltf = _load(cube(primitives=3))
    prims = gltf.meshes[0].primitives
    assert len(prims) == 3
    assert sum(gltf.accessors[p.indices].count for p in prims) == 36
    assert all(gltf.accessors[p.attributes.POSITION].count == 8 for p in prims)


def test_materials_default_round_robin_and_explicit_assignment() -> None:
    mats = [MaterialSpec(name="a"), MaterialSpec(name="b")]
    gltf = _load(cube(primitives=3, materials=mats))
    assert [p.material for p in gltf.meshes[0].primitives] == [0, 1, 0]
    gltf = _load(cube(primitives=2, materials=mats, primitive_materials=[1, None]))
    assert [p.material for p in gltf.meshes[0].primitives] == [1, None]
    assert [m.name for m in gltf.materials] == ["a", "b"]


def test_textured_set_builds_five_images_and_a_full_material() -> None:
    images, material = textured_set()
    gltf = _load(quad(images=images, materials=[material]))
    assert len(gltf.images) == 5
    assert len(gltf.textures) == 5
    assert [t.source for t in gltf.textures] == [0, 1, 2, 3, 4]
    mat = gltf.materials[0]
    assert mat.pbrMetallicRoughness.baseColorTexture.index == 0
    assert mat.pbrMetallicRoughness.metallicRoughnessTexture.index == 1
    assert mat.normalTexture.index == 2
    assert mat.occlusionTexture.index == 3
    assert mat.emissiveTexture.index == 4
    assert all(img.mimeType == "image/png" and img.bufferView is not None for img in gltf.images)


def test_material_factors_and_extensions_are_written() -> None:
    spec = MaterialSpec(
        base_color_image=0,
        normal_image=0,
        occlusion_image=0,
        base_color_factor=(0.5, 0.25, 1.0, 0.75),
        metallic_factor=0.2,
        roughness_factor=0.3,
        emissive_factor=(1.0, 0.5, 0.0),
        normal_scale=0.8,
        occlusion_strength=0.6,
        emissive_strength=4.0,
        texture_transform=True,
    )
    gltf = _load(quad(images=[solid_image((1, 2, 3, 4))], materials=[spec]))
    mat = gltf.materials[0]
    assert mat.pbrMetallicRoughness.baseColorFactor == [0.5, 0.25, 1.0, 0.75]
    assert mat.pbrMetallicRoughness.metallicFactor == 0.2
    assert mat.pbrMetallicRoughness.roughnessFactor == 0.3
    assert mat.emissiveFactor == [1.0, 0.5, 0.0]
    assert mat.normalTexture.scale == 0.8
    assert mat.occlusionTexture.strength == 0.6
    assert mat.extensions["KHR_materials_emissive_strength"]["emissiveStrength"] == 4.0
    assert "KHR_texture_transform" in mat.pbrMetallicRoughness.baseColorTexture.extensions
    assert set(gltf.extensionsUsed) == {"KHR_materials_emissive_strength", "KHR_texture_transform"}


def test_images_can_be_jpeg_and_data_uri() -> None:
    spec = ImageSpec(pixels=solid_image((9, 9, 9)).pixels, fmt="jpeg", embed="datauri")
    gltf = _load(quad(images=[spec]))
    assert gltf.images[0].uri.startswith("data:image/jpeg;base64,")
    assert gltf.images[0].bufferView is None
    assert encode_image(spec)[:3] == b"\xff\xd8\xff"
    assert encode_image(solid_image((1, 2, 3, 4)))[:4] == b"\x89PNG"


def test_image_helpers_have_expected_shapes_and_values() -> None:
    solid = solid_image((10, 20, 30, 40), size=4)
    assert solid.pixels.shape == (4, 4, 4)
    assert (solid.pixels == np.array([10, 20, 30, 40], dtype=np.uint8)).all()
    gradient = gradient_image(size=4)
    assert gradient.pixels.shape == (4, 4, 4)
    assert len({tuple(px) for px in gradient.pixels.reshape(-1, 4).tolist()}) == 16


def test_node_chain_and_matrix_nodes() -> None:
    chain = [
        NodeSpec(translation=(1.0, 2.0, 3.0), name="root"),
        NodeSpec(rotation=(0.0, 0.0, 0.0, 1.0), scale=(2.0, 2.0, 2.0)),
        NodeSpec(matrix=tuple(np.eye(4).ravel().tolist())),
    ]
    gltf = _load(quad(node_transform=chain))
    assert [n.children for n in gltf.nodes] == [[1], [2], []]
    assert gltf.nodes[0].translation == [1.0, 2.0, 3.0]
    assert gltf.nodes[1].scale == [2.0, 2.0, 2.0]
    assert gltf.nodes[2].matrix == [1.0 if i % 5 == 0 else 0.0 for i in range(16)]
    assert gltf.nodes[2].mesh == 0
    single = _load(quad(node_transform=NodeSpec(scale=(3.0, 3.0, 3.0))))
    assert single.nodes[0].scale == [3.0, 3.0, 3.0]


def test_unsupported_structure_knobs() -> None:
    sparse = _load(quad(sparse=True))
    assert sparse.accessors[0].sparse.count == 1
    strip = _load(quad(mode=5))
    assert strip.meshes[0].primitives[0].mode == 5
    draco = _load(quad(draco=True))
    assert "KHR_draco_mesh_compression" in draco.meshes[0].primitives[0].extensions
    meshopt = _load(quad(meshopt=True))
    assert "EXT_meshopt_compression" in meshopt.bufferViews[0].extensions
    required = _load(quad(extensions_required=("KHR_foo",)))
    assert required.extensionsRequired == ["KHR_foo"]


def test_interleaved_and_integer_uvs_and_unindexed() -> None:
    inter = _load(quad(interleave=True))
    assert inter.bufferViews[0].byteStride == 20
    assert inter.accessors[inter.meshes[0].primitives[0].attributes.TEXCOORD_0].byteOffset == 12
    np.testing.assert_array_equal(_positions(inter), quad_mesh().positions)
    for component, ctype in (("uint16", 5123), ("uint8", 5121)):
        gltf = _load(quad(uv_component=component))
        acc = gltf.accessors[gltf.meshes[0].primitives[0].attributes.TEXCOORD_0]
        assert (acc.componentType, acc.normalized) == (ctype, True)
    unindexed = _load(quad(indexed=False))
    assert unindexed.meshes[0].primitives[0].indices is None
    assert unindexed.accessors[0].count == 6


def test_missing_uvs_and_large_meshes_use_uint32_indices() -> None:
    mesh = quad_mesh()
    gltf = _load(from_mesh(mesh.positions, None, mesh.indices))
    assert gltf.meshes[0].primitives[0].attributes.TEXCOORD_0 is None
    n = 70_000
    positions = np.zeros((n, 3), dtype=np.float32)
    positions[:, 0] = np.arange(n)
    indices = np.arange(n - n % 3, dtype=np.uint32).reshape(-1, 3)
    big = _load(from_mesh(positions, None, indices))
    assert big.accessors[big.meshes[0].primitives[0].indices].componentType == 5125


def test_write_glb_creates_parents(tmp_path: Path) -> None:
    out = write_glb(tmp_path / "nested" / "dir" / "q.glb", quad())
    assert out.read_bytes()[:4] == b"glTF"


def test_rewrite_keeps_buffers_and_applies_mutation() -> None:
    original = quad()
    edited = rewrite(original, lambda g: setattr(g.accessors[0], "count", 999))
    gltf = _load(edited)
    assert gltf.accessors[0].count == 999
    assert gltf.binary_blob() == _load(original).binary_blob()


@pytest.mark.parametrize(("builder", "expected_max"), [(quad, 2.0), (cube, 1.0)])
def test_builders_forward_kwargs(builder: Callable[..., bytes], expected_max: float) -> None:
    assert _load(builder(scale=2.0)).accessors[0].max[0] == expected_max
