from __future__ import annotations

import math
import struct
from collections.abc import Callable
from pathlib import Path
from typing import Any

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
    quad,
    quad_mesh,
    rewrite,
    solid_image,
    textured_set,
    write_glb,
)
from grounded_pipeline.domain.errors import UnsupportedGlbError
from grounded_pipeline.gltf.loader import GlbTextureRef, load_glb, sniff_image_mime

GlbWriter = Callable[[bytes], Path]


@pytest.fixture
def glb(tmp_path: Path) -> GlbWriter:
    counter = iter(range(1000))

    def _write(data: bytes) -> Path:
        return write_glb(tmp_path / f"m{next(counter)}.glb", data)

    return _write


def _sorted_rows(a: np.ndarray) -> np.ndarray:
    return np.unique(np.round(a, 5), axis=0)


# ------------------------------------------------------------------------------ happy paths
def test_quad_round_trips(glb: GlbWriter) -> None:
    doc = load_glb(glb(quad()))
    mesh = quad_mesh()
    assert len(doc.meshes) == 1
    (m,) = doc.meshes
    assert m.positions.dtype == np.float32
    assert m.indices.dtype == np.uint32
    np.testing.assert_array_equal(m.positions, mesh.positions)
    assert m.uvs is not None
    np.testing.assert_array_equal(m.uvs, mesh.uvs)
    np.testing.assert_array_equal(m.indices, mesh.indices)
    assert m.material_index is None
    assert doc.materials == []
    assert doc.images == []


def test_cube_and_two_primitive_cube(glb: GlbWriter) -> None:
    one = load_glb(glb(cube()))
    two = load_glb(glb(cube(primitives=2)))
    assert one.meshes[0].positions.shape == (24, 3)
    assert one.meshes[0].indices.shape == (12, 3)
    assert len(two.meshes) == 2
    assert sum(len(m.indices) for m in two.meshes) == 12
    merged = np.concatenate([m.positions for m in two.meshes])
    np.testing.assert_array_equal(_sorted_rows(merged), _sorted_rows(one.meshes[0].positions))


def test_nested_nodes_yield_world_space_positions(glb: GlbWriter) -> None:
    chain = [NodeSpec(translation=(1.0, 2.0, 3.0)), NodeSpec(scale=(2.0, 2.0, 2.0))]
    doc = load_glb(glb(quad(node_transform=chain)))
    expected = quad_mesh().positions * 2 + np.array([1, 2, 3], dtype=np.float32)
    np.testing.assert_allclose(doc.meshes[0].positions, expected, atol=1e-6)


def test_rotation_quaternion_is_applied(glb: GlbWriter) -> None:
    half = math.sqrt(0.5)
    doc = load_glb(glb(quad(node_transform=NodeSpec(rotation=(0.0, 0.0, half, half)))))
    # 90 degrees about Z maps (1, 0, 0) -> (0, 1, 0)
    np.testing.assert_allclose(doc.meshes[0].positions[1], [0.0, 1.0, 0.0], atol=1e-6)


def test_matrix_node_translation(glb: GlbWriter) -> None:
    matrix = (1.0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 1.0, 0, 5.0, 6.0, 7.0, 1.0)  # column-major
    doc = load_glb(glb(quad(node_transform=NodeSpec(matrix=matrix))))
    expected = quad_mesh().positions + np.array([5, 6, 7], dtype=np.float32)
    np.testing.assert_allclose(doc.meshes[0].positions, expected, atol=1e-6)


def test_mesh_instanced_by_two_nodes_is_loaded_twice(glb: GlbWriter) -> None:
    def add_instance(g: pygltflib.GLTF2) -> None:
        g.nodes.append(pygltflib.Node(mesh=0, translation=[10.0, 0.0, 0.0]))
        g.scenes[0].nodes.append(1)

    doc = load_glb(glb(rewrite(quad(), add_instance)))
    assert len(doc.meshes) == 2
    assert doc.meshes[1].positions[:, 0].min() == 10.0


def test_without_scenes_root_nodes_are_inferred(glb: GlbWriter) -> None:
    def drop_scenes(g: pygltflib.GLTF2) -> None:
        g.scenes = []
        g.scene = None

    doc = load_glb(glb(rewrite(quad(node_transform=[NodeSpec(), NodeSpec()]), drop_scenes)))
    assert len(doc.meshes) == 1


def test_missing_uvs(glb: GlbWriter) -> None:
    mesh = quad_mesh()
    doc = load_glb(glb(from_mesh(mesh.positions, None, mesh.indices)))
    assert doc.meshes[0].uvs is None


def test_interleaved_buffer_views_honour_byte_stride(glb: GlbWriter) -> None:
    doc = load_glb(glb(quad(interleave=True)))
    mesh = quad_mesh()
    np.testing.assert_array_equal(doc.meshes[0].positions, mesh.positions)
    assert doc.meshes[0].uvs is not None
    np.testing.assert_array_equal(doc.meshes[0].uvs, mesh.uvs)


@pytest.mark.parametrize(("component", "tol"), [("uint16", 1 / 65535), ("uint8", 1 / 255)])
def test_normalized_integer_uvs_are_scaled(glb: GlbWriter, component: str, tol: float) -> None:
    doc = load_glb(glb(quad(uv_component=component)))
    expected = quad_mesh().uvs
    assert doc.meshes[0].uvs is not None
    assert expected is not None
    assert doc.meshes[0].uvs.dtype == np.float32
    np.testing.assert_allclose(doc.meshes[0].uvs, expected, atol=tol)


def test_normalized_signed_integers_clamp_to_minus_one(glb: GlbWriter) -> None:
    def make_signed(g: pygltflib.GLTF2) -> None:
        acc = g.accessors[g.meshes[0].primitives[0].attributes.TEXCOORD_0]
        acc.componentType = 5122  # SHORT, reinterpret the uint16 bytes
        acc.normalized = True

    doc = load_glb(glb(rewrite(quad(uv_component="uint16"), make_signed)))
    assert doc.meshes[0].uvs is not None
    assert doc.meshes[0].uvs.min() >= -1.0
    assert doc.meshes[0].uvs.max() <= 1.0


def test_unindexed_primitive_generates_indices(glb: GlbWriter) -> None:
    doc = load_glb(glb(quad(indexed=False)))
    np.testing.assert_array_equal(doc.meshes[0].indices, np.arange(6).reshape(2, 3))


def test_accessor_without_buffer_view_is_zeros(glb: GlbWriter) -> None:
    def detach(g: pygltflib.GLTF2) -> None:
        g.accessors[g.meshes[0].primitives[0].attributes.TEXCOORD_0].bufferView = None

    doc = load_glb(glb(rewrite(quad(), detach)))
    assert doc.meshes[0].uvs is not None
    assert not doc.meshes[0].uvs.any()


def test_empty_accessor_count(glb: GlbWriter) -> None:
    def empty(g: pygltflib.GLTF2) -> None:
        prim = g.meshes[0].primitives[0]
        g.accessors[prim.attributes.POSITION].count = 0
        g.accessors[prim.attributes.TEXCOORD_0].count = 0
        prim.indices = None

    doc = load_glb(glb(rewrite(quad(), empty)))
    assert doc.meshes[0].positions.shape == (0, 3)
    assert doc.meshes[0].indices.shape == (0, 3)


def test_large_mesh_uses_uint32_indices(glb: GlbWriter) -> None:
    n = 70_002
    positions = np.zeros((n, 3), dtype=np.float32)
    positions[:, 0] = np.arange(n)
    indices = np.arange(n, dtype=np.uint32).reshape(-1, 3)
    doc = load_glb(glb(from_mesh(positions, None, indices)))
    assert doc.meshes[0].indices.max() == n - 1


# ------------------------------------------------------------------------------ materials, images
def test_materials_factors_and_texture_refs(glb: GlbWriter) -> None:
    images, material = textured_set()
    spec = MaterialSpec(
        **{
            **material.__dict__,
            "base_color_factor": (0.5, 0.25, 1.0, 0.75),
            "metallic_factor": 0.2,
            "roughness_factor": 0.3,
            "emissive_factor": (1.0, 0.5, 0.0),
            "normal_scale": 0.8,
            "occlusion_strength": 0.6,
            "emissive_strength": 4.0,
        }
    )
    doc = load_glb(glb(quad(images=images, materials=[spec])))
    (mat,) = doc.materials
    assert mat.name == "mat"
    assert mat.base_color_factor == (0.5, 0.25, 1.0, 0.75)
    assert (mat.metallic_factor, mat.roughness_factor) == (0.2, 0.3)
    assert mat.emissive_factor == (1.0, 0.5, 0.0)
    assert (mat.normal_scale, mat.occlusion_strength, mat.emissive_strength) == (0.8, 0.6, 4.0)
    assert mat.base_color_texture == GlbTextureRef(texture_index=0, image_index=0, tex_coord=0)
    assert mat.metallic_roughness_texture == GlbTextureRef(1, 1)
    assert mat.normal_texture == GlbTextureRef(2, 2)
    assert mat.occlusion_texture == GlbTextureRef(3, 3)
    assert mat.emissive_texture == GlbTextureRef(4, 4)
    assert doc.meshes[0].material_index == 0
    assert [img.name for img in doc.images] == [i.name for i in images]


def test_material_defaults_when_pbr_block_missing(glb: GlbWriter) -> None:
    def strip(g: pygltflib.GLTF2) -> None:
        g.materials[0].pbrMetallicRoughness = None
        g.materials[0].emissiveFactor = None

    doc = load_glb(glb(rewrite(quad(materials=[MaterialSpec()]), strip)))
    (mat,) = doc.materials
    assert mat.base_color_factor == (1.0, 1.0, 1.0, 1.0)
    assert (mat.metallic_factor, mat.roughness_factor) == (1.0, 1.0)
    assert mat.emissive_factor == (0.0, 0.0, 0.0)
    assert mat.base_color_texture is None


def test_png_and_jpeg_images_by_buffer_view_and_data_uri(glb: GlbWriter) -> None:
    pixels = solid_image((200, 100, 50, 255)).pixels
    images = [
        ImageSpec(pixels=pixels, fmt="png", name="p"),
        ImageSpec(pixels=pixels, fmt="jpeg", name="j"),
        ImageSpec(pixels=pixels, fmt="png", name="pd", embed="datauri"),
        ImageSpec(pixels=pixels, fmt="jpeg", name="jd", embed="datauri"),
    ]
    doc = load_glb(glb(quad(images=images)))
    assert [img.mime_type for img in doc.images] == ["image/png", "image/jpeg"] * 2
    assert doc.images[0].data[:4] == b"\x89PNG"
    assert doc.images[1].data[:3] == b"\xff\xd8\xff"
    assert doc.images[2].data == doc.images[0].data
    assert doc.images[3].data == doc.images[1].data


def test_external_image_file_next_to_glb(tmp_path: Path) -> None:
    png = solid_image((1, 2, 3, 4))
    (tmp_path / "tex image.png").write_bytes(encode_image(png))

    def externalise(g: pygltflib.GLTF2) -> None:
        g.images[0] = pygltflib.Image(uri="tex%20image.png")

    path = write_glb(tmp_path / "m.glb", rewrite(quad(images=[png]), externalise))
    doc = load_glb(path)
    assert doc.images[0].data == encode_image(png)
    assert doc.images[0].mime_type == "image/png"


def test_sniff_image_mime() -> None:
    assert sniff_image_mime(b"\x89PNG\r\n\x1a\nxxx") == "image/png"
    assert sniff_image_mime(b"\xff\xd8\xff\xe0") == "image/jpeg"
    assert sniff_image_mime(b"RIFF....WEBP") is None


# ------------------------------------------------------------------------------ errors
def test_missing_file_and_garbage(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_glb(tmp_path / "nope.glb")
    bad = tmp_path / "bad.glb"
    bad.write_bytes(b"not a glb at all" * 4)
    with pytest.raises(UnsupportedGlbError):
        load_glb(bad)


@pytest.mark.parametrize(
    ("data", "fragment"),
    [
        (quad(sparse=True), "sparse"),
        (quad(mode=5), "mode 5"),
        (quad(mode=0), "mode 0"),
        (quad(draco=True), "draco"),
        (quad(meshopt=True), "meshopt"),
        (quad(extensions_required=("KHR_something",)), "KHR_something"),
        (
            quad(
                images=[solid_image((1, 2, 3, 4))],
                materials=[MaterialSpec(base_color_image=0, texture_transform=True)],
            ),
            "KHR_texture_transform",
        ),
    ],
    ids=["sparse", "strip", "points", "draco", "meshopt", "required-ext", "texture-transform"],
)
def test_unsupported_structures_raise(glb: GlbWriter, data: bytes, fragment: str) -> None:
    with pytest.raises(UnsupportedGlbError, match=fragment):
        load_glb(glb(data))


def test_required_mesh_quantization_is_allowed(glb: GlbWriter) -> None:
    doc = load_glb(glb(quad(extensions_required=("KHR_mesh_quantization",))))
    assert len(doc.meshes) == 1


def _set_attr(path: str, value: object) -> Callable[[pygltflib.GLTF2], None]:
    def mutate(g: pygltflib.GLTF2) -> None:
        target: Any = g
        parts = path.split(".")
        for part in parts[:-1]:
            target = target[int(part)] if part.isdigit() else getattr(target, part)
        setattr(target, parts[-1], value)

    return mutate


@pytest.mark.parametrize(
    ("mutate", "fragment"),
    [
        (_set_attr("accessors.0.count", 1000), "past the end"),
        (_set_attr("bufferViews.0.byteLength", 10_000), "exceeds its buffer"),
        (_set_attr("accessors.0.componentType", 9999), "unknown layout"),
        (_set_attr("accessors.0.type", "VEC2"), "not VEC3"),
        (_set_attr("meshes.0.primitives.0.attributes.POSITION", None), "no POSITION"),
        (_set_attr("accessors.2.count", 4), "multiple of 3"),
        (_set_attr("accessors.0.count", 3), "index out of range"),
        (_set_attr("images.0.bufferView", None), "neither bufferView nor uri"),
        (_set_attr("images.0.uri", "missing.png"), "not found"),
        (_set_attr("images.0.mimeType", "image/webp"), "unsupported image format"),
        (_set_attr("textures.0.source", None), "no image source"),
    ],
    ids=[
        "accessor-overrun",
        "bufferview-overrun",
        "component-type",
        "position-vec2",
        "no-position",
        "index-count",
        "index-range",
        "image-no-source",
        "image-missing-file",
        "image-webp",
        "texture-no-source",
    ],
)
def test_malformed_documents_raise(
    glb: GlbWriter, mutate: Callable[[pygltflib.GLTF2], None], fragment: str
) -> None:
    def prepare(g: pygltflib.GLTF2) -> None:
        if fragment == "not found":
            g.images[0].bufferView = None
        mutate(g)

    base = quad(images=[solid_image((1, 2, 3, 4))], materials=[MaterialSpec(base_color_image=0)])
    with pytest.raises(UnsupportedGlbError, match=fragment):
        load_glb(glb(rewrite(base, prepare)))


def test_malformed_data_uris_raise(glb: GlbWriter) -> None:
    for uri in ("data:image/png,plain", "data:image/png;base64,@@@"):

        def to_uri(g: pygltflib.GLTF2, uri: str = uri) -> None:
            g.images[0] = pygltflib.Image(uri=uri)

        data = rewrite(quad(images=[solid_image((1, 2, 3, 4))]), to_uri)
        with pytest.raises(UnsupportedGlbError, match="data URI"):
            load_glb(glb(data))


def test_buffer_pointing_at_missing_external_file_raises(glb: GlbWriter) -> None:
    with pytest.raises(UnsupportedGlbError, match="not found"):
        load_glb(glb(rewrite(quad(), _set_attr("buffers.0.uri", "missing.bin"))))


def test_glb_without_bin_chunk_raises(glb: GlbWriter) -> None:
    gltf = pygltflib.GLTF2.load_from_bytes(quad())
    json_blob = gltf.gltf_to_json(separators=(",", ":"), indent=None).encode()
    json_blob += b" " * (-len(json_blob) % 4)
    header = b"glTF" + struct.pack("<II", 2, 12 + 8 + len(json_blob))
    data = header + struct.pack("<I", len(json_blob)) + b"JSON" + json_blob
    with pytest.raises(UnsupportedGlbError, match="no binary data"):
        load_glb(glb(data))


def test_cube_mesh_fixture_matches_loader_counts(glb: GlbWriter) -> None:
    doc = load_glb(glb(cube()))
    assert len(np.unique(doc.meshes[0].positions, axis=0)) == len(
        np.unique(cube_mesh().positions, axis=0)
    )
