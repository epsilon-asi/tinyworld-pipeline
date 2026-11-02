"""Pure-Python construction of valid GLB files for tests (plan 15 §3.5).

Every builder returns the raw bytes of a binary glTF 2.0 container assembled with ``pygltflib``.
The geometry helpers (:func:`quad`, :func:`cube`, :func:`from_mesh`) accept knobs that mimic what a
texturing provider or a different exporter may legitimately do to a mesh (vertex shuffling, seam
re-splitting, primitive splitting, uniform normalisation, node nesting) as well as knobs that
produce the *unsupported* structures the loader must reject (sparse accessors, Draco / meshopt
extensions, non-triangle modes, ``KHR_texture_transform``).

Embedded images are tiny (2x2 or 4x4) PNG/JPEG files with known pixel values so texture extraction
and URP packing can be asserted exactly.
"""

from __future__ import annotations

import base64
import io
import struct
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import pygltflib
from PIL import Image as PilImage

MODE_POINTS = 0
MODE_TRIANGLES = 4
MODE_TRIANGLE_STRIP = 5

_TARGET_ARRAY_BUFFER = 34962
_TARGET_ELEMENT_ARRAY_BUFFER = 34963
_COMPONENT_FLOAT = 5126
_COMPONENT_UBYTE = 5121
_COMPONENT_USHORT = 5123
_COMPONENT_UINT = 5125

UvComponent = Literal["float32", "uint16", "uint8"]
ImageFormat = Literal["png", "jpeg"]
ImageEmbed = Literal["bufferview", "datauri"]

Float3 = tuple[float, float, float]
Float4 = tuple[float, float, float, float]


# --------------------------------------------------------------------------------------- specs
@dataclass(frozen=True)
class MeshArrays:
    """Plain arrays describing an indexed triangle mesh."""

    positions: npt.NDArray[np.float32]  # (N, 3)
    uvs: npt.NDArray[np.float32] | None  # (N, 2)
    indices: npt.NDArray[np.uint32]  # (M, 3)


@dataclass(frozen=True)
class ImageSpec:
    """An image to embed: ``pixels`` is ``(H, W, 3|4)`` uint8."""

    pixels: npt.NDArray[np.uint8]
    fmt: ImageFormat = "png"
    name: str | None = None
    embed: ImageEmbed = "bufferview"


@dataclass(frozen=True)
class MaterialSpec:
    """A ``pbrMetallicRoughness`` material; ``*_image`` fields index into ``images``."""

    name: str | None = None
    base_color_image: int | None = None
    metallic_roughness_image: int | None = None
    normal_image: int | None = None
    occlusion_image: int | None = None
    emissive_image: int | None = None
    base_color_factor: Float4 = (1.0, 1.0, 1.0, 1.0)
    metallic_factor: float = 1.0
    roughness_factor: float = 1.0
    emissive_factor: Float3 = (0.0, 0.0, 0.0)
    normal_scale: float = 1.0
    occlusion_strength: float = 1.0
    emissive_strength: float | None = None  # KHR_materials_emissive_strength
    texture_transform: bool = False  # adds KHR_texture_transform to baseColorTexture


@dataclass(frozen=True)
class NodeSpec:
    """One node in a parent→child chain; the last node of the chain carries the mesh."""

    translation: Float3 | None = None
    rotation: Float4 | None = None  # quaternion (x, y, z, w)
    scale: Float3 | None = None
    matrix: tuple[float, ...] | None = None  # 16 floats, column-major
    name: str | None = None


# --------------------------------------------------------------------------------------- images
def solid_image(
    rgba: tuple[int, ...],
    *,
    size: int = 2,
    fmt: ImageFormat = "png",
    name: str | None = None,
    embed: ImageEmbed = "bufferview",
) -> ImageSpec:
    """A ``sizexsize`` image where every pixel equals ``rgba`` (3 or 4 channels)."""
    pixels = np.tile(np.asarray(rgba, dtype=np.uint8), (size, size, 1))
    return ImageSpec(pixels=pixels, fmt=fmt, name=name, embed=embed)


def gradient_image(
    *, size: int = 4, fmt: ImageFormat = "png", name: str | None = None
) -> ImageSpec:
    """A ``sizexsize`` RGBA image whose pixels are all distinct (row/col encoded in R/G)."""
    rows = np.arange(size, dtype=np.uint8)
    r = np.repeat(rows[:, None] * (255 // max(size - 1, 1)), size, axis=1)
    g = np.repeat(rows[None, :] * (255 // max(size - 1, 1)), size, axis=0)
    b = np.full((size, size), 7, dtype=np.uint8)
    a = np.full((size, size), 255, dtype=np.uint8)
    pixels = np.stack([r, g, b, a], axis=-1).astype(np.uint8)
    return ImageSpec(pixels=pixels, fmt=fmt, name=name)


def encode_image(spec: ImageSpec) -> bytes:
    """Encode an :class:`ImageSpec` to PNG or JPEG bytes with Pillow."""
    pixels = spec.pixels
    if spec.fmt == "jpeg":
        img = PilImage.fromarray(pixels[..., :3], mode="RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=100, subsampling=0)
        return buf.getvalue()
    mode = "RGBA" if pixels.shape[-1] == 4 else "RGB"
    img = PilImage.fromarray(pixels, mode=mode)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def textured_set(
    *, size: int = 2, fmt: ImageFormat = "png"
) -> tuple[list[ImageSpec], MaterialSpec]:
    """Five known-value images plus a material referencing all of them.

    Pixel values (RGBA): basecolor (200, 100, 50, 255); metallic_roughness (0, 128, 64, 255) i.e.
    roughness 128 in G and metallic 64 in B; normal (128, 128, 255, 255); occlusion
    (180, 180, 180, 255); emissive (10, 20, 30, 255).
    """
    images = [
        solid_image((200, 100, 50, 255), size=size, fmt=fmt, name="basecolor"),
        solid_image((0, 128, 64, 255), size=size, fmt=fmt, name="metallic_roughness"),
        solid_image((128, 128, 255, 255), size=size, fmt=fmt, name="normal"),
        solid_image((180, 180, 180, 255), size=size, fmt=fmt, name="occlusion"),
        solid_image((10, 20, 30, 255), size=size, fmt=fmt, name="emissive"),
    ]
    material = MaterialSpec(
        name="mat",
        base_color_image=0,
        metallic_roughness_image=1,
        normal_image=2,
        occlusion_image=3,
        emissive_image=4,
    )
    return images, material


# --------------------------------------------------------------------------------------- geometry
def quad_mesh() -> MeshArrays:
    """Unit square in the XY plane, 4 vertices, 2 triangles, UVs equal to XY."""
    positions = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float32)
    uvs = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
    indices = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)
    return MeshArrays(positions, uvs, indices)


def cube_mesh() -> MeshArrays:
    """Unit cube centred at the origin, 24 vertices (4 per face, own UVs), 12 triangles."""
    faces = [
        ((0, 0, 1), (1, 0, 0), (0, 1, 0)),
        ((0, 0, -1), (-1, 0, 0), (0, 1, 0)),
        ((1, 0, 0), (0, 0, -1), (0, 1, 0)),
        ((-1, 0, 0), (0, 0, 1), (0, 1, 0)),
        ((0, 1, 0), (1, 0, 0), (0, 0, -1)),
        ((0, -1, 0), (1, 0, 0), (0, 0, 1)),
    ]
    corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=np.float32)
    positions: list[npt.NDArray[np.float32]] = []
    uvs: list[npt.NDArray[np.float32]] = []
    indices: list[list[int]] = []
    for face_index, (normal, right, up) in enumerate(faces):
        n = np.asarray(normal, dtype=np.float32)
        r = np.asarray(right, dtype=np.float32)
        u = np.asarray(up, dtype=np.float32)
        for cx, cy in corners:
            positions.append(0.5 * (n + cx * r + cy * u))
            uvs.append(np.array([(cx + 1) / 2, (cy + 1) / 2], dtype=np.float32))
        base = face_index * 4
        indices.append([base, base + 1, base + 2])
        indices.append([base, base + 2, base + 3])
    return MeshArrays(
        np.asarray(positions, dtype=np.float32),
        np.asarray(uvs, dtype=np.float32),
        np.asarray(indices, dtype=np.uint32),
    )


def quad(**kwargs: Any) -> bytes:
    """GLB of :func:`quad_mesh`; keyword arguments are forwarded to :func:`from_mesh`."""
    mesh = quad_mesh()
    return from_mesh(mesh.positions, mesh.uvs, mesh.indices, **kwargs)


def cube(**kwargs: Any) -> bytes:
    """GLB of :func:`cube_mesh`; keyword arguments are forwarded to :func:`from_mesh`."""
    mesh = cube_mesh()
    return from_mesh(mesh.positions, mesh.uvs, mesh.indices, **kwargs)


def write_glb(path: Path, data: bytes) -> Path:
    """Write GLB bytes to ``path`` (creating parent directories) and return the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def rewrite(data: bytes, mutate: Callable[[pygltflib.GLTF2], None]) -> bytes:
    """Apply ``mutate`` to the parsed glTF JSON of ``data`` and re-pack it without touching buffers.

    Unlike ``GLTF2.save_to_bytes`` this keeps buffer views exactly as edited, so tests can produce
    structures the loader must reject (out-of-range views, extra data-URI buffers, missing sources).
    """
    gltf = pygltflib.GLTF2.load_from_bytes(data)
    blob = bytes(gltf.binary_blob() or b"")
    mutate(gltf)
    json_blob = gltf.gltf_to_json(separators=(",", ":"), indent=None).encode("utf-8")
    json_blob += b" " * (-len(json_blob) % 4)
    blob += b"\0" * (-len(blob) % 4)
    total = 12 + 8 + len(json_blob) + 8 + len(blob)
    return b"".join(
        [
            b"glTF",
            struct.pack("<II", 2, total),
            struct.pack("<I", len(json_blob)),
            b"JSON",
            json_blob,
            struct.pack("<I", len(blob)),
            b"BIN\0",
            blob,
        ]
    )


# --------------------------------------------------------------------------------------- builder
class _Blob:
    """Accumulates 4-byte aligned buffer views in a single binary buffer."""

    def __init__(self, gltf: pygltflib.GLTF2) -> None:
        self.gltf = gltf
        self.data = bytearray()

    def add(self, payload: bytes, *, target: int | None = None, stride: int | None = None) -> int:
        while len(self.data) % 4:
            self.data += b"\0"
        view = pygltflib.BufferView(
            buffer=0, byteOffset=len(self.data), byteLength=len(payload), target=target
        )
        if stride is not None:
            view.byteStride = stride
        self.data += payload
        self.gltf.bufferViews.append(view)
        return len(self.gltf.bufferViews) - 1

    def accessor(
        self,
        view: int,
        *,
        component_type: int,
        count: int,
        kind: str,
        byte_offset: int = 0,
        normalized: bool = False,
        bounds: npt.NDArray[Any] | None = None,
    ) -> int:
        acc = pygltflib.Accessor(
            bufferView=view,
            byteOffset=byte_offset,
            componentType=component_type,
            normalized=normalized,
            count=count,
            type=kind,
        )
        if bounds is not None and len(bounds):
            acc.min = [float(v) for v in bounds.min(axis=0)]
            acc.max = [float(v) for v in bounds.max(axis=0)]
        self.gltf.accessors.append(acc)
        return len(self.gltf.accessors) - 1


def _prepare_arrays(
    positions: npt.ArrayLike,
    uvs: npt.ArrayLike | None,
    indices: npt.ArrayLike,
    *,
    split_seams: bool,
    shuffle: bool,
    scale: float,
    translate: Float3,
    seed: int,
) -> MeshArrays:
    pos = np.asarray(positions, dtype=np.float32) * np.float32(scale)
    pos = (pos + np.asarray(translate, dtype=np.float32)).astype(np.float32)
    uv = None if uvs is None else np.asarray(uvs, dtype=np.float32).reshape(-1, 2)
    tri = np.asarray(indices, dtype=np.uint32).reshape(-1, 3)
    if split_seams:
        flat = tri.ravel()
        pos = pos[flat]
        uv = None if uv is None else uv[flat]
        tri = np.arange(flat.size, dtype=np.uint32).reshape(-1, 3)
    if shuffle:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(pos))
        inverse = np.empty_like(perm)
        inverse[perm] = np.arange(len(perm))
        pos = pos[perm]
        uv = None if uv is None else uv[perm]
        tri = inverse[tri].astype(np.uint32)
        tri = tri[rng.permutation(len(tri))]
        tri = np.roll(tri, shift=1, axis=1)  # rotate each triangle's winding start
    return MeshArrays(pos, uv, tri.astype(np.uint32))


def _compact(mesh: MeshArrays, tri: npt.NDArray[np.uint32]) -> MeshArrays:
    """Keep only the vertices referenced by ``tri`` and remap indices."""
    used, inverse = np.unique(tri.ravel(), return_inverse=True)
    uv = None if mesh.uvs is None else mesh.uvs[used]
    return MeshArrays(mesh.positions[used], uv, inverse.astype(np.uint32).reshape(-1, 3))


def _encode_uvs(uv: npt.NDArray[np.float32], component: UvComponent) -> tuple[bytes, int, bool]:
    if component == "uint16":
        return np.rint(uv * 65535).astype("<u2").tobytes(), _COMPONENT_USHORT, True
    if component == "uint8":
        return np.rint(uv * 255).astype("<u1").tobytes(), _COMPONENT_UBYTE, True
    return uv.astype("<f4").tobytes(), _COMPONENT_FLOAT, False


def _add_sparse(blob: _Blob, accessor_index: int) -> None:
    idx_view = blob.add(np.array([0], dtype="<u2").tobytes())
    val_view = blob.add(np.array([0, 0, 0], dtype="<f4").tobytes())
    acc = blob.gltf.accessors[accessor_index]
    acc.sparse = pygltflib.Sparse(
        count=1,
        indices=pygltflib.AccessorSparseIndices(
            bufferView=idx_view, componentType=_COMPONENT_USHORT
        ),
        values=pygltflib.AccessorSparseValues(bufferView=val_view),
    )


def _add_vertex_data(
    blob: _Blob, mesh: MeshArrays, *, interleave: bool, uv_component: UvComponent, sparse: bool
) -> pygltflib.Attributes:
    count = len(mesh.positions)
    if interleave and mesh.uvs is not None and uv_component == "float32":
        packed = np.concatenate([mesh.positions, mesh.uvs], axis=1).astype("<f4")
        view = blob.add(packed.tobytes(), target=_TARGET_ARRAY_BUFFER, stride=20)
        pos_acc = blob.accessor(
            view, component_type=_COMPONENT_FLOAT, count=count, kind="VEC3", bounds=mesh.positions
        )
        uv_acc = blob.accessor(
            view, component_type=_COMPONENT_FLOAT, count=count, kind="VEC2", byte_offset=12
        )
        attrs = pygltflib.Attributes(POSITION=pos_acc, TEXCOORD_0=uv_acc)
    else:
        view = blob.add(mesh.positions.astype("<f4").tobytes(), target=_TARGET_ARRAY_BUFFER)
        pos_acc = blob.accessor(
            view, component_type=_COMPONENT_FLOAT, count=count, kind="VEC3", bounds=mesh.positions
        )
        attrs = pygltflib.Attributes(POSITION=pos_acc)
        if mesh.uvs is not None:
            payload, ctype, normalized = _encode_uvs(mesh.uvs, uv_component)
            uv_view = blob.add(payload, target=_TARGET_ARRAY_BUFFER)
            attrs.TEXCOORD_0 = blob.accessor(
                uv_view, component_type=ctype, count=count, kind="VEC2", normalized=normalized
            )
    if sparse:
        _add_sparse(blob, pos_acc)
    return attrs


def _add_indices(blob: _Blob, tri: npt.NDArray[np.uint32], vertex_count: int) -> int:
    if vertex_count <= 0xFFFF:
        payload, ctype = tri.astype("<u2").tobytes(), _COMPONENT_USHORT
    else:
        payload, ctype = tri.astype("<u4").tobytes(), _COMPONENT_UINT
    view = blob.add(payload, target=_TARGET_ELEMENT_ARRAY_BUFFER)
    return blob.accessor(view, component_type=ctype, count=tri.size, kind="SCALAR")


def _add_images(blob: _Blob, images: list[ImageSpec]) -> None:
    gltf = blob.gltf
    for index, spec in enumerate(images):
        payload = encode_image(spec)
        mime = "image/jpeg" if spec.fmt == "jpeg" else "image/png"
        if spec.embed == "datauri":
            uri = f"data:{mime};base64,{base64.b64encode(payload).decode('ascii')}"
            gltf.images.append(pygltflib.Image(uri=uri, name=spec.name))
        else:
            view = blob.add(payload)
            gltf.images.append(pygltflib.Image(bufferView=view, mimeType=mime, name=spec.name))
        gltf.textures.append(pygltflib.Texture(source=index))


def _texture_info(image: int | None) -> pygltflib.TextureInfo | None:
    return None if image is None else pygltflib.TextureInfo(index=image)


def _add_materials(gltf: pygltflib.GLTF2, materials: list[MaterialSpec]) -> None:
    for spec in materials:
        base = _texture_info(spec.base_color_image)
        if spec.texture_transform and base is not None:
            base.extensions = {"KHR_texture_transform": {"scale": [2.0, 2.0]}}
            _use_extension(gltf, "KHR_texture_transform")
        material = pygltflib.Material(
            name=spec.name,
            pbrMetallicRoughness=pygltflib.PbrMetallicRoughness(
                baseColorFactor=list(spec.base_color_factor),
                baseColorTexture=base,
                metallicFactor=spec.metallic_factor,
                roughnessFactor=spec.roughness_factor,
                metallicRoughnessTexture=_texture_info(spec.metallic_roughness_image),
            ),
            emissiveFactor=list(spec.emissive_factor),
            emissiveTexture=_texture_info(spec.emissive_image),
        )
        if spec.normal_image is not None:
            material.normalTexture = pygltflib.NormalMaterialTexture(
                index=spec.normal_image, scale=spec.normal_scale
            )
        if spec.occlusion_image is not None:
            material.occlusionTexture = pygltflib.OcclusionTextureInfo(
                index=spec.occlusion_image, strength=spec.occlusion_strength
            )
        if spec.emissive_strength is not None:
            material.extensions = {
                "KHR_materials_emissive_strength": {"emissiveStrength": spec.emissive_strength}
            }
            _use_extension(gltf, "KHR_materials_emissive_strength")
        gltf.materials.append(material)


def _use_extension(gltf: pygltflib.GLTF2, name: str) -> None:
    if name not in gltf.extensionsUsed:
        gltf.extensionsUsed.append(name)


def _add_nodes(gltf: pygltflib.GLTF2, chain: list[NodeSpec], mesh_index: int) -> None:
    for depth, spec in enumerate(chain):
        node = pygltflib.Node(name=spec.name)
        if spec.matrix is not None:
            node.matrix = list(spec.matrix)
        else:
            node.translation = None if spec.translation is None else list(spec.translation)
            node.rotation = None if spec.rotation is None else list(spec.rotation)
            node.scale = None if spec.scale is None else list(spec.scale)
        if depth == len(chain) - 1:
            node.mesh = mesh_index
        else:
            node.children = [depth + 1]
        gltf.nodes.append(node)
    gltf.scenes.append(pygltflib.Scene(nodes=[0]))
    gltf.scene = 0


def _resolve_materials(
    primitive_count: int,
    materials: list[MaterialSpec] | None,
    primitive_materials: list[int | None] | None,
) -> list[int | None]:
    if primitive_materials is not None:
        return list(primitive_materials)
    if not materials:
        return [None] * primitive_count
    return [i % len(materials) for i in range(primitive_count)]


def from_mesh(
    positions: npt.ArrayLike,
    uvs: npt.ArrayLike | None,
    indices: npt.ArrayLike,
    *,
    split_seams: bool = False,
    shuffle: bool = False,
    scale: float = 1.0,
    translate: Float3 = (0.0, 0.0, 0.0),
    primitives: int = 1,
    materials: list[MaterialSpec] | None = None,
    primitive_materials: list[int | None] | None = None,
    images: list[ImageSpec] | None = None,
    node_transform: NodeSpec | list[NodeSpec] | None = None,
    seed: int = 0,
    indexed: bool = True,
    interleave: bool = False,
    uv_component: UvComponent = "float32",
    sparse: bool = False,
    mode: int = MODE_TRIANGLES,
    draco: bool = False,
    meshopt: bool = False,
    extensions_required: tuple[str, ...] = (),
) -> bytes:
    """Build a GLB from arrays.

    Args:
        positions: ``(N, 3)`` vertex positions.
        uvs: ``(N, 2)`` texture coordinates or ``None`` for a mesh without ``TEXCOORD_0``.
        indices: ``(M, 3)`` (or flat ``3M``) triangle indices.
        split_seams: Give every triangle its own three vertices (as an exporter splitting seams
            aggressively would). Does not change the surface.
        shuffle: Permute vertex order, triangle order and winding start deterministically.
        scale: Uniform scale applied to positions (mimics provider normalisation).
        translate: Translation applied after ``scale``.
        primitives: Split the triangle list into this many primitives (vertices compacted).
        materials: Material specs; primitive ``i`` uses ``materials[i % len(materials)]`` unless
            ``primitive_materials`` is given.
        primitive_materials: Explicit material index (or ``None``) per primitive.
        images: Images to embed; one texture is created per image (texture ``i`` → image ``i``).
        node_transform: A node or chain of nested nodes; the mesh hangs off the last one.
        seed: Seed for ``shuffle``.
        indexed: ``False`` writes non-indexed primitives (vertices consumed three at a time).
        interleave: Store positions and float UVs in one strided buffer view (``byteStride=20``).
        uv_component: Component type for UVs; integer types are stored ``normalized``.
        sparse: Attach a sparse block to the position accessor (loader must reject).
        mode: Primitive mode; anything but ``4`` is rejected by the loader.
        draco: Mark primitives with ``KHR_draco_mesh_compression`` (loader must reject).
        meshopt: Mark the first buffer view with ``EXT_meshopt_compression`` (loader must reject).
        extensions_required: Extra names to list under ``extensionsRequired``.

    Returns:
        The GLB file contents.
    """
    mesh = _prepare_arrays(
        positions,
        uvs,
        indices,
        split_seams=split_seams or not indexed,
        shuffle=shuffle,
        scale=scale,
        translate=translate,
        seed=seed,
    )
    gltf = pygltflib.GLTF2(asset=pygltflib.Asset(version="2.0", generator="glb_builder"))
    blob = _Blob(gltf)
    material_of = _resolve_materials(primitives, materials, primitive_materials)
    prims: list[pygltflib.Primitive] = []
    for chunk, material_index in zip(
        np.array_split(mesh.indices, primitives), material_of, strict=True
    ):
        part = _compact(mesh, chunk)
        attrs = _add_vertex_data(
            blob, part, interleave=interleave, uv_component=uv_component, sparse=sparse
        )
        prim = pygltflib.Primitive(attributes=attrs, mode=mode, material=material_index)
        if indexed:
            prim.indices = _add_indices(blob, part.indices, len(part.positions))
        if draco:
            prim.extensions = {"KHR_draco_mesh_compression": {"bufferView": 0, "attributes": {}}}
            _use_extension(gltf, "KHR_draco_mesh_compression")
        prims.append(prim)
    gltf.meshes.append(pygltflib.Mesh(primitives=prims, name="mesh"))
    _add_images(blob, images or [])
    _add_materials(gltf, materials or [])
    if meshopt:
        gltf.bufferViews[0].extensions = {"EXT_meshopt_compression": {"buffer": 0}}
        _use_extension(gltf, "EXT_meshopt_compression")
    gltf.extensionsRequired.extend(extensions_required)
    chain = (
        [NodeSpec()]
        if node_transform is None
        else ([node_transform] if isinstance(node_transform, NodeSpec) else list(node_transform))
    )
    _add_nodes(gltf, chain, mesh_index=0)
    gltf.buffers.append(pygltflib.Buffer(byteLength=len(blob.data)))
    gltf.set_binary_blob(bytes(blob.data))
    return b"".join(gltf.save_to_bytes())
