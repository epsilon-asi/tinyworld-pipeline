"""Load a GLB into plain numpy arrays (plan 11 §1).

The loader turns a binary glTF 2.0 file into a :class:`GlbDocument`: one :class:`GlbMesh` per
primitive (positions already in world space), the materials referenced by the file and the embedded
images as bytes. It is deliberately narrow: the pipeline only ever reads GLBs that we exported
ourselves or that Tripo returned, both of which are plain, uncompressed, Y-up, triangle-only files.

Extension handling
    * ``extensionsRequired`` containing anything except ``KHR_mesh_quantization`` →
      :class:`UnsupportedGlbError` (we cannot honour an extension we do not implement).
    * ``KHR_draco_mesh_compression`` on a primitive → :class:`UnsupportedGlbError`.
    * ``EXT_meshopt_compression`` on a buffer view → :class:`UnsupportedGlbError`.
    * ``KHR_texture_transform`` on any texture reference → :class:`UnsupportedGlbError`
      (plan 11 §5.1: never seen from Tripo; fail loudly rather than mis-map textures).
    * ``KHR_materials_emissive_strength`` → read into :attr:`GlbMaterial.emissive_strength`.
    * Every other *optional* extension (``extensionsUsed``) is ignored.

Accessor handling
    * Any component type / element type; ``byteStride`` honoured; normalized integer attributes
      are converted to floats per the glTF specification.
    * Sparse accessors → :class:`UnsupportedGlbError`.
    * Primitive modes other than ``TRIANGLES`` (4) → :class:`UnsupportedGlbError`; non-indexed
      triangle primitives are accepted (indices generated).
"""

from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote

import numpy as np
import numpy.typing as npt
from pygltflib import GLTF2

from grounded_pipeline.domain.errors import UnsupportedGlbError

MODE_TRIANGLES = 4
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
JPEG_MAGIC = b"\xff\xd8\xff"

_ALLOWED_REQUIRED_EXTENSIONS = frozenset({"KHR_mesh_quantization"})
_COMPONENT_DTYPES: dict[int, str] = {
    5120: "<i1",
    5121: "<u1",
    5122: "<i2",
    5123: "<u2",
    5125: "<u4",
    5126: "<f4",
}
_TYPE_COMPONENTS: dict[str, int] = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
    "MAT2": 4,
    "MAT3": 9,
    "MAT4": 16,
}


# ------------------------------------------------------------------------------------ value objects
@dataclass(frozen=True)
class GlbTextureRef:
    """A material's reference to a texture, resolved down to the image it samples."""

    texture_index: int
    image_index: int
    tex_coord: int = 0


@dataclass(frozen=True)
class GlbMaterial:
    """``pbrMetallicRoughness`` material: factors plus the texture references that exist."""

    name: str | None
    base_color_factor: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    metallic_factor: float = 1.0
    roughness_factor: float = 1.0
    emissive_factor: tuple[float, float, float] = (0.0, 0.0, 0.0)
    normal_scale: float = 1.0
    occlusion_strength: float = 1.0
    emissive_strength: float = 1.0
    base_color_texture: GlbTextureRef | None = None
    metallic_roughness_texture: GlbTextureRef | None = None
    normal_texture: GlbTextureRef | None = None
    occlusion_texture: GlbTextureRef | None = None
    emissive_texture: GlbTextureRef | None = None


@dataclass(frozen=True)
class GlbImage:
    """An embedded image: encoded bytes plus MIME type (``image/png`` or ``image/jpeg``)."""

    data: bytes
    mime_type: str
    name: str | None = None


@dataclass(frozen=True)
class GlbMesh:
    """One primitive with positions in world space (node transforms applied)."""

    positions: npt.NDArray[np.float32]  # (N, 3)
    uvs: npt.NDArray[np.float32] | None  # (N, 2) TEXCOORD_0
    indices: npt.NDArray[np.uint32]  # (M, 3)
    material_index: int | None


@dataclass(frozen=True)
class GlbDocument:
    """Everything the pipeline needs from a GLB."""

    meshes: list[GlbMesh]  # one per primitive, all nodes traversed
    materials: list[GlbMaterial]
    images: list[GlbImage]


# ------------------------------------------------------------------------------------ public API
def load_glb(path: Path) -> GlbDocument:
    """Parse a GLB file.

    Args:
        path: Path to a ``.glb`` file.

    Returns:
        The decoded document; mesh positions are world space, Y-up as stored.

    Raises:
        FileNotFoundError: ``path`` does not exist.
        UnsupportedGlbError: The file is not a GLB or uses a structure the loader rejects (see
            the module docstring).
    """
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        gltf = GLTF2.load_binary(str(path))
    except (OSError, ValueError, KeyError) as exc:
        raise UnsupportedGlbError(f"{path.name} is not a readable GLB: {exc}", cause=exc) from exc
    reader = _Reader(gltf, path.parent)
    _check_extensions(gltf)
    meshes = [
        _load_primitive(reader, prim, matrix)
        for mesh_index, matrix in _mesh_instances(gltf)
        for prim in gltf.meshes[mesh_index].primitives
    ]
    materials = [_load_material(gltf, m) for m in gltf.materials]
    images = [_load_image(reader, img) for img in gltf.images]
    return GlbDocument(meshes=meshes, materials=materials, images=images)


def sniff_image_mime(data: bytes) -> str | None:
    """Return ``image/png`` / ``image/jpeg`` from the magic bytes, or ``None`` when unknown."""
    if data.startswith(PNG_MAGIC):
        return "image/png"
    if data.startswith(JPEG_MAGIC):
        return "image/jpeg"
    return None


# ------------------------------------------------------------------------------------ buffers
class _Reader:
    """Decodes accessors and buffer views against the GLB's buffers."""

    def __init__(self, gltf: Any, base_dir: Path) -> None:  # noqa: ANN401 - pygltflib is untyped
        self.gltf = gltf
        self.base_dir = base_dir
        self._buffers: dict[int, bytes] = {}

    def buffer(self, index: int) -> bytes:
        if index not in self._buffers:
            uri = self.gltf.buffers[index].uri
            if uri is None:
                data = self.gltf.binary_blob()
            elif uri.startswith("data:"):
                data = _decode_data_uri(uri)
            else:
                data = _read_external(self.base_dir, uri)
            if data is None:
                raise UnsupportedGlbError(f"buffer {index} has no binary data")
            self._buffers[index] = bytes(data)
        return self._buffers[index]

    def buffer_view(self, index: int) -> bytes:
        view = self.gltf.bufferViews[index]
        if "EXT_meshopt_compression" in (view.extensions or {}):
            raise UnsupportedGlbError("EXT_meshopt_compression buffer views are not supported")
        data = self.buffer(view.buffer)
        start = view.byteOffset or 0
        end = start + view.byteLength
        if end > len(data):
            raise UnsupportedGlbError(f"bufferView {index} exceeds its buffer")
        return data[start:end]

    def accessor(self, index: int) -> npt.NDArray[Any]:
        """Decode accessor ``index`` to a ``(count, components)`` array in its native dtype."""
        acc = self.gltf.accessors[index]
        if acc.sparse is not None:
            raise UnsupportedGlbError(
                f"accessor {index} is sparse; sparse accessors are unsupported"
            )
        try:
            dtype = np.dtype(_COMPONENT_DTYPES[acc.componentType])
            components = _TYPE_COMPONENTS[acc.type]
        except KeyError as exc:
            raise UnsupportedGlbError(f"accessor {index} has an unknown layout: {exc}") from exc
        count = int(acc.count)
        if acc.bufferView is None:
            return np.zeros((count, components), dtype=dtype)
        view = self.gltf.bufferViews[acc.bufferView]
        data = self.buffer_view(acc.bufferView)
        element = dtype.itemsize * components
        stride = int(view.byteStride) if view.byteStride else element
        offset = int(acc.byteOffset or 0)
        needed = offset + stride * (count - 1) + element if count else offset
        if needed > len(data):
            raise UnsupportedGlbError(f"accessor {index} reads past the end of its bufferView")
        if count == 0:
            return np.zeros((0, components), dtype=dtype)
        strided: npt.NDArray[Any] = np.ndarray(
            (count, components),
            dtype=dtype,
            buffer=data,
            offset=offset,
            strides=(stride, dtype.itemsize),
        )
        return np.array(strided)  # owning, contiguous copy

    def float_accessor(self, index: int) -> npt.NDArray[np.float32]:
        """Decode an attribute accessor to float32, applying ``normalized`` integer scaling."""
        raw = self.accessor(index)
        acc = self.gltf.accessors[index]
        if raw.dtype.kind in "iu" and acc.normalized:
            info = np.iinfo(raw.dtype)
            values = raw.astype(np.float32) / np.float32(info.max)
            if info.min < 0:
                values = np.maximum(values, np.float32(-1.0))
            return values.astype(np.float32)
        return raw.astype(np.float32)


def _decode_data_uri(uri: str) -> bytes:
    """Decode a base64 ``data:`` URI of any MIME type (pygltflib only handles octet-stream)."""
    header, _, payload = uri.partition(",")
    if not payload or not header.endswith(";base64"):
        raise UnsupportedGlbError("only base64 data URIs are supported")
    try:
        return base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise UnsupportedGlbError(f"malformed data URI: {exc}", cause=exc) from exc


def _read_external(base_dir: Path, uri: str) -> bytes:
    candidate = base_dir / unquote(uri)
    if not candidate.is_file():
        raise UnsupportedGlbError(f"external resource {uri!r} not found next to the GLB")
    return candidate.read_bytes()


# ------------------------------------------------------------------------------------ structure
def _check_extensions(gltf: Any) -> None:  # noqa: ANN401 - pygltflib is untyped
    unsupported = sorted(set(gltf.extensionsRequired or ()) - _ALLOWED_REQUIRED_EXTENSIONS)
    if unsupported:
        raise UnsupportedGlbError(
            f"GLB requires unsupported extension(s): {', '.join(unsupported)}",
            detail={"extensions_required": unsupported},
        )


def _node_matrix(node: Any) -> npt.NDArray[np.float64]:  # noqa: ANN401 - pygltflib is untyped
    if node.matrix:
        return np.asarray(node.matrix, dtype=np.float64).reshape(4, 4, order="F")
    matrix = np.eye(4)
    if node.scale:
        matrix = np.diag([*node.scale, 1.0])
    if node.rotation:
        matrix = _quaternion_matrix(node.rotation) @ matrix
    if node.translation:
        translation = np.eye(4)
        translation[:3, 3] = node.translation
        matrix = translation @ matrix
    return matrix


def _quaternion_matrix(q: list[float]) -> npt.NDArray[np.float64]:
    x, y, z, w = (float(v) for v in q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w), 0.0],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w), 0.0],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y), 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def _root_nodes(gltf: Any) -> list[int]:  # noqa: ANN401 - pygltflib is untyped
    if gltf.scenes:
        scene = gltf.scenes[gltf.scene or 0]
        return [int(i) for i in scene.nodes or []]
    children = {int(c) for node in gltf.nodes for c in node.children or []}
    return [i for i in range(len(gltf.nodes)) if i not in children]


def _mesh_instances(gltf: Any) -> list[tuple[int, npt.NDArray[np.float64]]]:  # noqa: ANN401
    """Depth-first traversal returning ``(mesh_index, world_matrix)`` for every mesh node."""
    found: list[tuple[int, npt.NDArray[np.float64]]] = []
    stack = [(index, np.eye(4)) for index in reversed(_root_nodes(gltf))]
    while stack:
        index, parent = stack.pop()
        node = gltf.nodes[index]
        world = parent @ _node_matrix(node)
        if node.mesh is not None:
            found.append((int(node.mesh), world))
        stack.extend((int(child), world) for child in reversed(node.children or []))
    return found


def _load_primitive(
    reader: _Reader,
    prim: Any,  # noqa: ANN401 - pygltflib is untyped
    matrix: npt.NDArray[np.float64],
) -> GlbMesh:
    mode = MODE_TRIANGLES if prim.mode is None else int(prim.mode)
    if mode != MODE_TRIANGLES:
        raise UnsupportedGlbError(f"primitive mode {mode} is not TRIANGLES (4)")
    if "KHR_draco_mesh_compression" in (prim.extensions or {}):
        raise UnsupportedGlbError("KHR_draco_mesh_compression primitives are not supported")
    position_index = getattr(prim.attributes, "POSITION", None)
    if position_index is None:
        raise UnsupportedGlbError("primitive has no POSITION attribute")
    local = reader.float_accessor(position_index)
    if local.shape[1] != _TYPE_COMPONENTS["VEC3"]:
        raise UnsupportedGlbError("POSITION accessor is not VEC3")
    homogeneous = np.concatenate([local.astype(np.float64), np.ones((len(local), 1))], axis=1)
    positions = (homogeneous @ matrix.T)[:, :3].astype(np.float32)
    uv_index = getattr(prim.attributes, "TEXCOORD_0", None)
    uvs = None if uv_index is None else reader.float_accessor(uv_index)[:, :2]
    indices = _triangle_indices(reader, prim, len(positions))
    material = None if prim.material is None else int(prim.material)
    return GlbMesh(positions=positions, uvs=uvs, indices=indices, material_index=material)


def _triangle_indices(
    reader: _Reader,
    prim: Any,  # noqa: ANN401 - pygltflib is untyped
    vertex_count: int,
) -> npt.NDArray[np.uint32]:
    if prim.indices is None:
        flat = np.arange(vertex_count, dtype=np.uint32)
    else:
        flat = reader.accessor(prim.indices).reshape(-1).astype(np.uint32)
    if flat.size % 3:
        raise UnsupportedGlbError(f"index count {flat.size} is not a multiple of 3")
    if flat.size and int(flat.max()) >= vertex_count:
        raise UnsupportedGlbError("primitive index out of range")
    return flat.reshape(-1, 3)


# ------------------------------------------------------------------------------------ materials
def _texture_ref(
    gltf: Any,  # noqa: ANN401 - pygltflib is untyped
    info: Any,  # noqa: ANN401 - pygltflib is untyped
) -> GlbTextureRef | None:
    if info is None:
        return None
    if "KHR_texture_transform" in (info.extensions or {}):
        raise UnsupportedGlbError("KHR_texture_transform is not supported")
    texture = gltf.textures[info.index]
    if texture.source is None:
        raise UnsupportedGlbError(f"texture {info.index} has no image source")
    return GlbTextureRef(
        texture_index=int(info.index),
        image_index=int(texture.source),
        tex_coord=int(info.texCoord or 0),
    )


def _load_material(
    gltf: Any,  # noqa: ANN401 - pygltflib is untyped
    material: Any,  # noqa: ANN401 - pygltflib is untyped
) -> GlbMaterial:
    pbr = material.pbrMetallicRoughness
    emissive_ext = (material.extensions or {}).get("KHR_materials_emissive_strength") or {}
    base_factor = (1.0, 1.0, 1.0, 1.0)
    metallic, roughness = 1.0, 1.0
    base_tex = mr_tex = None
    if pbr is not None:
        if pbr.baseColorFactor is not None:
            base_factor = _tuple4(pbr.baseColorFactor)
        metallic = 1.0 if pbr.metallicFactor is None else float(pbr.metallicFactor)
        roughness = 1.0 if pbr.roughnessFactor is None else float(pbr.roughnessFactor)
        base_tex = _texture_ref(gltf, pbr.baseColorTexture)
        mr_tex = _texture_ref(gltf, pbr.metallicRoughnessTexture)
    normal = material.normalTexture
    occlusion = material.occlusionTexture
    emissive_factor = material.emissiveFactor or [0.0, 0.0, 0.0]
    return GlbMaterial(
        name=material.name,
        base_color_factor=base_factor,
        metallic_factor=metallic,
        roughness_factor=roughness,
        emissive_factor=(
            float(emissive_factor[0]),
            float(emissive_factor[1]),
            float(emissive_factor[2]),
        ),
        normal_scale=1.0 if normal is None or normal.scale is None else float(normal.scale),
        occlusion_strength=(
            1.0 if occlusion is None or occlusion.strength is None else float(occlusion.strength)
        ),
        emissive_strength=float(emissive_ext.get("emissiveStrength", 1.0)),
        base_color_texture=base_tex,
        metallic_roughness_texture=mr_tex,
        normal_texture=_texture_ref(gltf, normal),
        occlusion_texture=_texture_ref(gltf, occlusion),
        emissive_texture=_texture_ref(gltf, material.emissiveTexture),
    )


def _tuple4(values: list[float]) -> tuple[float, float, float, float]:
    return (float(values[0]), float(values[1]), float(values[2]), float(values[3]))


def _load_image(
    reader: _Reader,
    image: Any,  # noqa: ANN401 - pygltflib is untyped
) -> GlbImage:
    if image.bufferView is not None:
        data = reader.buffer_view(image.bufferView)
    elif image.uri is not None and image.uri.startswith("data:"):
        data = _decode_data_uri(image.uri)
    elif image.uri is not None:
        data = _read_external(reader.base_dir, image.uri)
    else:
        raise UnsupportedGlbError("image has neither bufferView nor uri")
    mime = image.mimeType or sniff_image_mime(data)
    if mime not in ("image/png", "image/jpeg"):
        raise UnsupportedGlbError(f"unsupported image format {mime!r}; expected PNG or JPEG")
    return GlbImage(data=data, mime_type=mime, name=image.name)
