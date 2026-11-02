from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from fakes.glb_builder import (
    ImageSpec,
    MaterialSpec,
    cube,
    encode_image,
    quad,
    solid_image,
    textured_set,
    write_glb,
)
from tinyworld_pipeline.domain.errors import MultiMaterialResultError, UnsupportedGlbError
from tinyworld_pipeline.domain.textures import RawTextureSet, TextureFactors
from tinyworld_pipeline.gltf.extract import extract_textures
from tinyworld_pipeline.gltf.loader import (
    GlbDocument,
    GlbImage,
    GlbMaterial,
    GlbMesh,
    GlbTextureRef,
    load_glb,
)

GlbWriter = Callable[[bytes], Path]


@pytest.fixture
def glb(tmp_path: Path) -> GlbWriter:
    counter = iter(range(1000))

    def _write(data: bytes) -> Path:
        return write_glb(tmp_path / f"m{next(counter)}.glb", data)

    return _write


def _pixel(path: Path) -> list[int]:
    with Image.open(path) as img:
        return list(np.asarray(img.convert("RGBA"))[0, 0])


def test_extract_all_maps_verbatim_png(glb: GlbWriter, tmp_path: Path) -> None:
    images, material = textured_set()
    raw = extract_textures(
        load_glb(glb(cube(images=images, materials=[material]))), tmp_path / "raw"
    )
    assert isinstance(raw, RawTextureSet)
    assert raw.source == "extract"
    assert raw.basecolor == tmp_path / "raw" / "basecolor.png"
    assert raw.metallic_roughness == tmp_path / "raw" / "metallic_roughness.png"
    assert raw.normal == tmp_path / "raw" / "normal.png"
    assert raw.occlusion == tmp_path / "raw" / "occlusion.png"
    assert raw.emissive == tmp_path / "raw" / "emissive.png"
    assert raw.basecolor.read_bytes() == encode_image(images[0])  # PNG bytes copied as-is
    assert _pixel(raw.metallic_roughness) == [0, 128, 64, 255]
    assert _pixel(raw.normal) == [128, 128, 255, 255]
    assert _pixel(raw.occlusion) == [180, 180, 180, 255]
    assert _pixel(raw.emissive) == [10, 20, 30, 255]
    assert raw.factors == TextureFactors()


def test_extract_subset_of_maps(glb: GlbWriter, tmp_path: Path) -> None:
    images = [solid_image((1, 2, 3, 255)), solid_image((128, 128, 255, 255))]
    material = MaterialSpec(base_color_image=0, normal_image=1)
    raw = extract_textures(
        load_glb(glb(quad(images=images, materials=[material]))), tmp_path / "raw"
    )
    assert raw.basecolor is not None
    assert raw.normal is not None
    assert raw.metallic_roughness is None
    assert raw.occlusion is None
    assert raw.emissive is None
    assert sorted(p.name for p in (tmp_path / "raw").iterdir()) == ["basecolor.png", "normal.png"]


def test_extract_without_any_material(glb: GlbWriter, tmp_path: Path) -> None:
    raw = extract_textures(load_glb(glb(quad())), tmp_path / "raw")
    assert raw == RawTextureSet(None, None, None, None, None, TextureFactors(), "extract")
    assert (tmp_path / "raw").is_dir()


def test_extract_captures_factors(glb: GlbWriter, tmp_path: Path) -> None:
    images, base = textured_set()
    material = MaterialSpec(
        **{
            **base.__dict__,
            "base_color_factor": (0.5, 0.25, 1.0, 0.75),
            "metallic_factor": 0.2,
            "roughness_factor": 0.3,
            "emissive_factor": (1.0, 0.5, 0.0),
            "normal_scale": 0.8,
            "occlusion_strength": 0.6,
            "emissive_strength": 4.0,
        }
    )
    raw = extract_textures(
        load_glb(glb(quad(images=images, materials=[material]))), tmp_path / "raw"
    )
    assert raw.factors == TextureFactors(
        base_color=(0.5, 0.25, 1.0, 0.75),
        metallic=0.2,
        roughness=0.3,
        emissive=(1.0, 0.5, 0.0),
        normal_scale=0.8,
        occlusion_strength=0.6,
        emissive_strength=4.0,
    )


def test_extract_reencodes_jpeg_as_png(glb: GlbWriter, tmp_path: Path) -> None:
    pixels = solid_image((200, 100, 50, 255)).pixels
    images = [ImageSpec(pixels=pixels, fmt="jpeg", embed="datauri")]
    doc = load_glb(glb(quad(images=images, materials=[MaterialSpec(base_color_image=0)])))
    raw = extract_textures(doc, tmp_path / "raw")
    assert raw.basecolor is not None
    assert raw.basecolor.read_bytes()[:4] == b"\x89PNG"
    r, g, b, a = _pixel(raw.basecolor)
    assert (abs(r - 200), abs(g - 100), abs(b - 50)) <= (3, 3, 3)
    assert a == 255


def test_same_material_on_several_primitives_is_fine(glb: GlbWriter, tmp_path: Path) -> None:
    images, material = textured_set()
    doc = load_glb(glb(cube(primitives=3, images=images, materials=[material])))
    raw = extract_textures(doc, tmp_path / "raw")
    assert raw.basecolor is not None


def test_two_materials_with_identical_content_are_one_texture_set(
    glb: GlbWriter, tmp_path: Path
) -> None:
    images, material = textured_set()
    twins = [
        MaterialSpec(**{**material.__dict__, "name": "a"}),
        MaterialSpec(**{**material.__dict__, "name": "b"}),
    ]
    doc = load_glb(glb(cube(primitives=2, images=images, materials=twins)))
    raw = extract_textures(doc, tmp_path / "raw")
    assert raw.basecolor is not None


def test_distinct_materials_raise_multi_material(glb: GlbWriter, tmp_path: Path) -> None:
    images, material = textured_set()
    other = MaterialSpec(name="other", base_color_image=4)
    doc = load_glb(glb(cube(primitives=2, images=images, materials=[material, other])))
    with pytest.raises(MultiMaterialResultError) as info:
        extract_textures(doc, tmp_path / "raw")
    assert info.value.code == "E_MULTI_MATERIAL"
    assert info.value.detail["materials"] == ["mat", "other"]
    assert not (tmp_path / "raw").exists()  # nothing written before the check


def test_factor_only_difference_is_still_multi_material(glb: GlbWriter, tmp_path: Path) -> None:
    images, material = textured_set()
    variant = MaterialSpec(**{**material.__dict__, "metallic_factor": 0.5})
    doc = load_glb(glb(cube(primitives=2, images=images, materials=[material, variant])))
    with pytest.raises(MultiMaterialResultError):
        extract_textures(doc, tmp_path / "raw")


def test_primitive_without_material_next_to_textured_one_is_multi_material(
    glb: GlbWriter, tmp_path: Path
) -> None:
    images, material = textured_set()
    doc = load_glb(
        glb(cube(primitives=2, images=images, materials=[material], primitive_materials=[0, None]))
    )
    with pytest.raises(MultiMaterialResultError) as info:
        extract_textures(doc, tmp_path / "raw")
    assert info.value.detail["materials"] == ["mat"]


def test_texture_transform_is_rejected_at_load(glb: GlbWriter) -> None:
    images = [solid_image((1, 2, 3, 4))]
    material = MaterialSpec(base_color_image=0, texture_transform=True)
    with pytest.raises(UnsupportedGlbError, match="KHR_texture_transform"):
        load_glb(glb(quad(images=images, materials=[material])))


def test_undecodable_image_raises(tmp_path: Path) -> None:
    mesh = GlbMesh(
        positions=np.zeros((3, 3), dtype=np.float32),
        uvs=np.zeros((3, 2), dtype=np.float32),
        indices=np.array([[0, 1, 2]], dtype=np.uint32),
        material_index=0,
    )
    material = GlbMaterial(name="m", base_color_texture=GlbTextureRef(0, 0))
    doc = GlbDocument(
        meshes=[mesh],
        materials=[material],
        images=[GlbImage(data=b"\xff\xd8\xffgarbage", mime_type="image/jpeg")],
    )
    with pytest.raises(UnsupportedGlbError, match="basecolor image"):
        extract_textures(doc, tmp_path / "raw")
