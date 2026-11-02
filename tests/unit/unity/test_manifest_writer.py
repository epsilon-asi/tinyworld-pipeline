from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pytest

from tinyworld_pipeline.domain.errors import HandoffError
from tinyworld_pipeline.domain.library_config import UnitySection
from tinyworld_pipeline.domain.manifest import Manifest, ManifestProvenance
from tinyworld_pipeline.unity.manifest_writer import (
    MAP_NAMES,
    UrpMaterialInputs,
    build_manifest,
    first_root_bone,
    manifest_relpaths,
    write_manifest,
)

ROOT = Path(__file__).resolve().parents[3]
GOLDEN = ROOT / "tests" / "fixtures" / "manifest_golden.json"

ALL_MAPS = {
    "basecolor": "textures/cube_rigged_basecolor.png",
    "normal": "textures/cube_rigged_normal.png",
    "metallic_smoothness": "textures/cube_rigged_metallic_smoothness.png",
    "occlusion": "textures/cube_rigged_occlusion.png",
    "emission": "textures/cube_rigged_emission.png",
}
GOLDEN_MAPS = {k: ALL_MAPS[k] for k in ("basecolor", "normal", "metallic_smoothness")}


def provenance() -> ManifestProvenance:
    return ManifestProvenance(
        source_fbx_sha256="3f" * 32,
        source_license="CC0 — synthetic fixture",
        subdivision_levels=1,
        subdivision_type="catmull_clark",
        uv_decision="kept",
        texture_provider="tripo",
        texture_task_id="task_cube_rigged_0001",
        texture_route="extract",
        texture_seed=4242,
        prompt_sha256="a7" * 32,
        credits_consumed=20.0,
        rig_fidelity="pass",
    )


def build(
    *,
    texture_files: dict[str, str] | None = None,
    material: UrpMaterialInputs | None = None,
    unity: UnitySection | None = None,
    source_asset_path: str | None = "Assets/Vendor/Cubes/cube_rigged.fbx",
    animations: list[tuple[str, str]] | None = None,
    fbx_name: str = "cube_rigged.fbx",
) -> Manifest:
    return build_manifest(
        model_id="cube_rigged",
        display_name="Rigged Cube",
        pipeline_version="0.1.0",
        generated_at="2026-11-02T12:00:00Z",
        fbx_name=fbx_name,
        root_bone="Root",
        unity=unity or UnitySection(),
        source_asset_path=source_asset_path,
        material=material
        or UrpMaterialInputs(
            bump_scale=1.0,
            occlusion_strength=1.0,
            emission_color=(0.0, 0.0, 0.0),
            metallic_fallback=0.0,
            smoothness_fallback=1.0,
        ),
        texture_files=GOLDEN_MAPS if texture_files is None else texture_files,
        animations=[("animations/cube_rigged@walk.fbx", "walk")]
        if animations is None
        else animations,
        provenance=provenance(),
    )


def names(items: Sequence[object], attr: str = "name") -> list[str]:
    return [str(getattr(item, attr)) for item in items]


def floats(manifest: Manifest) -> dict[str, float]:
    return {f.name: f.value for f in manifest.materials[0].floats}


# ------------------------------------------------------------------------------ golden
def test_golden_manifest_matches_builder(tmp_path: Path) -> None:
    out = write_manifest(build(), tmp_path / "manifest.json")
    assert out.read_text(encoding="utf-8") == GOLDEN.read_text(encoding="utf-8")
    assert Manifest.model_validate_json(GOLDEN.read_text(encoding="utf-8")) == build()


def test_golden_keeps_schema_field_order() -> None:
    doc = json.loads(GOLDEN.read_text(encoding="utf-8"))
    assert list(doc) == list(Manifest.model_fields)
    assert list(doc["materials"][0]) == [
        "name",
        "shader",
        "renderers",
        "textures",
        "floats",
        "colors",
        "keywords",
    ]


# ------------------------------------------------------------------------------ textures / keywords
def test_all_maps_produce_every_slot_keyword_and_float() -> None:
    material = UrpMaterialInputs(
        bump_scale=0.8,
        occlusion_strength=0.6,
        emission_color=(2.0, 1.0, 0.5),
        metallic_fallback=0.3,
        smoothness_fallback=0.4,
    )
    manifest = build(texture_files=ALL_MAPS, material=material)
    mat = manifest.materials[0]
    assert mat.name == "cube_rigged_mat"
    assert mat.shader == "Universal Render Pipeline/Lit"
    assert mat.renderers == []
    assert names(mat.textures, "slot") == [
        "_BaseMap",
        "_BumpMap",
        "_MetallicGlossMap",
        "_OcclusionMap",
        "_EmissionMap",
    ]
    assert [(t.srgb, t.kind) for t in mat.textures] == [
        (True, "default"),
        (False, "normal"),
        (False, "default"),
        (False, "default"),
        (True, "default"),
    ]
    assert mat.keywords == ["_NORMALMAP", "_METALLICSPECGLOSSMAP", "_OCCLUSIONMAP", "_EMISSION"]
    assert floats(manifest) == {
        "_WorkflowMode": 1.0,
        "_Surface": 0.0,
        "_Smoothness": 1.0,
        "_SmoothnessTextureChannel": 0.0,
        "_BumpScale": 0.8,
        "_OcclusionStrength": 0.6,
    }
    assert "_Metallic" not in floats(manifest)
    colors = {c.name: (c.r, c.g, c.b, c.a) for c in mat.colors}
    assert colors == {
        "_BaseColor": (1.0, 1.0, 1.0, 1.0),
        "_EmissionColor": (2.0, 1.0, 0.5, 1.0),
    }


def test_basecolor_only_uses_fallbacks_and_no_keywords() -> None:
    material = UrpMaterialInputs(metallic_fallback=0.25, smoothness_fallback=0.5)
    manifest = build(texture_files={"basecolor": ALL_MAPS["basecolor"]}, material=material)
    mat = manifest.materials[0]
    assert names(mat.textures, "slot") == ["_BaseMap"]
    assert mat.keywords == []
    assert floats(manifest) == {
        "_WorkflowMode": 1.0,
        "_Surface": 0.0,
        "_Metallic": 0.25,
        "_Smoothness": 0.5,
    }
    assert names(mat.colors) == ["_BaseColor"]


def test_no_textures_at_all_is_allowed() -> None:
    manifest = build(texture_files={})
    mat = manifest.materials[0]
    assert mat.textures == []
    assert mat.keywords == []
    assert set(floats(manifest)) == {"_WorkflowMode", "_Surface", "_Metallic", "_Smoothness"}


def test_subset_order_follows_slot_table_not_input_order() -> None:
    files = {"emission": ALL_MAPS["emission"], "normal": ALL_MAPS["normal"]}
    manifest = build(texture_files=files)
    mat = manifest.materials[0]
    assert names(mat.textures, "slot") == ["_BumpMap", "_EmissionMap"]
    assert mat.keywords == ["_NORMALMAP", "_EMISSION"]
    assert "_BumpScale" in floats(manifest)
    assert "_OcclusionStrength" not in floats(manifest)
    assert names(mat.colors) == ["_BaseColor", "_EmissionColor"]


def test_unknown_map_name_is_rejected() -> None:
    with pytest.raises(HandoffError, match="unknown texture map"):
        build(texture_files={"roughness": "textures/x.png"})
    assert MAP_NAMES == ("basecolor", "normal", "metallic_smoothness", "occlusion", "emission")


@pytest.mark.parametrize(
    "bad",
    ["/abs/textures/a.png", "../escape.png", "textures\\win.png", "", "textures/../a.png"],
)
def test_non_relative_paths_are_rejected(bad: str) -> None:
    with pytest.raises(HandoffError, match="relative forward-slash"):
        build(texture_files={"basecolor": bad})
    with pytest.raises(HandoffError, match="relative forward-slash"):
        build(animations=[(bad, "walk")])
    with pytest.raises(HandoffError, match="relative forward-slash"):
        build(fbx_name=bad)


# ------------------------------------------------------------------------------ rig
def test_rig_block_from_unity_section_and_source_asset_path() -> None:
    rig = build().rig
    assert rig.animation_type == "Generic"
    assert rig.avatar_mode == "copy_from_source"
    assert rig.source_avatar_asset_path == "Assets/Vendor/Cubes/cube_rigged.fbx"
    assert rig.root_bone == "Root"


@pytest.mark.parametrize("source", [None, ""])
def test_missing_source_asset_path_becomes_empty_string_not_root_join(source: str | None) -> None:
    unity = UnitySection(avatar_mode="create_from_model", source_asset_root="Assets/Vendor")
    rig = build(unity=unity, source_asset_path=source).rig
    assert rig.source_avatar_asset_path == ""
    assert rig.avatar_mode == "create_from_model"


def test_shader_comes_from_unity_section() -> None:
    unity = UnitySection(shader="Universal Render Pipeline/Simple Lit")
    assert build(unity=unity).materials[0].shader == "Universal Render Pipeline/Simple Lit"


@dataclass(frozen=True)
class FakeBone:
    name: str
    parent: str | None


def test_first_root_bone_picks_first_parentless_bone_in_order() -> None:
    bones = [FakeBone("Spine", "Root"), FakeBone("Root", None), FakeBone("Root2", None)]
    assert first_root_bone(bones) == "Root"


def test_first_root_bone_requires_a_root() -> None:
    with pytest.raises(HandoffError, match="no root bone"):
        first_root_bone([FakeBone("A", "B"), FakeBone("B", "A")])


# ------------------------------------------------------------------------------ scalars adapter
@dataclass(frozen=True)
class FakeUrpSet:
    bump_scale: float
    occlusion_strength: float
    emission_color: tuple[float, float, float]
    metallic_fallback: float
    smoothness_fallback: float


def test_material_inputs_from_texture_set_is_structural() -> None:
    fake = FakeUrpSet(0.5, 0.75, (1.0, 2.0, 3.0), 0.0, 0.5)
    assert UrpMaterialInputs.from_texture_set(fake) == UrpMaterialInputs(
        bump_scale=0.5,
        occlusion_strength=0.75,
        emission_color=(1.0, 2.0, 3.0),
        metallic_fallback=0.0,
        smoothness_fallback=0.5,
    )


# ------------------------------------------------------------------------------ relpaths
def test_animations_and_relpaths_cover_every_referenced_file() -> None:
    manifest = build(
        texture_files=ALL_MAPS,
        animations=[
            ("animations/cube_rigged@walk.fbx", "walk"),
            ("animations/cube_rigged@idle.fbx", "idle"),
        ],
    )
    assert [(a.file, a.clip_name) for a in manifest.animations] == [
        ("animations/cube_rigged@walk.fbx", "walk"),
        ("animations/cube_rigged@idle.fbx", "idle"),
    ]
    assert manifest_relpaths(manifest) == [
        "cube_rigged.fbx",
        *ALL_MAPS.values(),
        "animations/cube_rigged@walk.fbx",
        "animations/cube_rigged@idle.fbx",
    ]


def test_relpaths_deduplicate() -> None:
    manifest = build(animations=[("cube_rigged.fbx", "embedded")])
    paths = manifest_relpaths(manifest)
    assert paths.count("cube_rigged.fbx") == 1


# ------------------------------------------------------------------------------ write_manifest
def test_write_manifest_is_atomic_and_json_utility_clean(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "manifest.json"
    write_manifest(build(), target)
    assert target.exists()
    assert not (tmp_path / "nested" / "manifest.json.tmp").exists()
    text = target.read_text(encoding="utf-8")
    assert text.endswith("}\n")
    assert "—" in text  # ensure_ascii=False keeps the em dash in the licence
    assert json.loads(text) == build().model_dump(mode="json")


def test_write_manifest_rejects_json_utility_violations(tmp_path: Path) -> None:
    manifest = build()
    # Bypass validation to simulate a model drift that re-introduces a null.
    broken = manifest.model_copy(update={"display_name": None})
    with pytest.raises(HandoffError, match="JsonUtility") as info:
        write_manifest(broken, tmp_path / "manifest.json")
    assert "$.display_name" in str(info.value)
    assert not (tmp_path / "manifest.json").exists()
