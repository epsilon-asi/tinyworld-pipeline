"""``prepare_model`` against ``FakeScene`` (plan 09 §2.10)."""

from __future__ import annotations

import hashlib
import json
import struct
import sys
import types
import zlib
from pathlib import Path
from typing import Any

import pytest

from fakes.fake_scene import (
    FakeScene,
    cube_object,
    cube_uv_atlas,
    cube_uv_palette,
    scene_no_armature,
    scene_palette_uv,
    scene_two_armatures,
    scene_two_meshes_shapekeys,
    scene_unskinned_extra,
    scene_with_rigged_mesh,
)
from tinyworld_pipeline.blender import protocol as orch
from tinyworld_pipeline.blender_jobs import protocol as bj
from tinyworld_pipeline.blender_jobs.errors import JobFailure
from tinyworld_pipeline.blender_jobs.fbx_settings import FBX_IMPORT_KWARGS, GLB_EXPORT_KWARGS
from tinyworld_pipeline.blender_jobs.jobs import JOBS, prepare_model
from tinyworld_pipeline.blender_jobs.jobs.prepare_model import (
    SUBSURF_MODIFIER,
    GeometryStrategy,
    SubdivideStrategy,
    image_dimensions,
    parse_params,
    run_with_scene,
    select_strategy,
)
from tinyworld_pipeline.blender_jobs.scene_api import SceneApi
from tinyworld_pipeline.blender_jobs.uv_heuristics import analyse


# --------------------------------------------------------------------------- helpers
def make_params(tmp_path: Path, **overrides: Any) -> bj.PrepareModelParams:
    geometry = {k[4:]: v for k, v in overrides.items() if k.startswith("geo_")}
    uv = {k[3:]: v for k, v in overrides.items() if k.startswith("uv_")}
    hints = overrides.get("hints", [])
    return bj.PrepareModelParams(
        source_fbx=str(tmp_path / "source" / "cube.fbx"),
        out_blend=str(tmp_path / "work" / "prepared.blend"),
        out_static_glb=str(tmp_path / "work" / "static.glb"),
        out_report=str(tmp_path / "work" / "prepare_report.json"),
        geometry=bj.GeometrySection(**geometry),
        uv=bj.UvSection(**uv),
        material_name="cube_rigged_mat",
        palette_hint_textures=[str(h) for h in hints],
    )


def run(
    scene: FakeScene, tmp_path: Path, **overrides: Any
) -> tuple[bj.PrepareModelData, list[str]]:
    return run_with_scene(make_params(tmp_path, **overrides), scene)


def failure(scene: FakeScene, tmp_path: Path, **overrides: Any) -> JobFailure:
    with pytest.raises(JobFailure) as info:
        run(scene, tmp_path, **overrides)
    return info.value


def png_bytes(width: int, height: int) -> bytes:
    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    raw = b"".join(b"\x00" + b"\x00" * (3 * width) for _ in range(height))
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


def jpeg_bytes(width: int, height: int) -> bytes:
    app0 = b"\xff\xe0" + struct.pack(">H", 16) + b"JFIF\x00" + b"\x00" * 9
    sof0 = b"\xff\xc0" + struct.pack(">HBHHB", 11, 8, height, width, 1) + b"\x01\x11\x00"
    return b"\xff\xd8" + app0 + b"\xff\xd0" + sof0 + b"\xff\xd9"


# --------------------------------------------------------------------------- happy path
def test_single_mesh_happy_path(tmp_path: Path) -> None:
    scene = scene_with_rigged_mesh()
    data, warnings = run(scene, tmp_path)

    assert warnings == []
    assert data.mesh == bj.MeshStats(
        object_names=["Cube"],
        vertices_before=8,
        faces_before=6,
        vertices_after=26,
        faces_after=24,
        triangles_after=48,
        shapekey_meshes=[],
        shapekeys_policy_applied="none",
        material_slots_before=1,
        material_slots_after=1,
    )
    assert data.uv.decision == "kept"
    assert data.uv.reason == "uvs usable"
    assert data.uv.uv_layers == ["UVMap"]
    assert data.uv.overlap_ratio == 0.0
    assert data.uv.island_count == 6
    assert data.uv.source_texture_max_px == 256
    assert data.rig.bone_count == 3
    assert [b.transform_path for b in data.rig.bones] == [
        "Armature/Root",
        "Armature/Root/Spine",
        "Armature/Root/Spine/Head",
    ]
    assert data.rig.actions == ["idle"]
    assert data.static_glb_triangles == 48
    assert data.static_glb_unique_positions == 26
    assert data.bbox_diagonal == pytest.approx(25.0)
    glb = Path(make_params(tmp_path).out_static_glb)
    assert data.static_glb_sha256 == hashlib.sha256(glb.read_bytes()).hexdigest()
    assert scene.objects["Cube"].materials == ["cube_rigged_mat"]
    assert scene.objects["Cube"].smooth is True
    assert scene.objects["Cube"].modifiers[0].kind == "ARMATURE"  # subsurf applied and gone


def test_happy_path_call_order_and_report(tmp_path: Path) -> None:
    scene = scene_with_rigged_mesh()
    data, _ = run(scene, tmp_path)

    order = scene.call_order()
    expected = [
        "import_fbx",
        "merge_material_slots",
        "keep_first_uv_layer",
        "add_subsurf",
        "move_modifier_to_top",
        "apply_modifier",
        "shade_smooth",
        "save_blend",
        "export_glb",
    ]
    assert [name for name in order if name in expected] == expected
    assert scene.method_calls("add_subsurf") == [
        ("add_subsurf", "Cube", SUBSURF_MODIFIER, 1, "catmull", "PRESERVE_BOUNDARIES", False)
    ]
    assert scene.imports[0][2] == FBX_IMPORT_KWARGS
    kind, objs, _, kwargs = scene.exports[0]
    assert (kind, objs) == ("glb", ["Cube"])
    assert kwargs == GLB_EXPORT_KWARGS
    # The report is the orchestrator-side contract, byte for byte.
    report = json.loads(Path(make_params(tmp_path).out_report).read_text())
    parsed = orch.PrepareModelData.model_validate(report)
    assert parsed.mesh.faces_after == 24
    assert report == bj.to_dict(data)


def test_multi_mesh_with_drop_policy(tmp_path: Path) -> None:
    scene = scene_two_meshes_shapekeys()
    data, warnings = run(scene, tmp_path, geo_shapekeys="drop")
    assert data.mesh.object_names == ["Body", "Face"]
    assert data.mesh.vertices_after == 52
    assert data.mesh.faces_after == 48
    assert data.mesh.triangles_after == 96
    assert data.mesh.shapekey_meshes == ["Face"]
    assert data.mesh.shapekeys_policy_applied == "dropped"
    assert data.mesh.material_slots_before == 2
    assert data.mesh.material_slots_after == 2
    assert ("clear_shapekeys", "Face") in scene.calls
    assert any("dropped" in w for w in warnings)
    assert data.rig.actions == ["idle"]  # Key| action filtered out
    assert data.static_glb_triangles == 96


def test_levels_zero_adds_no_modifier(tmp_path: Path) -> None:
    scene = scene_with_rigged_mesh()
    data, _ = run(scene, tmp_path, geo_subdivision_levels=0)
    assert data.mesh.faces_after == 6
    assert data.mesh.vertices_after == 8
    assert data.mesh.triangles_after == 12
    assert scene.method_calls("add_subsurf") == []
    assert scene.method_calls("apply_modifier") == []


def test_levels_two_and_simple_type(tmp_path: Path) -> None:
    scene = scene_with_rigged_mesh()
    data, _ = run(scene, tmp_path, geo_subdivision_levels=2, geo_subdivision_type="simple")
    assert (data.mesh.vertices_after, data.mesh.faces_after) == (98, 96)
    assert scene.method_calls("add_subsurf")[0][4] == "simple"


def test_shade_smooth_and_merge_can_be_disabled(tmp_path: Path) -> None:
    scene = scene_with_rigged_mesh(materials=("a", "b"))
    data, _ = run(scene, tmp_path, geo_shade_smooth=False, geo_merge_material_slots=False)
    assert scene.method_calls("shade_smooth") == []
    assert scene.method_calls("merge_material_slots") == []
    assert scene.objects["Cube"].materials == ["a", "b"]
    assert data.mesh.material_slots_after == 2


# --------------------------------------------------------------------------- UV decisions
def test_palette_texture_is_unwrapped_before_subdivision(tmp_path: Path) -> None:
    scene = scene_palette_uv()
    data, _ = run(scene, tmp_path)
    assert data.uv.decision == "unwrapped"
    assert "palette" in data.uv.reason
    assert data.uv.overlap_ratio == 1.0
    assert data.uv.source_texture_max_px == 16
    order = scene.call_order()
    assert order.index("smart_unwrap") < order.index("add_subsurf")
    assert scene.method_calls("smart_unwrap") == [("smart_unwrap", ("Cube",), 66.0, 0.02, 0.01)]
    faces = scene.objects["Cube"].uv_layers["UVMap"]
    assert len(faces) == 24  # subdivided after the unwrap
    assert analyse(faces, []).overlap_ratio == 0.0


def test_overlapping_uvs_with_large_texture_are_unwrapped(tmp_path: Path) -> None:
    scene = scene_with_rigged_mesh(image_sizes=((1024, 1024),))
    scene.objects["Cube"].uv_layers["UVMap"] = cube_uv_palette()
    data, _ = run(scene, tmp_path)
    assert data.uv.decision == "unwrapped"
    assert data.uv.reason.startswith("overlap 1.00")


def test_missing_uvs_are_created(tmp_path: Path) -> None:
    scene = scene_with_rigged_mesh(uv=False)
    data, _ = run(scene, tmp_path)
    assert data.uv.decision == "created"
    assert data.uv.reason == "no uv layer"
    assert data.uv.uv_layers == ["UVMap"]
    assert scene.method_calls("smart_unwrap")


def test_keep_mode(tmp_path: Path) -> None:
    data, _ = run(scene_palette_uv(), tmp_path, uv_mode="keep")
    assert data.uv.decision == "kept"
    assert data.uv.reason == "mode=keep"
    err = failure(scene_with_rigged_mesh(uv=False), tmp_path, uv_mode="keep")
    assert err.code == bj.E_BJ_NO_UV_KEEP


def test_unwrap_mode(tmp_path: Path) -> None:
    assert run(scene_with_rigged_mesh(), tmp_path, uv_mode="unwrap")[0].uv.decision == "unwrapped"
    assert (
        run(scene_with_rigged_mesh(uv=False), tmp_path, uv_mode="unwrap")[0].uv.decision
        == "created"
    )


def test_extra_uv_layers_are_removed_only_when_configured(tmp_path: Path) -> None:
    extra = {"Lightmap": cube_uv_atlas()}
    scene = scene_with_rigged_mesh(extra_uv_layers=extra)
    data, _ = run(scene, tmp_path)
    assert data.uv.uv_layers == ["UVMap"]
    scene = scene_with_rigged_mesh(extra_uv_layers=extra)
    data, _ = run(scene, tmp_path, geo_keep_only_one_uv_layer=False)
    assert data.uv.uv_layers == ["UVMap", "Lightmap"]


def test_mesh_without_uv_among_uv_meshes_counts_as_no_uv(tmp_path: Path) -> None:
    scene = scene_two_meshes_shapekeys()
    scene.objects["Face"].uv_layers = {}
    scene.objects["Face"].shapekeys = False
    data, _ = run(scene, tmp_path)
    assert data.uv.decision == "created"
    assert scene.method_calls("smart_unwrap")[0][1] == ("Body", "Face")


def test_palette_hint_textures_drive_the_decision(tmp_path: Path) -> None:
    hint = tmp_path / "source" / "textures" / "palette.png"
    hint.parent.mkdir(parents=True)
    hint.write_bytes(png_bytes(16, 4))
    scene = scene_with_rigged_mesh(image_sizes=())
    data, _ = run(scene, tmp_path, hints=[hint, tmp_path / "missing.png"])
    assert data.uv.decision == "unwrapped"
    assert data.uv.reason == "palette texture 16px"


def test_image_dimensions_parsers(tmp_path: Path) -> None:
    png, jpg, junk = tmp_path / "a.png", tmp_path / "b.jpg", tmp_path / "c.bin"
    png.write_bytes(png_bytes(640, 480))
    jpg.write_bytes(jpeg_bytes(320, 200))
    junk.write_bytes(b"not an image at all")
    assert image_dimensions(str(png)) == (640, 480)
    assert image_dimensions(str(jpg)) == (320, 200)
    assert image_dimensions(str(junk)) is None
    assert image_dimensions(str(tmp_path / "missing.png")) is None
    assert image_dimensions(str(tmp_path)) is None  # a directory
    truncated = tmp_path / "d.jpg"
    truncated.write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF")
    assert image_dimensions(str(truncated)) is None


# --------------------------------------------------------------------------- shape keys
def test_shapekeys_fail_policy(tmp_path: Path) -> None:
    err = failure(scene_two_meshes_shapekeys(), tmp_path)
    assert err.code == bj.E_BJ_SHAPEKEYS
    assert err.detail["meshes"] == ["Face"]
    assert "drop" in err.detail["hint"]


def test_shapekeys_skip_policy_leaves_mesh_unsubdivided(tmp_path: Path) -> None:
    scene = scene_two_meshes_shapekeys()
    data, warnings = run(scene, tmp_path, geo_shapekeys="skip")
    assert data.mesh.shapekeys_policy_applied == "skipped"
    assert scene.objects["Face"].faces == 6
    assert scene.objects["Body"].faces == 24
    assert scene.objects["Face"].shapekeys is True
    assert data.mesh.faces_after == 30
    assert any("skip" in w for w in warnings)
    assert scene.method_calls("clear_shapekeys") == []


# --------------------------------------------------------------------------- unskinned meshes
def test_unskinned_meshes_are_exported_but_never_subdivided(tmp_path: Path) -> None:
    scene = scene_unskinned_extra()
    data, warnings = run(scene, tmp_path)
    assert any("Prop" in w for w in warnings)
    assert data.mesh.object_names == ["Cube"]
    assert scene.objects["Prop"].faces == 6
    assert scene.exports[0][1] == ["Cube", "Prop"]
    assert data.static_glb_triangles == 48 + 12
    assert data.static_glb_unique_positions == 26  # Prop's synthetic positions coincide with Cube's


def test_unskinned_meshes_excluded_when_configured(tmp_path: Path) -> None:
    scene = scene_unskinned_extra()
    data, _ = run(scene, tmp_path, geo_include_unskinned=False)
    assert scene.exports[0][1] == ["Cube"]
    assert data.static_glb_triangles == 48


# --------------------------------------------------------------------------- E_BJ_* paths
def test_no_armature(tmp_path: Path) -> None:
    err = failure(scene_no_armature(), tmp_path)
    assert err.code == bj.E_BJ_NO_ARMATURE
    assert err.detail["meshes"] == ["Cube"]


def test_multi_armature(tmp_path: Path) -> None:
    err = failure(scene_two_armatures(), tmp_path)
    assert err.code == bj.E_BJ_MULTI_ARMATURE
    assert err.detail["armatures"] == ["Armature", "Armature.001"]


def test_no_skinned_mesh(tmp_path: Path) -> None:
    scene = scene_with_rigged_mesh()
    scene.objects["Cube"].parent_armature = None
    err = failure(scene, tmp_path)
    assert err.code == bj.E_BJ_NO_SKINNED_MESH


def test_import_failed(tmp_path: Path) -> None:
    scene = scene_with_rigged_mesh()
    scene.import_error = RuntimeError("Error: FBX version too old")
    err = failure(scene, tmp_path)
    assert err.code == bj.E_BJ_IMPORT_FAILED
    assert "too old" in err.message


def test_export_failed_when_exporter_raises_or_writes_nothing(tmp_path: Path) -> None:
    scene = scene_with_rigged_mesh()
    scene.export_error = RuntimeError("boom")
    assert failure(scene, tmp_path).code == bj.E_BJ_EXPORT_FAILED
    scene = scene_with_rigged_mesh()
    scene.export_writes_file = False
    assert failure(scene, tmp_path).code == bj.E_BJ_EXPORT_FAILED


def test_unwrap_failed(tmp_path: Path) -> None:
    scene = scene_with_rigged_mesh(uv=False)
    scene.unwrap_zero_area = True
    assert failure(scene, tmp_path).code == bj.E_BJ_UNWRAP_FAILED


def test_face_limit_is_checked_before_applying(tmp_path: Path) -> None:
    scene = scene_with_rigged_mesh()
    err = failure(scene, tmp_path, geo_subdivision_levels=3, geo_max_triangles=100)
    assert err.code == bj.E_BJ_FACE_LIMIT
    assert err.detail["projected_triangles"] == 12 * 64
    assert "subdivision_levels" in err.detail["hint"]
    assert scene.method_calls("add_subsurf") == []
    assert scene.method_calls("apply_modifier") == []
    assert scene.objects["Cube"].faces == 6


def test_face_limit_counts_only_subdividable_meshes(tmp_path: Path) -> None:
    scene = scene_two_meshes_shapekeys()  # Face skipped → only Body's 12 tris x 4
    data, _ = run(scene, tmp_path, geo_shapekeys="skip", geo_max_triangles=60)
    assert data.mesh.faces_after == 30


class MutatingStrategy:
    """Deliberately violates the rig invariant (plan 16 WP-04 acceptance)."""

    def __init__(self, scene: FakeScene) -> None:
        self.scene = scene

    def apply(self, scene: SceneApi, meshes: list[str], params: bj.PrepareModelParams) -> list[str]:
        del scene, meshes, params
        self.scene.mutate_rig()
        return ["mutated"]


def test_rig_mutation_guard(tmp_path: Path) -> None:
    scene = scene_with_rigged_mesh()
    strategy: GeometryStrategy = MutatingStrategy(scene)
    with pytest.raises(JobFailure) as info:
        run_with_scene(make_params(tmp_path), scene, strategy=strategy)
    assert info.value.code == bj.E_BJ_RIG_MUTATED
    assert info.value.detail["before"] != info.value.detail["after"]


def test_strategy_warnings_are_propagated(tmp_path: Path) -> None:
    class Noisy(SubdivideStrategy):
        def apply(
            self, scene: SceneApi, meshes: list[str], params: bj.PrepareModelParams
        ) -> list[str]:
            super().apply(scene, meshes, params)
            return ["strategy says hi"]

    _, warnings = run_with_scene(make_params(tmp_path), scene_with_rigged_mesh(), strategy=Noisy())
    assert warnings == ["strategy says hi"]


def test_unknown_strategy_and_bad_params() -> None:
    with pytest.raises(JobFailure) as info:
        select_strategy("remesh_transfer")
    assert info.value.code == bj.E_BJ_BAD_PARAMS
    assert isinstance(select_strategy("subdivide"), SubdivideStrategy)
    with pytest.raises(JobFailure) as info:
        parse_params({"bogus": 1})
    assert info.value.code == bj.E_BJ_BAD_PARAMS


def test_subdivide_strategy_noop_without_meshes(tmp_path: Path) -> None:
    assert SubdivideStrategy().apply(scene_with_rigged_mesh(), [], make_params(tmp_path)) == []


# --------------------------------------------------------------------------- registry wrapper
def test_run_builds_scene_lazily_and_returns_json(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    scene = scene_with_rigged_mesh()
    module = types.ModuleType("tinyworld_pipeline.blender_jobs.bpy_scene")
    module.BpyScene = lambda: scene  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "tinyworld_pipeline.blender_jobs.bpy_scene", module)
    raw = orch.PrepareModelParams.model_validate(bj.to_dict(make_params(tmp_path))).model_dump(
        mode="json"
    )

    data, warnings = prepare_model.run(raw)

    assert warnings == []
    assert orch.PrepareModelData.model_validate(data).mesh.faces_after == 24
    assert scene.saved_blends == [raw["out_blend"]]


def test_registry_exposes_prepare_model() -> None:
    assert JOBS["prepare_model"] is prepare_model.run


def test_cube_object_builder_defaults() -> None:
    cube = cube_object()
    assert (cube.verts, cube.faces, cube.tris) == (8, 6, 12)
    assert cube.parent_armature == "Armature"
