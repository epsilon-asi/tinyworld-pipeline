"""The fake must be trustworthy: subdivision math, atlas, exports, hooks, BpyScene parity."""

from __future__ import annotations

import importlib
import inspect
import sys
import types
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from fakes.fake_scene import (
    FAKE_BLEND_BYTES,
    FAKE_GLB_HEADER,
    FakeModifier,
    FakeObject,
    FakeScene,
    cube_object,
    cube_uv_atlas,
    principled_inputs,
    scene_for_bake,
    scene_no_armature,
    scene_palette_uv,
    scene_two_armatures,
    scene_two_meshes_shapekeys,
    scene_unskinned_extra,
    scene_with_rigged_mesh,
    source_object,
    three_bone_chain,
)
from tinyworld_pipeline.blender_jobs.rig_fingerprint import fingerprint_sha256
from tinyworld_pipeline.blender_jobs.scene_api import SceneApi
from tinyworld_pipeline.blender_jobs.uv_heuristics import analyse


def test_builders_shape() -> None:
    assert scene_with_rigged_mesh().armatures() == ["Armature"]
    assert scene_with_rigged_mesh().skinned_meshes("Armature") == ["Cube"]
    assert scene_palette_uv().image_sizes("Cube") == [(16, 4)]
    assert scene_two_meshes_shapekeys().has_shapekeys("Face") is True
    assert scene_no_armature().armatures() == []
    assert scene_two_armatures().armatures() == ["Armature", "Armature.001"]
    assert scene_unskinned_extra().meshes() == ["Cube", "Prop"]
    assert scene_unskinned_extra().skinned_meshes("Armature") == ["Cube"]
    assert scene_with_rigged_mesh().blender_version() == (4, 5, 12)
    assert scene_with_rigged_mesh().armature_data_name("Armature") == "Armature"


def test_actions_filter_non_object_actions_and_strip_prefixes() -> None:
    scene = scene_two_meshes_shapekeys()
    assert scene.actions() == ["idle"]
    scene.raw_actions.append(("Armature|Armature|walk", "OBJECT"))
    assert scene.actions() == ["idle", "walk"]


def test_clean_action_names_renames_object_actions_without_collisions() -> None:
    scene = scene_two_meshes_shapekeys()
    scene.raw_actions.append(("Rig|Rig|idle", "OBJECT"))  # would collide with "idle"
    scene.clean_action_names()
    assert scene.raw_actions == [
        ("idle", "OBJECT"),
        ("Key|Armature|idle", "KEY"),  # non-object actions untouched
        ("Rig|Rig|idle", "OBJECT"),  # clean name taken: left alone
    ]
    assert scene.actions() == ["idle", "idle"]
    assert scene.method_calls("clean_action_names") == [("clean_action_names",)]


def test_reset_pose_and_fbx_payload_hooks() -> None:
    scene = FakeScene()
    scene.fbx_payload = {"Armature": scene_with_rigged_mesh().objects["Armature"]}
    scene.fbx_actions = [("Armature|Armature|walk", "OBJECT")]
    scene.import_fbx("clip.fbx")
    assert scene.armatures() == ["Armature"]
    assert scene.actions() == ["walk"]
    scene.reset_pose("Armature")
    assert scene.pose_resets == ["Armature"]
    scene.objects["Cube"] = cube_object()
    with pytest.raises(TypeError):
        scene.reset_pose("Cube")


def test_subdivision_math_closed_quads() -> None:
    scene = scene_with_rigged_mesh()
    scene.add_subsurf("Cube", "GP_Subdiv", 2, "catmull", "PRESERVE_BOUNDARIES", False)
    assert [m.name for m in scene.objects["Cube"].modifiers] == ["Armature", "GP_Subdiv"]
    scene.move_modifier_to_top("Cube", "GP_Subdiv")
    assert [m.name for m in scene.objects["Cube"].modifiers] == ["GP_Subdiv", "Armature"]
    scene.apply_modifier("Cube", "GP_Subdiv")
    assert scene.mesh_counts("Cube") == (98, 96, 192)
    assert [m.name for m in scene.objects["Cube"].modifiers] == ["Armature"]
    faces = scene.objects["Cube"].uv_layers["UVMap"]
    assert len(faces) == 96
    assert analyse(faces, []).overlap_ratio == 0.0  # subdivided atlas stays non-overlapping
    assert len(scene.face_loop_ranges("Cube")) == 96
    assert scene.face_loop_ranges("Cube")[1] == (4, 8)


def test_apply_modifier_rejects_shape_keys_and_unknown_names() -> None:
    scene = scene_two_meshes_shapekeys()
    scene.add_subsurf("Face", "GP_Subdiv", 1, "catmull", "PRESERVE_BOUNDARIES", False)
    with pytest.raises(RuntimeError, match="shape keys"):
        scene.apply_modifier("Face", "GP_Subdiv")
    scene.clear_shapekeys("Face")
    scene.apply_modifier("Face", "GP_Subdiv")
    assert scene.mesh_counts("Face") == (26, 24, 48)
    with pytest.raises(KeyError):
        scene.move_modifier_to_top("Face", "Nope")


def test_smart_unwrap_produces_a_non_overlapping_shared_atlas() -> None:
    scene = scene_two_meshes_shapekeys()
    scene.objects["Face"].uv_layers = {}
    scene.smart_unwrap(["Body", "Face"], 66.0, 0.02, 0.01)
    assert scene.uv_layers("Face") == ["UVMap"]
    faces = [
        scene.uv_loops(o, "UVMap")[s:e]
        for o in ("Body", "Face")
        for s, e in scene.face_loop_ranges(o)
    ]
    assert len(faces) == 12
    assert analyse(faces, []).overlap_ratio == 0.0
    assert analyse(faces, []).island_count == 12


def test_smart_unwrap_handles_triangles_and_zero_area_flag() -> None:
    tri_mesh = FakeObject(
        kind="MESH",
        verts=3,
        faces=1,
        tris=1,
        uv_layers={"UVMap": [[(0, 0), (1, 0), (0, 1)]]},
        parent_armature="A",
    )
    scene = FakeScene({"A": FakeObject(kind="ARMATURE", bones=three_bone_chain()), "Tri": tri_mesh})
    scene.smart_unwrap(["Tri"], 66.0, 0.02, 0.01)
    assert len(scene.uv_loops("Tri", "UVMap")) == 3
    scene.unwrap_zero_area = True
    scene.smart_unwrap(["Tri"], 66.0, 0.02, 0.01)
    assert analyse([scene.uv_loops("Tri", "UVMap")], []).uv_bbox_coverage == 0.0


def test_keep_first_uv_layer_and_merge_materials() -> None:
    scene = scene_with_rigged_mesh(
        extra_uv_layers={"Lightmap": cube_uv_atlas()}, materials=("a", "b")
    )
    scene.keep_first_uv_layer("Cube")
    assert scene.uv_layers("Cube") == ["UVMap"]
    scene.merge_material_slots(["Cube"], "m")
    assert scene.material_slots("Cube") == ["m"]
    assert scene.material_image_sizes("m") == [(256, 256)]
    scene.keep_first_uv_layer("Cube")  # idempotent
    cube_no_uv = cube_object(uv=None)
    assert cube_no_uv.face_loop_counts() == [4] * 6


def test_exports_and_blend_write_deterministic_files(tmp_path: Path) -> None:
    scene = scene_with_rigged_mesh()
    glb, fbx, blend = tmp_path / "a" / "s.glb", tmp_path / "b" / "r.fbx", tmp_path / "c" / "p.blend"
    scene.export_glb(["Cube"], str(glb), export_format="GLB")
    scene.export_fbx(["Armature", "Cube"], str(fbx), use_selection=True)
    scene.save_blend(str(blend))
    scene.open_blend(str(blend))
    assert glb.read_bytes() == FAKE_GLB_HEADER + b"Cube"
    assert fbx.read_bytes().startswith(b"FAKE_FBX")
    assert blend.read_bytes() == FAKE_BLEND_BYTES
    assert scene.exports[0] == ("glb", ["Cube"], str(glb), {"export_format": "GLB"})
    assert scene.exports[1][0] == "fbx"
    assert scene.saved_blends == [str(blend)]
    assert scene.opened_blends == [str(blend)]
    scene.export_error = RuntimeError("x")
    with pytest.raises(RuntimeError):
        scene.export_glb(["Cube"], str(glb))


def test_import_records_and_can_fail(tmp_path: Path) -> None:
    scene = scene_with_rigged_mesh()
    scene.import_fbx("a.fbx", use_anim=True)
    scene.import_glb("b.glb")
    assert scene.imports == [("fbx", "a.fbx", {"use_anim": True}), ("glb", "b.glb", {})]
    scene.import_error = RuntimeError("nope")
    with pytest.raises(RuntimeError):
        scene.import_glb("b.glb")
    scene.reset()
    assert scene.objects == {}
    assert scene.call_order()[-1] == "reset"


def test_positions_bbox_and_transforms() -> None:
    scene = scene_with_rigged_mesh()
    assert scene.world_vertex_positions("Cube")[:2] == [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]
    assert scene.object_bbox(["Cube", "Armature"]) == ((0.0, 0.0, 0.0), (7.0, 0.0, 0.0))
    assert scene.bbox_diagonal(["Cube"]) == 7.0
    assert scene.object_bbox(["Armature"]) == ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    scene.transform_objects(["Cube"], 2.0, (1.0, 0.5, 0.0))
    assert scene.world_vertex_positions("Cube")[1] == (3.0, 0.5, 0.0)
    scene.transform_objects(["Cube"], 0.5, (0.0, 0.0, 0.0))
    assert scene.world_vertex_positions("Cube")[0] == (0.5, 0.25, 0.0)


def test_mutate_rig_changes_the_fingerprint() -> None:
    scene = scene_with_rigged_mesh()
    before = fingerprint_sha256(scene.bones("Armature"))
    scene.mutate_rig()
    assert fingerprint_sha256(scene.bones("Armature")) != before
    assert scene.bones("Armature")[0].name == "Root"


def test_bake_and_material_methods(tmp_path: Path) -> None:
    scene = scene_with_rigged_mesh()
    scene.set_render_engine_cycles("CPU", 16)
    scene.configure_bake(16, 0.01, 0.05)
    scene.hide_from_render("Armature")
    out = tmp_path / "raw" / "basecolor.png"
    scene.bake_map(
        ["Cube"],
        ["Cube"],
        map_name="basecolor",
        bake_type="DIFFUSE",
        resolution=64,
        out_path=str(out),
        colorspace="sRGB",
        emit_input=None,
    )
    scene.bind_urp_material(["Cube"], "cube_mat", {"basecolor": "t/basecolor.png"})
    assert scene.render == {"engine": "CYCLES", "device": "CPU", "samples": 16}
    assert scene.bake_settings == {
        "margin_px": 16,
        "cage_extrusion": 0.01,
        "max_ray_distance": 0.05,
    }
    assert scene.objects["Armature"].hide_render is True
    assert out.read_bytes().startswith(b"\x89PNG")
    assert scene.baked[0]["map_name"] == "basecolor"
    assert scene.material_slots("Cube") == ["cube_mat"]
    assert scene.bound_textures == {"Cube": {"basecolor": "t/basecolor.png"}}
    scene.shade_smooth("Cube")
    assert scene.objects["Cube"].smooth is True


def test_bake_fake_imports_sources_rewires_and_reports_stats(tmp_path: Path) -> None:
    scene = scene_for_bake(source=source_object(scale=0.5, translation=(1.0, 0.0, 0.0)))
    assert "result" not in scene.objects
    scene.import_glb("result.glb", merge_vertices=False)
    assert scene.meshes() == ["Cube", "result"]
    assert scene.object_bbox(["result"]) == ((1.0, 0.0, 0.0), (4.5, 0.0, 0.0))
    assert [s.name for s in scene.shader_inputs("result")][:3] == [
        "Base Color",
        "Normal",
        "Roughness",
    ]
    scene.set_render_engine_cycles("CPU", 1)
    out = tmp_path / "metallic.png"
    scene.bake_map(
        ["Cube"],
        ["result"],
        map_name="metallic",
        bake_type="EMIT",
        resolution=8,
        out_path=str(out),
        colorspace="Non-Color",
        emit_input="Metallic",
    )
    assert scene.rewires == [
        ("rewire", "metallic", "Metallic"),
        ("restore", "metallic", "Metallic"),
    ]
    assert scene.baked[-1]["targets"] == ["Cube"]
    assert scene.baked[-1]["samples"] == 1
    assert scene.image_stats(str(out)) == (0.5, 0.02)
    scene.empty_images.add("metallic")
    assert scene.image_stats(str(out)) == (0.0, 0.0)
    with pytest.raises(RuntimeError, match="cannot load"):
        scene.image_stats(str(tmp_path / "missing.png"))
    scene.set_pose_position("Armature", True)
    assert scene.objects["Armature"].pose_rest is True
    with pytest.raises(TypeError):
        scene.set_pose_position("Cube", True)


def test_bake_fake_failure_injection_still_restores_rewiring(tmp_path: Path) -> None:
    scene = scene_for_bake()
    scene.import_glb("result.glb")
    scene.bake_errors["occlusion"] = RuntimeError("no active image")
    with pytest.raises(RuntimeError, match="no active image"):
        scene.bake_map(
            ["Cube"],
            ["result"],
            map_name="occlusion",
            bake_type="EMIT",
            resolution=8,
            out_path=str(tmp_path / "occlusion.png"),
            colorspace="Non-Color",
            emit_input="Occlusion",
        )
    assert scene.rewires[-1] == ("restore", "occlusion", "Occlusion")
    assert scene.baked == []
    assert not (tmp_path / "occlusion.png").exists()


def test_principled_inputs_builder_knobs() -> None:
    names = [s.name for s in principled_inputs()]
    assert names == [
        "Base Color",
        "Normal",
        "Roughness",
        "Metallic",
        "Emission Color",
        "Emission Strength",
    ]
    sparse = principled_inputs(normal_linked=False, roughness=None, metallic=None)
    assert [s.name for s in sparse] == ["Base Color", "Emission Color", "Emission Strength"]
    occ = principled_inputs(occlusion_linked=True, emission=(1.0, 0.5, 0.0))
    assert any(s.name == "Occlusion" and s.linked for s in occ)
    assert next(s for s in occ if s.name == "Emission Color").value == (1.0, 0.5, 0.0, 1.0)
    assert scene_for_bake(targets=2).skinned_meshes("Armature") == ["Cube", "Cube.001"]


def test_type_guards() -> None:
    scene = scene_with_rigged_mesh()
    with pytest.raises(TypeError):
        scene.mesh_counts("Armature")
    with pytest.raises(TypeError):
        scene.bones("Cube")
    assert FakeModifier("x", "SUBSURF").levels == 0


# --------------------------------------------------------------------------- parity with BpyScene
def _stub_module(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__getattr__ = lambda attr: MagicMock(name=f"{name}.{attr}")  # type: ignore[method-assign]
    return module


@pytest.fixture
def bpy_scene_module(monkeypatch: pytest.MonkeyPatch) -> Any:
    for name in ("bpy", "bmesh", "mathutils"):
        monkeypatch.setitem(sys.modules, name, _stub_module(name))
    monkeypatch.delitem(sys.modules, "tinyworld_pipeline.blender_jobs.bpy_scene", raising=False)
    module = importlib.import_module("tinyworld_pipeline.blender_jobs.bpy_scene")
    yield module
    monkeypatch.delitem(sys.modules, "tinyworld_pipeline.blender_jobs.bpy_scene", raising=False)


def _public_methods(cls: type) -> dict[str, inspect.Signature]:
    return {
        name: inspect.signature(member)
        for name, member in inspect.getmembers(cls, inspect.isfunction)
        if not name.startswith("_")
    }


def _shape(sig: inspect.Signature) -> list[tuple[str, inspect._ParameterKind, object]]:
    return [(p.name, p.kind, p.default) for p in sig.parameters.values()]


def test_fake_scene_and_bpy_scene_have_identical_public_methods(bpy_scene_module: Any) -> None:
    protocol = _public_methods(SceneApi)
    fake = _public_methods(FakeScene)
    real = _public_methods(bpy_scene_module.BpyScene)
    extras = {"method_calls", "call_order", "mutate_rig"}  # test-only hooks on the fake
    assert set(protocol) <= set(fake)
    assert set(fake) - extras == set(protocol)
    assert set(real) == set(protocol)
    for name, signature in protocol.items():
        assert _shape(fake[name]) == _shape(signature), name
        assert _shape(real[name]) == _shape(signature), name
