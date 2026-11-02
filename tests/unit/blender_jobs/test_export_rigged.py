"""``export_rigged`` against ``FakeScene`` (plan 12 §1.2, §5)."""

from __future__ import annotations

import hashlib
import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from fakes.fake_scene import (
    FakeScene,
    scene_no_armature,
    scene_two_armatures,
    scene_unskinned_extra,
    scene_with_rigged_mesh,
)
from grounded_pipeline.blender import protocol as orch
from grounded_pipeline.blender_jobs import protocol as bj
from grounded_pipeline.blender_jobs.errors import JobFailure
from grounded_pipeline.blender_jobs.fbx_settings import fbx_export_kwargs
from grounded_pipeline.blender_jobs.jobs import JOBS, export_rigged
from grounded_pipeline.blender_jobs.jobs.export_rigged import (
    URP_TEXTURE_KEYS,
    find_urp_textures,
    parse_params,
    run_with_scene,
)

MODEL = "cube_rigged"
MATERIAL = f"{MODEL}_mat"


# --------------------------------------------------------------------------- helpers
def make_params(
    tmp_path: Path, *, textures: bool = True, include_unskinned: bool = True, **export: Any
) -> bj.ExportRiggedParams:
    blend = tmp_path / "work" / "prepared.blend"
    blend.parent.mkdir(parents=True, exist_ok=True)
    blend.write_bytes(b"BLENDER")
    return bj.ExportRiggedParams(
        prepared_blend=str(blend),
        textures_dir=str(tmp_path / "work" / "textures" / "urp") if textures else None,
        out_fbx=str(tmp_path / "work" / "export" / f"{MODEL}.fbx"),
        out_rig_fingerprint=str(tmp_path / "work" / "export" / "rig_fingerprint_out.json"),
        export=bj.ExportSection(**export),
        material_name=MATERIAL,
        include_unskinned=include_unskinned,
    )


def write_urp(tmp_path: Path, *names: str) -> Path:
    urp = tmp_path / "work" / "textures" / "urp"
    urp.mkdir(parents=True, exist_ok=True)
    for name in names:
        (urp / name).write_bytes(b"\x89PNG")
    return urp


def run(scene: FakeScene, params: bj.ExportRiggedParams) -> tuple[bj.ExportRiggedData, list[str]]:
    return run_with_scene(params, scene)


def failure(scene: FakeScene, params: bj.ExportRiggedParams) -> JobFailure:
    with pytest.raises(JobFailure) as info:
        run(scene, params)
    return info.value


# --------------------------------------------------------------------------- happy path
def test_happy_path_with_all_maps(tmp_path: Path) -> None:
    urp = write_urp(
        tmp_path,
        f"{MODEL}_basecolor.png",
        f"{MODEL}_normal.png",
        f"{MODEL}_metallic_smoothness.png",
        f"{MODEL}_emission.png",
        f"{MODEL}_occlusion.png",
    )
    scene = scene_with_rigged_mesh()
    params = make_params(tmp_path)

    data, warnings = run(scene, params)

    assert warnings == []
    assert scene.opened_blends == [params.prepared_blend]
    assert scene.call_order() == [
        "open_blend",
        "bind_urp_material",
        "clean_action_names",
        "reset_pose",
        "export_fbx",
    ]
    assert scene.raw_actions == [("idle", "OBJECT")]  # "Armature|Armature|idle" renamed
    assert scene.objects["Cube"].materials == [MATERIAL]
    assert scene.bound_textures["Cube"] == {
        "basecolor": str(urp / f"{MODEL}_basecolor.png"),
        "normal": str(urp / f"{MODEL}_normal.png"),
        "metallic_smoothness": str(urp / f"{MODEL}_metallic_smoothness.png"),
        "emission": str(urp / f"{MODEL}_emission.png"),
    }  # occlusion has no Principled input and is not bound
    assert scene.pose_resets == ["Armature"]
    kind, objs, path, kwargs = scene.exports[0]
    assert (kind, objs, path) == ("fbx", ["Armature", "Cube"], params.out_fbx)
    assert kwargs == fbx_export_kwargs("FBX_SCALE_ALL", False)

    assert data.rig.armature_object == "Armature"
    assert [b.name for b in data.rig.bones] == ["Root", "Spine", "Head"]
    assert data.rig.actions == ["idle"]
    assert data.actions_exported == ["idle"]
    assert data.fbx_sha256 == hashlib.sha256(Path(params.out_fbx).read_bytes()).hexdigest()
    written = json.loads(Path(params.out_rig_fingerprint).read_text(encoding="utf-8"))
    assert written == bj.to_dict(data.rig)
    assert orch.RigFingerprint.model_validate(written).fingerprint_sha256 == (
        data.rig.fingerprint_sha256
    )


def test_export_kwargs_follow_the_export_section(tmp_path: Path) -> None:
    scene = scene_with_rigged_mesh()
    params = make_params(
        tmp_path, textures=False, fbx_scale_option="FBX_SCALE_UNITS", embed_textures=True
    )
    run(scene, params)
    kwargs = scene.exports[0][3]
    assert kwargs == fbx_export_kwargs("FBX_SCALE_UNITS", True)
    assert kwargs["apply_scale_options"] == "FBX_SCALE_UNITS"
    assert kwargs["path_mode"] == "COPY"
    assert kwargs["embed_textures"] is True
    assert kwargs["add_leaf_bones"] is False
    assert kwargs["bake_anim_use_all_actions"] is True


# --------------------------------------------------------------------------- material binding
def test_no_textures_dir_skips_material_binding(tmp_path: Path) -> None:
    scene = scene_with_rigged_mesh()
    data, warnings = run(scene, make_params(tmp_path, textures=False))
    assert warnings == []
    assert scene.method_calls("bind_urp_material") == []
    assert scene.objects["Cube"].materials == ["cube_mat"]
    assert data.actions_exported == ["idle"]


def test_empty_textures_dir_warns_and_exports_without_binding(tmp_path: Path) -> None:
    write_urp(tmp_path)
    scene = scene_with_rigged_mesh()
    _data, warnings = run(scene, make_params(tmp_path))
    assert len(warnings) == 1
    assert "no URP textures found" in warnings[0]
    assert scene.method_calls("bind_urp_material") == []
    assert scene.method_calls("export_fbx")


def test_missing_textures_dir_behaves_like_empty(tmp_path: Path) -> None:
    scene = scene_with_rigged_mesh()
    _data, warnings = run(scene, make_params(tmp_path))  # dir never created
    assert len(warnings) == 1
    assert scene.method_calls("bind_urp_material") == []


def test_partial_maps_bind_only_what_exists(tmp_path: Path) -> None:
    urp = write_urp(tmp_path, f"{MODEL}_basecolor.png", f"{MODEL}_normal.png")
    scene = scene_with_rigged_mesh()
    run(scene, make_params(tmp_path))
    assert scene.bound_textures["Cube"] == {
        "basecolor": str(urp / f"{MODEL}_basecolor.png"),
        "normal": str(urp / f"{MODEL}_normal.png"),
    }


def test_find_urp_textures_prefers_model_prefix_then_any_suffix_match(tmp_path: Path) -> None:
    urp = write_urp(
        tmp_path, "other_basecolor.png", f"{MODEL}_basecolor.png", "zzz_normal.png", "a_normal.png"
    )
    found = find_urp_textures(str(urp), MATERIAL)
    assert found == {
        "basecolor": str(urp / f"{MODEL}_basecolor.png"),
        "normal": str(urp / "a_normal.png"),
    }
    assert find_urp_textures(str(tmp_path / "nope"), MATERIAL) == {}
    assert URP_TEXTURE_KEYS == ("basecolor", "normal", "metallic_smoothness", "emission")


def test_material_is_bound_to_every_skinned_mesh_only(tmp_path: Path) -> None:
    write_urp(tmp_path, f"{MODEL}_basecolor.png")
    scene = scene_unskinned_extra()
    run(scene, make_params(tmp_path))
    call = scene.method_calls("bind_urp_material")[0]
    assert call[1] == ("Cube",)
    assert scene.objects["Prop"].materials == ["prop_mat"]


# --------------------------------------------------------------------------- selection
def test_unskinned_meshes_are_exported_when_included(tmp_path: Path) -> None:
    scene = scene_unskinned_extra()
    run(scene, make_params(tmp_path, textures=False))
    assert scene.exports[0][1] == ["Armature", "Cube", "Prop"]


def test_unskinned_meshes_are_dropped_when_excluded(tmp_path: Path) -> None:
    scene = scene_unskinned_extra()
    run(scene, make_params(tmp_path, textures=False, include_unskinned=False))
    assert scene.exports[0][1] == ["Armature", "Cube"]


def test_pose_is_reset_before_export(tmp_path: Path) -> None:
    scene = scene_with_rigged_mesh()
    run(scene, make_params(tmp_path, textures=False))
    order = scene.call_order()
    assert order.index("reset_pose") < order.index("export_fbx")


# --------------------------------------------------------------------------- failures
def test_missing_blend_is_import_failure(tmp_path: Path) -> None:
    params = make_params(tmp_path, textures=False)
    Path(params.prepared_blend).unlink()
    err = failure(scene_with_rigged_mesh(), params)
    assert err.code == bj.E_BJ_IMPORT_FAILED
    assert err.detail["prepared_blend"] == params.prepared_blend


def test_open_blend_error_is_import_failure(tmp_path: Path) -> None:
    scene = scene_with_rigged_mesh()

    def boom(path: str) -> None:
        raise RuntimeError(f"corrupt: {path}")

    scene.open_blend = boom  # type: ignore[method-assign]
    err = failure(scene, make_params(tmp_path, textures=False))
    assert err.code == bj.E_BJ_IMPORT_FAILED
    assert "corrupt" in err.message


def test_no_armature(tmp_path: Path) -> None:
    err = failure(scene_no_armature(), make_params(tmp_path, textures=False))
    assert err.code == bj.E_BJ_NO_ARMATURE
    assert err.detail["meshes"] == ["Cube"]


def test_two_armatures(tmp_path: Path) -> None:
    err = failure(scene_two_armatures(), make_params(tmp_path, textures=False))
    assert err.code == bj.E_BJ_MULTI_ARMATURE
    assert err.detail["armatures"] == ["Armature", "Armature.001"]


def test_no_skinned_mesh(tmp_path: Path) -> None:
    scene = scene_with_rigged_mesh()
    scene.objects["Cube"].parent_armature = None
    err = failure(scene, make_params(tmp_path, textures=False))
    assert err.code == bj.E_BJ_NO_SKINNED_MESH
    assert err.detail["armature"] == "Armature"


def test_export_error_and_missing_file_are_export_failures(tmp_path: Path) -> None:
    scene = scene_with_rigged_mesh()
    scene.export_error = RuntimeError("disk full")
    err = failure(scene, make_params(tmp_path, textures=False))
    assert err.code == bj.E_BJ_EXPORT_FAILED
    assert "disk full" in err.message

    scene = scene_with_rigged_mesh()
    scene.export_writes_file = False
    err = failure(scene, make_params(tmp_path, textures=False))
    assert err.code == bj.E_BJ_EXPORT_FAILED
    assert err.message == "FBX export wrote no file"


def test_mutation_guard(tmp_path: Path) -> None:
    scene = scene_with_rigged_mesh()
    original_export = scene.export_fbx

    def export_and_mutate(objs: list[str], path: str, **kwargs: Any) -> None:
        original_export(objs, path, **kwargs)
        scene.mutate_rig()

    scene.export_fbx = export_and_mutate  # type: ignore[method-assign]
    err = failure(scene, make_params(tmp_path, textures=False))
    assert err.code == bj.E_BJ_RIG_MUTATED
    assert err.detail["before"] != err.detail["after"]


# --------------------------------------------------------------------------- params / registry
def test_parse_params_round_trips_the_orchestrator_model(tmp_path: Path) -> None:
    params = make_params(tmp_path)
    raw = orch.ExportRiggedParams.model_validate(bj.to_dict(params)).model_dump(mode="json")
    assert parse_params(raw) == params
    assert parse_params({**raw, "textures_dir": None}).textures_dir is None


def test_parse_params_rejects_unknown_keys(tmp_path: Path) -> None:
    raw = bj.to_dict(make_params(tmp_path))
    with pytest.raises(JobFailure) as info:
        parse_params({**raw, "bogus": 1})
    assert info.value.code == bj.E_BJ_BAD_PARAMS


def test_run_builds_scene_lazily_and_returns_json(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    scene = scene_with_rigged_mesh()
    module = types.ModuleType("grounded_pipeline.blender_jobs.bpy_scene")
    module.BpyScene = lambda: scene  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "grounded_pipeline.blender_jobs.bpy_scene", module)
    raw = bj.to_dict(make_params(tmp_path, textures=False))

    data, warnings = export_rigged.run(raw)

    assert warnings == []
    validated = orch.ExportRiggedData.model_validate(data)
    assert validated.actions_exported == ["idle"]
    assert validated.rig.bone_count == 3


def test_registry_exposes_export_rigged() -> None:
    assert JOBS["export_rigged"] is export_rigged.run
