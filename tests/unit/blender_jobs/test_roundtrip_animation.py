"""``roundtrip_animation`` against ``FakeScene`` (plan 12 §1.3, §5)."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import pytest

from fakes.fake_scene import (
    FakeScene,
    armature_object,
    cube_object,
    scene_no_armature,
    scene_two_armatures,
    three_bone_chain,
)
from grounded_pipeline.blender import protocol as orch
from grounded_pipeline.blender_jobs import protocol as bj
from grounded_pipeline.blender_jobs.errors import JobFailure
from grounded_pipeline.blender_jobs.fbx_settings import FBX_IMPORT_KWARGS, fbx_export_kwargs
from grounded_pipeline.blender_jobs.jobs import JOBS, roundtrip_animation
from grounded_pipeline.blender_jobs.jobs.roundtrip_animation import (
    compare_bone_names,
    parse_params,
    run_with_scene,
)

BONES = ["Root", "Spine", "Head"]


def clip_scene(*, with_mesh: bool = False, bones: list[Any] | None = None) -> FakeScene:
    """An empty scene whose ``import_fbx`` yields ``rigged_cube@walk.fbx`` (armature + action)."""
    scene = FakeScene()
    scene.fbx_payload = {"Armature": armature_object(bones)}
    if with_mesh:
        scene.fbx_payload["Dummy"] = cube_object(materials=())
    scene.fbx_actions = [("Armature|Armature|walk", "OBJECT")]
    return scene


def make_params(tmp_path: Path, expected: list[str] | None = None, **export: Any) -> Any:
    source = tmp_path / "source" / "animations" / "cube_rigged@walk.fbx"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"FBX")
    return bj.RoundtripAnimationParams(
        source_fbx=str(source),
        out_fbx=str(tmp_path / "work" / "export" / "animations" / "cube_rigged@walk.fbx"),
        export=bj.ExportSection(**export),
        expected_bone_names=BONES if expected is None else expected,
    )


def failure(scene: FakeScene, params: Any) -> JobFailure:
    with pytest.raises(JobFailure) as info:
        run_with_scene(params, scene)
    return info.value


# --------------------------------------------------------------------------- happy path
def test_happy_path(tmp_path: Path) -> None:
    scene = clip_scene()
    params = make_params(tmp_path)

    data, warnings = run_with_scene(params, scene)

    assert warnings == []
    assert scene.call_order() == [
        "reset",
        "import_fbx",
        "clean_action_names",
        "reset_pose",
        "export_fbx",
    ]
    assert scene.raw_actions == [("walk", "OBJECT")]
    assert scene.imports == [("fbx", params.source_fbx, FBX_IMPORT_KWARGS)]
    assert scene.pose_resets == ["Armature"]
    kind, objs, path, kwargs = scene.exports[0]
    assert (kind, objs, path) == ("fbx", ["Armature"], params.out_fbx)
    assert kwargs == fbx_export_kwargs("FBX_SCALE_ALL", False)
    assert kwargs["object_types"] == {"ARMATURE", "MESH"}
    assert Path(params.out_fbx).is_file()

    assert [b.name for b in data.rig.bones] == BONES
    assert data.rig.actions == ["walk"]
    assert data.actions_exported == ["walk"]
    assert data.missing_bones == []
    assert data.extra_bones == []


def test_export_section_is_honoured(tmp_path: Path) -> None:
    scene = clip_scene()
    run_with_scene(make_params(tmp_path, fbx_scale_option="FBX_SCALE_NONE"), scene)
    assert scene.exports[0][3] == fbx_export_kwargs("FBX_SCALE_NONE", False)


def test_dummy_meshes_are_kept_untouched_and_exported(tmp_path: Path) -> None:
    scene = clip_scene(with_mesh=True)
    data, _warnings = run_with_scene(make_params(tmp_path), scene)
    assert scene.exports[0][1] == ["Armature", "Dummy"]
    assert scene.method_calls("add_subsurf") == []
    assert scene.method_calls("smart_unwrap") == []
    assert scene.method_calls("merge_material_slots") == []
    assert scene.objects["Dummy"].faces == 6
    assert data.rig.bone_count == 3


# --------------------------------------------------------------------------- bone comparison
def test_missing_and_extra_bones_are_recorded_and_warned(tmp_path: Path) -> None:
    scene = clip_scene()
    params = make_params(tmp_path, expected=["Root", "Spine", "Neck"])
    data, warnings = run_with_scene(params, scene)
    assert data.missing_bones == ["Neck"]
    assert data.extra_bones == ["Head"]
    assert len(warnings) == 2
    assert "Neck" in warnings[0]
    assert "Head" in warnings[1]


def test_compare_bone_names_sorts_both_lists() -> None:
    rig = bj.RigFingerprint(
        armature_object="Armature",
        armature_data="Armature",
        bone_count=2,
        bones=[
            bj.Bone("Zed", None, (0, 0, 0), (0, 0, 1), [[1.0]], "Armature/Zed"),
            bj.Bone("Alpha", None, (0, 0, 0), (0, 0, 1), [[1.0]], "Armature/Alpha"),
        ],
        actions=[],
        fingerprint_sha256="0" * 64,
    )
    assert compare_bone_names(rig, ["Zed", "Mid", "Beta"]) == (["Beta", "Mid"], ["Alpha"])
    assert compare_bone_names(rig, ["Alpha", "Zed"]) == ([], [])


def test_rig_fingerprint_is_the_imported_rig(tmp_path: Path) -> None:
    bones = three_bone_chain()
    scene = clip_scene(bones=bones)
    data, _warnings = run_with_scene(make_params(tmp_path), scene)
    assert [b.head for b in data.rig.bones] == [b.head for b in bones]
    assert [b.transform_path for b in data.rig.bones] == [
        "Armature/Root",
        "Armature/Root/Spine",
        "Armature/Root/Spine/Head",
    ]


# --------------------------------------------------------------------------- failures
def test_import_failure(tmp_path: Path) -> None:
    scene = clip_scene()
    scene.import_error = RuntimeError("not an FBX")
    err = failure(scene, make_params(tmp_path))
    assert err.code == bj.E_BJ_IMPORT_FAILED
    assert "not an FBX" in err.message


def test_no_armature(tmp_path: Path) -> None:
    scene = FakeScene()
    scene.fbx_payload = dict(scene_no_armature().objects)
    err = failure(scene, make_params(tmp_path))
    assert err.code == bj.E_BJ_NO_ARMATURE
    assert err.detail["meshes"] == ["Cube"]


def test_two_armatures(tmp_path: Path) -> None:
    scene = FakeScene()
    scene.fbx_payload = dict(scene_two_armatures().objects)
    err = failure(scene, make_params(tmp_path))
    assert err.code == bj.E_BJ_MULTI_ARMATURE


def test_reset_runs_before_import_so_stale_objects_vanish(tmp_path: Path) -> None:
    scene = clip_scene()
    scene.objects["Stale"] = cube_object(armature=None)
    data, _warnings = run_with_scene(make_params(tmp_path), scene)
    assert "Stale" not in scene.objects
    assert scene.exports[0][1] == ["Armature"]
    assert data.rig.armature_object == "Armature"


def test_export_failure(tmp_path: Path) -> None:
    scene = clip_scene()
    scene.export_error = RuntimeError("nope")
    err = failure(scene, make_params(tmp_path))
    assert err.code == bj.E_BJ_EXPORT_FAILED


def test_mutation_guard(tmp_path: Path) -> None:
    scene = clip_scene()
    original_export = scene.export_fbx

    def export_and_mutate(objs: list[str], path: str, **kwargs: Any) -> None:
        original_export(objs, path, **kwargs)
        scene.mutate_rig()

    scene.export_fbx = export_and_mutate  # type: ignore[method-assign]
    err = failure(scene, make_params(tmp_path))
    assert err.code == bj.E_BJ_RIG_MUTATED


# --------------------------------------------------------------------------- params / registry
def test_parse_params_round_trips_and_rejects_drift(tmp_path: Path) -> None:
    params = make_params(tmp_path)
    raw = orch.RoundtripAnimationParams.model_validate(bj.to_dict(params)).model_dump(mode="json")
    assert parse_params(raw) == params
    with pytest.raises(JobFailure) as info:
        parse_params({**raw, "bogus": 1})
    assert info.value.code == bj.E_BJ_BAD_PARAMS


def test_run_builds_scene_lazily_and_returns_json(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    scene = clip_scene()
    module = types.ModuleType("grounded_pipeline.blender_jobs.bpy_scene")
    module.BpyScene = lambda: scene  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "grounded_pipeline.blender_jobs.bpy_scene", module)

    data, warnings = roundtrip_animation.run(bj.to_dict(make_params(tmp_path)))

    assert warnings == []
    assert orch.RoundtripAnimationData.model_validate(data).actions_exported == ["walk"]


def test_registry_exposes_roundtrip_animation() -> None:
    assert JOBS["roundtrip_animation"] is roundtrip_animation.run
