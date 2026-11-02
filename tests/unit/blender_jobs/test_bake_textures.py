"""``bake_textures`` against ``FakeScene`` (plan 11 §5.2, §8)."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import pytest

from fakes.fake_scene import (
    FakeScene,
    cube_object,
    principled_inputs,
    scene_for_bake,
    source_object,
)
from grounded_pipeline.blender import protocol as orch
from grounded_pipeline.blender_jobs import protocol as bj
from grounded_pipeline.blender_jobs.errors import JobFailure
from grounded_pipeline.blender_jobs.jobs import JOBS, bake_textures
from grounded_pipeline.blender_jobs.jobs.bake_textures import (
    ALIGN_TOLERANCE,
    CYCLES_DEVICE_ENV,
    GLB_IMPORT_KWARGS,
    MAP_SPECS,
    alignment,
    cycles_device,
    parse_params,
    run_with_scene,
    source_presence,
)
from grounded_pipeline.blender_jobs.scene_api import ShaderInput

ALL_MAPS = ["basecolor", "normal", "metallic", "roughness", "occlusion", "emission"]


# --------------------------------------------------------------------------- helpers
def make_params(tmp_path: Path, **bake: Any) -> bj.BakeTexturesParams:
    return bj.BakeTexturesParams(
        prepared_blend=str(tmp_path / "work" / "prepared.blend"),
        source_glb=str(tmp_path / "work" / "tripo" / "result.glb"),
        out_dir=str(tmp_path / "work" / "textures" / "raw"),
        bake=bj.BakeSection(**bake),
        bbox_diagonal=2.0,
    )


def run(
    scene: FakeScene, tmp_path: Path, *, env: dict[str, str] | None = None, **bake: Any
) -> tuple[bj.BakeTexturesData, list[str]]:
    return run_with_scene(make_params(tmp_path, **bake), scene, env=env or {})


def failure(scene: FakeScene, tmp_path: Path, **bake: Any) -> JobFailure:
    with pytest.raises(JobFailure) as info:
        run(scene, tmp_path, **bake)
    return info.value


def full_source(**overrides: Any) -> Any:
    """Source whose material carries every input (textures for all maps)."""
    inputs = principled_inputs(
        roughness_linked=True, metallic_linked=True, occlusion_linked=True, emission_linked=True
    )
    return source_object(inputs=inputs, **overrides)


# --------------------------------------------------------------------------- map loop
def test_map_loop_bakes_every_configured_map_in_order(tmp_path: Path) -> None:
    scene = scene_for_bake(source=full_source())

    data, warnings = run(scene, tmp_path, maps=ALL_MAPS, resolution=512, samples=8)

    assert warnings == []
    assert [b["map_name"] for b in scene.baked] == ALL_MAPS
    assert list(data.maps) == ALL_MAPS
    assert data.resolution == 512
    assert data.samples == 8
    out = tmp_path / "work" / "textures" / "raw"
    assert data.maps["basecolor"] == str(out / "basecolor.png")
    assert all(Path(p).is_file() for p in data.maps.values())
    assert set(data.bake_seconds) == set(ALL_MAPS)
    assert all(s >= 0.0 for s in data.bake_seconds.values())
    assert scene.opened_blends == [str(tmp_path / "work" / "prepared.blend")]
    assert scene.imports == [
        ("glb", str(tmp_path / "work" / "tripo" / "result.glb"), GLB_IMPORT_KWARGS)
    ]
    assert scene.saved_blends == []  # never saves prepared.blend


def test_type_table_colorspaces_emit_inputs_and_samples(tmp_path: Path) -> None:
    scene = scene_for_bake(source=full_source())
    run(scene, tmp_path, maps=ALL_MAPS, samples=16)
    by_name = {b["map_name"]: b for b in scene.baked}
    assert by_name["normal"]["bake_type"] == "NORMAL"
    assert by_name["roughness"]["bake_type"] == "ROUGHNESS"
    emit_maps = ("basecolor", "metallic", "occlusion", "emission")
    assert all(by_name[m]["bake_type"] == "EMIT" for m in emit_maps)
    assert by_name["basecolor"]["emit_input"] == "Base Color"  # exact; DIFFUSE is energy-scaled
    assert by_name["metallic"]["emit_input"] == "Metallic"
    assert by_name["occlusion"]["emit_input"] == "Occlusion"
    assert by_name["emission"]["emit_input"] is None  # native EMIT honours Emission Strength
    assert by_name["basecolor"]["colorspace"] == "sRGB"
    assert by_name["emission"]["colorspace"] == "sRGB"
    assert all(
        by_name[m]["colorspace"] == "Non-Color"
        for m in ("normal", "roughness", "metallic", "occlusion")
    )
    assert {m: by_name[m]["samples"] for m in ALL_MAPS} == {
        "basecolor": 1,
        "normal": 16,
        "roughness": 16,
        "metallic": 1,
        "occlusion": 1,
        "emission": 1,
    }
    assert all(b["targets"] == ["Cube"] and b["sources"] == ["result"] for b in scene.baked)
    assert all(b["resolution"] == 2048 for b in scene.baked)


def test_emit_rewiring_is_restored_before_the_next_map(tmp_path: Path) -> None:
    scene = scene_for_bake(source=full_source())
    run(scene, tmp_path, maps=["metallic", "normal", "occlusion", "basecolor", "emission"])
    assert scene.rewires == [
        ("rewire", "metallic", "Metallic"),
        ("restore", "metallic", "Metallic"),
        ("rewire", "occlusion", "Occlusion"),
        ("restore", "occlusion", "Occlusion"),
        ("rewire", "basecolor", "Base Color"),
        ("restore", "basecolor", "Base Color"),
    ]  # normal (native) and emission (native EMIT) never rewire


def test_skips_maps_the_source_cannot_feed_with_a_warning(tmp_path: Path) -> None:
    # Base colour texture, flat roughness 0.5, metallic 0, no normal/occlusion/emission.
    scene = scene_for_bake(source=source_object(inputs=principled_inputs(normal_linked=False)))

    data, warnings = run(scene, tmp_path, maps=ALL_MAPS)

    assert list(data.maps) == ["basecolor", "roughness"]
    assert [b["map_name"] for b in scene.baked] == ["basecolor", "roughness"]
    assert warnings == [
        "normal: source material has no Normal data; skipped",
        "metallic: source material has no Metallic data; skipped",
        "occlusion: source material has no Occlusion data; skipped",
        "emission: source material has no Emission Color data; skipped",
    ]
    assert set(data.bake_seconds) == {"basecolor", "roughness"}


def test_constant_metallic_and_emission_are_baked(tmp_path: Path) -> None:
    inputs = principled_inputs(metallic=0.25, emission=(0.2, 0.0, 0.0), emission_strength=2.0)
    scene = scene_for_bake(source=source_object(inputs=inputs))
    data, warnings = run(scene, tmp_path, maps=["metallic", "emission", "occlusion"])
    assert list(data.maps) == ["metallic", "emission"]
    assert warnings == ["occlusion: source material has no Occlusion data; skipped"]
    assert scene.method_calls("image_stats") == []  # constants are allowed to bake black


def test_unknown_map_name_is_bad_params(tmp_path: Path) -> None:
    scene = scene_for_bake(source=full_source())
    error = failure(scene, tmp_path, maps=["basecolor", "specular"])
    assert error.code == bj.E_BJ_BAD_PARAMS
    assert error.detail["known"] == sorted(MAP_SPECS)


# --------------------------------------------------------------------------- scene setup
def test_alignment_branch_transforms_only_the_source(tmp_path: Path) -> None:
    # Target cube spans x 0..7; a half-size source shifted by +1 spans 1..4.5.
    scene = scene_for_bake(source=full_source(scale=0.5, translation=(1.0, 0.0, 0.0)))

    _, warnings = run(scene, tmp_path, maps=["basecolor"])

    calls = scene.method_calls("transform_objects")
    assert len(calls) == 1
    (_, objs, scale, translation) = calls[0]
    assert objs == ("result",)
    assert scale == pytest.approx(2.0)
    assert translation == pytest.approx((-2.0, 0.0, 0.0))
    assert scene.object_bbox(["result"]) == scene.object_bbox(["Cube"])
    assert scene.world_vertex_positions("Cube")[1] == (1.0, 0.0, 0.0)  # target untouched
    assert len(warnings) == 1
    assert warnings[0].startswith("source aligned to target bbox: scale=2")
    order = scene.call_order()
    assert order.index("transform_objects") < order.index("bake_map")


def test_no_alignment_when_boxes_already_match(tmp_path: Path) -> None:
    scene = scene_for_bake(source=full_source())
    _, warnings = run(scene, tmp_path, maps=["basecolor"])
    assert scene.method_calls("transform_objects") == []
    assert warnings == []


def test_armature_hidden_and_put_in_rest_pose_around_the_bakes(tmp_path: Path) -> None:
    scene = scene_for_bake(source=full_source())
    run(scene, tmp_path, maps=["basecolor", "normal"])
    assert scene.objects["Armature"].hide_render is True
    assert scene.method_calls("set_pose_position") == [
        ("set_pose_position", "Armature", True),
        ("set_pose_position", "Armature", False),
    ]
    order = scene.call_order()
    assert order.index("hide_from_render") < order.index("configure_bake")
    assert order.index("set_pose_position") < order.index("bake_map")
    assert order[-1] == "set_pose_position"  # restored after the last bake
    assert scene.objects["Armature"].pose_rest is False


def test_rest_pose_is_restored_even_when_a_bake_fails(tmp_path: Path) -> None:
    scene = scene_for_bake(source=full_source())
    scene.bake_errors["normal"] = RuntimeError("boom")
    failure(scene, tmp_path, maps=["basecolor", "normal"])
    assert scene.call_order()[-1] == "set_pose_position"
    assert scene.objects["Armature"].pose_rest is False


def test_cycles_device_and_bake_settings_from_config(tmp_path: Path) -> None:
    scene = scene_for_bake(source=full_source())
    _, warnings = run(
        scene,
        tmp_path,
        env={CYCLES_DEVICE_ENV: "gpu"},
        maps=["roughness"],  # native bake type → cfg.samples
        margin_px=8,
        cage_extrusion_ratio=0.02,
        max_ray_distance_ratio=0.1,
    )
    assert warnings == []
    assert scene.render == {"engine": "CYCLES", "device": "GPU", "samples": 16}
    assert scene.bake_settings == {
        "margin_px": 8,
        "cage_extrusion": pytest.approx(0.04),  # ratio x bbox_diagonal (2.0)
        "max_ray_distance": pytest.approx(0.2),
    }


def test_unknown_cycles_device_falls_back_to_cpu_with_warning(tmp_path: Path) -> None:
    scene = scene_for_bake(source=full_source())
    _, warnings = run(scene, tmp_path, env={CYCLES_DEVICE_ENV: "TPU"}, maps=["basecolor"])
    assert scene.render["device"] == "CPU"
    assert warnings == [f"{CYCLES_DEVICE_ENV}='TPU' is not CPU or GPU; using CPU"]


def test_multiple_skinned_meshes_bake_into_one_image(tmp_path: Path) -> None:
    scene = scene_for_bake(source=full_source(), targets=2)
    run(scene, tmp_path, maps=["basecolor"])
    assert scene.baked[0]["targets"] == ["Cube", "Cube.001"]
    assert scene.method_calls("merge_material_slots") == []  # already one shared material


def test_target_with_several_materials_is_merged_first(tmp_path: Path) -> None:
    scene = scene_for_bake(source=full_source())
    scene.objects["Cube"].materials = ["a_mat", "b_mat"]
    _, warnings = run(scene, tmp_path, maps=["basecolor"])
    assert scene.method_calls("merge_material_slots") == [
        ("merge_material_slots", ("Cube",), "a_mat")
    ]
    assert scene.material_slots("Cube") == ["a_mat"]
    assert warnings == ["target materials merged into 'a_mat' before baking"]


def test_target_without_material_gets_a_fallback_name(tmp_path: Path) -> None:
    scene = scene_for_bake(source=full_source())
    scene.objects["Cube"].materials = []
    run(scene, tmp_path, maps=["basecolor"])
    assert scene.material_slots("Cube") == ["Cube_mat"]


def test_source_objects_are_the_meshes_added_by_the_import(tmp_path: Path) -> None:
    scene = scene_for_bake(source=full_source(), source_name="tripo_mesh")
    scene.glb_objects["tripo_extra"] = full_source()
    run(scene, tmp_path, maps=["basecolor"])
    assert scene.baked[0]["sources"] == ["tripo_extra", "tripo_mesh"]
    assert scene.baked[0]["targets"] == ["Cube"]


# --------------------------------------------------------------------------- errors
def test_no_source_material(tmp_path: Path) -> None:
    scene = scene_for_bake(source=source_object(inputs=[], materials=()))
    error = failure(scene, tmp_path)
    assert error.code == bj.E_BJ_BAKE_NO_SOURCE_MATERIAL
    assert error.detail["sources"] == ["result"]
    assert scene.baked == []


def test_bake_failed_wraps_operator_errors(tmp_path: Path) -> None:
    scene = scene_for_bake(source=full_source())
    scene.bake_errors["roughness"] = RuntimeError("No active image found in material")
    error = failure(scene, tmp_path, maps=["basecolor", "roughness", "normal"])
    assert error.code == bj.E_BJ_BAKE_FAILED
    assert error.detail == {"map": "roughness", "bake_type": "ROUGHNESS"}
    assert "No active image" in error.message
    assert [b["map_name"] for b in scene.baked] == ["basecolor"]  # stopped at the failure


def test_bake_empty_when_textured_source_bakes_black(tmp_path: Path) -> None:
    scene = scene_for_bake(source=full_source())
    scene.empty_images.add("basecolor")
    error = failure(scene, tmp_path, maps=["basecolor"])
    assert error.code == bj.E_BJ_BAKE_EMPTY
    assert error.detail["map"] == "basecolor"
    assert error.detail["mean"] == 0.0
    assert "hint" in error.detail


def test_import_failure_and_empty_import(tmp_path: Path) -> None:
    scene = scene_for_bake(source=full_source())
    scene.import_error = RuntimeError("not a glb")
    error = failure(scene, tmp_path)
    assert error.code == bj.E_BJ_IMPORT_FAILED
    assert "not a glb" in error.message

    empty = scene_for_bake(source=full_source())
    empty.glb_objects = {}
    error = failure(empty, tmp_path)
    assert error.code == bj.E_BJ_IMPORT_FAILED
    assert "no mesh objects" in error.message


def test_no_skinned_mesh_in_prepared_blend(tmp_path: Path) -> None:
    scene = scene_for_bake(source=full_source())
    del scene.objects["Cube"]
    error = failure(scene, tmp_path)
    assert error.code == bj.E_BJ_NO_SKINNED_MESH


def test_without_armature_every_mesh_is_a_target(tmp_path: Path) -> None:
    scene = scene_for_bake(source=full_source())
    del scene.objects["Armature"]
    scene.objects["Cube"] = cube_object(armature=None, materials=("cube_rigged_mat",))
    run(scene, tmp_path, maps=["basecolor"])
    assert scene.baked[0]["targets"] == ["Cube"]
    assert scene.method_calls("hide_from_render") == []


def test_parse_params_rejects_drift() -> None:
    with pytest.raises(JobFailure) as info:
        parse_params({"bogus": 1})
    assert info.value.code == bj.E_BJ_BAD_PARAMS


# --------------------------------------------------------------------------- pure helpers
def test_alignment_math() -> None:
    target = ((0.0, 0.0, 0.0), (2.0, 2.0, 2.0))
    assert alignment(target, target) is None
    fit = alignment(((1.0, 1.0, 1.0), (2.0, 2.0, 2.0)), target)
    assert fit is not None
    scale, translation = fit
    assert scale == pytest.approx(2.0)  # diagonals sqrt(3) -> sqrt(12)
    assert translation == pytest.approx((-2.0, -2.0, -2.0))  # centre 1.5 -> 1.0 at scale 2
    # Inside tolerance → no transform; just outside → transform.
    tiny = ((0.0, 0.0, 0.0), (2.0 * (1 + ALIGN_TOLERANCE / 2), 2.0, 2.0))
    assert alignment(tiny, target) is None
    shifted = ((0.01, 0.0, 0.0), (2.01, 2.0, 2.0))
    assert alignment(shifted, target) is not None
    assert alignment(((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)), target) is None  # degenerate source


@pytest.mark.parametrize(
    ("map_name", "inputs", "expected"),
    [
        ("basecolor", [ShaderInput("Base Color", True)], "linked"),
        ("basecolor", [ShaderInput("Base Color", False, (0.5, 0.5, 0.5, 1.0))], "constant"),
        ("basecolor", [], "absent"),
        ("normal", [ShaderInput("Normal", False)], "absent"),
        ("normal", [ShaderInput("Normal", True)], "linked"),
        ("roughness", [ShaderInput("Roughness", False, (0.5,))], "constant"),
        ("metallic", [ShaderInput("Metallic", False, (0.0,))], "absent"),
        ("metallic", [ShaderInput("Metallic", False, (0.3,))], "constant"),
        ("metallic", [ShaderInput("Metallic", True, (0.0,))], "linked"),
        ("occlusion", [ShaderInput("Occlusion", False, (1.0,))], "absent"),
        ("occlusion", [ShaderInput("Occlusion", True, (1.0,))], "linked"),
        ("emission", [ShaderInput("Emission Color", False, (0.0, 0.0, 0.0, 1.0))], "absent"),
        ("emission", [ShaderInput("Emission Color", False, (0.0, 0.4, 0.0, 1.0))], "constant"),
        (
            "emission",
            [
                ShaderInput("Emission Color", False, (0.0, 0.4, 0.0, 1.0)),
                ShaderInput("Emission Strength", False, (0.0,)),
            ],
            "absent",
        ),
        ("emission", [ShaderInput("Emission Color", True, (0.0, 0.0, 0.0, 1.0))], "linked"),
        (
            "emission",
            [
                ShaderInput("Emission Color", True, (1.0, 1.0, 1.0, 1.0)),
                ShaderInput("Emission Strength", False, (1.0,)),
            ],
            "linked",
        ),
        (  # glTF emissiveTexture with the default black emissiveFactor (importer: strength 0)
            "emission",
            [
                ShaderInput("Emission Color", True, (1.0, 1.0, 1.0, 1.0)),
                ShaderInput("Emission Strength", False, (0.0,)),
            ],
            "absent",
        ),
    ],
)
def test_source_presence(map_name: str, inputs: list[ShaderInput], expected: str) -> None:
    assert source_presence(inputs, map_name) == expected


def test_cycles_device_parsing() -> None:
    warnings: list[str] = []
    assert cycles_device({}, warnings) == "CPU"
    assert cycles_device({CYCLES_DEVICE_ENV: " gpu "}, warnings) == "GPU"
    assert cycles_device({CYCLES_DEVICE_ENV: ""}, warnings) == "CPU"
    assert warnings == []
    assert cycles_device({CYCLES_DEVICE_ENV: "metal"}, warnings) == "CPU"
    assert len(warnings) == 1


# --------------------------------------------------------------------------- registry wrapper
def test_run_builds_scene_lazily_and_returns_json(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    scene = scene_for_bake(source=full_source())
    module = types.ModuleType("grounded_pipeline.blender_jobs.bpy_scene")
    module.BpyScene = lambda: scene  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "grounded_pipeline.blender_jobs.bpy_scene", module)
    monkeypatch.delenv(CYCLES_DEVICE_ENV, raising=False)
    raw = orch.BakeTexturesParams.model_validate(
        bj.to_dict(make_params(tmp_path, maps=["basecolor", "normal"]))
    ).model_dump(mode="json")

    data, warnings = bake_textures.run(raw)

    assert warnings == []
    parsed = orch.BakeTexturesData.model_validate(data)
    assert sorted(parsed.maps) == ["basecolor", "normal"]
    assert parsed.resolution == 2048
    assert scene.render["device"] == "CPU"


def test_registry_exposes_bake_textures() -> None:
    assert JOBS["bake_textures"] is bake_textures.run
