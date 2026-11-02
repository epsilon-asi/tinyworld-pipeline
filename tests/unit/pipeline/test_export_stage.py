"""``ExportStage`` (plan 12 §1.1): fingerprint matrix, clip naming, RFC flows, outputs/report."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest

from fakes.fake_blender_runner import (
    FakeBlenderRunner,
    export_rigged_response,
    job_error_response,
    roundtrip_animation_response,
    sample_prepare_data,
    sample_rig,
)
from tinyworld_pipeline.blender.protocol import (
    Bone,
    ExportRiggedParams,
    RigFingerprint,
    RoundtripAnimationParams,
)
from tinyworld_pipeline.domain.errors import (
    AnimNameCollisionError,
    BlenderJobError,
    ProtocolDriftError,
    RigFidelityError,
    StagePrereqError,
)
from tinyworld_pipeline.pipeline import hashing
from tinyworld_pipeline.pipeline.context import ModelContext, PipelineContext
from tinyworld_pipeline.pipeline.stages import export as export_mod
from tinyworld_pipeline.pipeline.stages.export import (
    ANIMATIONS_DIR,
    EXPORT_DIR,
    RIG_FIDELITY_REPORT,
    RIG_FINGERPRINT_OUT,
    ExportReport,
    ExportStage,
    clip_name,
    clip_names,
    companion_files,
    export_outputs,
    read_export_report,
    read_source_rig,
    source_label,
    urp_textures,
)

MODEL = "cube_rigged"
URP = Path("work/textures/urp")


# --------------------------------------------------------------------------- helpers
def prepare_outputs(model: ModelContext, *, rig: RigFingerprint | None = None) -> None:
    work = model.paths.work
    work.mkdir(parents=True, exist_ok=True)
    (work / "prepared.blend").write_bytes(b"BLENDER-v405")
    data = sample_prepare_data(static_glb_sha256="0" * 64)
    if rig is not None:
        data = data.model_copy(update={"rig": rig})
    (work / "prepare_report.json").write_text(data.model_dump_json(indent=2), encoding="utf-8")


def write_urp(model: ModelContext, *names: str) -> None:
    urp = model.paths.folder / URP
    urp.mkdir(parents=True, exist_ok=True)
    for name in names:
        (urp / name).write_bytes(b"\x89PNG" + name.encode())


def with_runner(ctx: PipelineContext, runner: FakeBlenderRunner) -> PipelineContext:
    return dataclasses.replace(ctx, blender=runner)


def with_config(model: ModelContext, **sections: dict[str, Any]) -> ModelContext:
    config = model.config.model_validate({**model.config.model_dump(mode="json"), **sections})
    resolved = model.resolved.model_copy(update={"config": config})
    return ModelContext(resolved=resolved, paths=model.paths, log=model.log)


def happy_runner(**kwargs: Any) -> FakeBlenderRunner:
    return FakeBlenderRunner(
        {
            "export_rigged": export_rigged_response(**kwargs.get("export", {})),
            "roundtrip_animation": roundtrip_animation_response(**kwargs.get("roundtrip", {})),
        }
    )


def extra_bone_rig(actions: list[str]) -> RigFingerprint:
    rig = sample_rig()
    tail = Bone(
        name="Tail",
        parent="Root",
        head=(0.0, 0.0, 0.0),
        tail=(0.0, 0.0, 0.2),
        matrix_local=[[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0], [0, 0, 0, 1.0]],
        transform_path="Armature/Root/Tail",
    )
    bones = [rig.bones[0], tail, *rig.bones[1:]]
    return rig.model_copy(update={"bones": bones, "bone_count": 4, "actions": actions})


@pytest.fixture(autouse=True)
def _clear_hash_caches() -> None:
    hashing.clear_caches()


@pytest.fixture
def prepared(rigged: ModelContext) -> ModelContext:
    prepare_outputs(rigged)
    return rigged


# --------------------------------------------------------------------------- clip naming
@pytest.mark.parametrize(
    ("stem", "prefixes", "expected"),
    [
        ("Ant_Walk", ("ant",), "walk"),
        ("Walk", ("ant",), "walk"),
        ("cube_rigged@walk", ("cube_rigged", "cube_rigged"), "walk"),
        ("ANT-Run Fast!", ("ant",), "run_fast"),
        ("ant_worker_idle", ("ant", "ant_worker"), "idle"),
        ("ant", ("ant",), "ant"),
        ("Ant.Attack.01", ("Ant",), "attack_01"),
        ("@@@", (), "clip"),
        ("Idle", (), "idle"),
    ],
)
def test_clip_name(stem: str, prefixes: tuple[str, ...], expected: str) -> None:
    assert clip_name(stem, prefixes) == expected


def test_clip_names_keeps_order_and_detects_collisions(tmp_path: Path) -> None:
    files = [tmp_path / "Ant_Walk.fbx", tmp_path / "ant@idle.fbx"]
    assert clip_names(files, ("ant",)) == [(files[0], "walk"), (files[1], "idle")]
    with pytest.raises(AnimNameCollisionError) as info:
        clip_names([*files, tmp_path / "WALK.fbx"], ("ant",))
    assert info.value.code == "E_ANIM_NAME_COLLISION"
    assert info.value.detail == {"collisions": {"walk": ["Ant_Walk.fbx", "WALK.fbx"]}}
    assert clip_names([], ("ant",)) == []


def test_companion_files_follow_config_and_sort_by_name(rigged: ModelContext) -> None:
    assert [p.name for p in companion_files(rigged)] == ["cube_rigged@walk.fbx"]
    off = with_config(rigged, export={"roundtrip_companion_animations": False})
    assert companion_files(off) == []


def test_source_label_falls_back_to_the_file_name_outside_the_model_folder(
    rigged: ModelContext, tmp_path: Path
) -> None:
    inside = rigged.animation_fbx[0]
    assert source_label(rigged, inside) == "source/animations/cube_rigged@walk.fbx"
    outside = tmp_path / "elsewhere" / "Shared_Walk.fbx"
    outside.parent.mkdir()
    outside.write_bytes(b"FBX")
    assert source_label(rigged, outside) == "Shared_Walk.fbx"


def test_urp_textures_lists_pngs_only(rigged: ModelContext) -> None:
    assert urp_textures(rigged.paths.folder) == []
    write_urp(rigged, "b.png", "a.PNG", "notes.txt")
    assert [p.name for p in urp_textures(rigged.paths.folder)] == ["a.PNG", "b.png"]


# --------------------------------------------------------------------------- fingerprint
def test_fingerprint_matrix(prepared: ModelContext, ctx: PipelineContext) -> None:
    stage = ExportStage()
    folder = prepared.paths.folder
    base = stage.input_fingerprint(prepared, ctx)
    assert base == stage.input_fingerprint(prepared, ctx)
    assert len(base) == 64
    seen = {base}

    def differs(label: str, model: ModelContext = prepared) -> None:
        fp = stage.input_fingerprint(model, ctx)
        assert fp not in seen, label
        seen.add(fp)

    (prepared.paths.work / "prepared.blend").write_bytes(b"BLENDER-v405-changed")
    differs("blend bytes")
    write_urp(prepared, f"{MODEL}_basecolor.png")
    differs("urp texture added")
    (folder / URP / f"{MODEL}_basecolor.png").write_bytes(b"other")
    differs("urp texture bytes")
    companion = prepared.animation_fbx[0]
    companion.write_bytes(companion.read_bytes() + b"\0")
    differs("companion bytes")
    differs("export section", with_config(prepared, export={"fbx_scale_option": "FBX_SCALE_NONE"}))
    differs("embed textures", with_config(prepared, export={"embed_textures": True}))
    differs(
        "roundtrip off", with_config(prepared, export={"roundtrip_companion_animations": False})
    )
    differs("include_unskinned", with_config(prepared, geometry={"include_unskinned": False}))


def test_fingerprint_sees_companion_renames_and_ignores_unrelated_config(
    prepared: ModelContext, ctx: PipelineContext
) -> None:
    stage = ExportStage()
    base = stage.input_fingerprint(prepared, ctx)
    unrelated = with_config(
        prepared, texture={"quality": "detailed"}, bake={"samples": 64}, unity={"shader": "X"}
    )
    assert stage.input_fingerprint(unrelated, ctx) == base
    companion = prepared.animation_fbx[0]
    renamed = companion.with_name("cube_rigged@run.fbx")
    companion.rename(renamed)
    resolved = prepared.resolved.model_copy(update={"animation_fbx": [renamed]})
    moved = ModelContext(resolved=resolved, paths=prepared.paths, log=prepared.log)
    assert stage.input_fingerprint(moved, ctx) != base


def test_fingerprint_changes_with_blender_jobs_code(
    prepared: ModelContext, ctx: PipelineContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    stage = ExportStage()
    base = stage.input_fingerprint(prepared, ctx)
    monkeypatch.setattr(export_mod, "blender_jobs_code_sha", lambda: "f" * 64)
    assert stage.input_fingerprint(prepared, ctx) != base


def test_fingerprint_requires_prepared_blend(rigged: ModelContext, ctx: PipelineContext) -> None:
    with pytest.raises(StagePrereqError, match=r"prepared\.blend"):
        ExportStage().input_fingerprint(rigged, ctx)


# --------------------------------------------------------------------------- run: happy path
def test_run_exports_roundtrips_and_reports(prepared: ModelContext, ctx: PipelineContext) -> None:
    write_urp(prepared, f"{MODEL}_basecolor.png", f"{MODEL}_normal.png")
    folder = prepared.paths.folder
    stale = folder / EXPORT_DIR / "animations" / "stale.fbx"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"old")
    runner = happy_runner(export={"warnings": ("unskinned: Prop",)})

    outcome = ExportStage().run(prepared, with_runner(ctx, runner))

    assert outcome.status == "succeeded"
    assert outcome.outputs == [
        "work/export/cube_rigged.fbx",
        "work/export/animations/cube_rigged@walk.fbx",
        "work/export/rig_fingerprint_out.json",
        "work/export/rig_fidelity_report.json",
    ]
    for rel in outcome.outputs:
        assert (folder / rel).is_file(), rel
    assert not stale.exists()
    assert outcome.extra == {
        "rig_fidelity": "pass",
        "clips": ["walk"],
        "actions_exported": ["idle"],
    }
    assert outcome.warnings == []

    assert runner.call_count("export_rigged") == 1
    assert runner.call_count("roundtrip_animation") == 1
    export_req, clip_req = runner.requests
    assert (export_req.job, export_req.model_id) == ("export_rigged", MODEL)
    params = ExportRiggedParams.model_validate(export_req.params)
    assert params == ExportStage().export_params(prepared)
    assert params.prepared_blend == prepared.paths.work / "prepared.blend"
    assert params.textures_dir == folder / URP
    assert params.out_fbx == folder / EXPORT_DIR / "cube_rigged.fbx"
    assert params.out_rig_fingerprint == folder / RIG_FINGERPRINT_OUT
    assert params.material_name == "cube_rigged_mat"
    assert params.export == prepared.config.export
    assert params.include_unskinned is True
    clip_params = RoundtripAnimationParams.model_validate(clip_req.params)
    assert clip_params.source_fbx == prepared.animation_fbx[0]
    assert clip_params.out_fbx == folder / ANIMATIONS_DIR / "cube_rigged@walk.fbx"
    assert clip_params.expected_bone_names == ["Root", "Spine", "Head"]
    assert clip_params.export == prepared.config.export
    assert runner.log_paths == [prepared.paths.logs / f"export-{ctx.run_id}.log"] * 2
    assert runner.timeouts == [prepared.config.blender.job_timeout_s] * 2

    report = read_export_report(folder)
    assert report.overall == "pass"
    assert report.fbx == "work/export/cube_rigged.fbx"
    assert report.fbx_sha256 == hashing.file_sha(folder / EXPORT_DIR / "cube_rigged.fbx")
    assert report.actions_exported == ["idle"]
    assert report.tolerances == {
        "position_abs": 1e-4,
        "matrix_abs": 1e-4,
        "allow_extra_actions": True,
    }
    assert report.model.mode == "model"
    assert [c.clip_name for c in report.clips] == ["walk"]
    assert report.clips[0].file == "work/export/animations/cube_rigged@walk.fbx"
    assert report.clips[0].source == "source/animations/cube_rigged@walk.fbx"
    assert report.clips[0].actions_exported == ["walk"]
    assert report.clips[0].report.mode == "animation"
    assert report.clips[0].missing_bones == []
    assert report.clips[0].extra_bones == []
    rig_out = json.loads((folder / RIG_FINGERPRINT_OUT).read_text(encoding="utf-8"))
    assert RigFingerprint.model_validate(rig_out) == sample_rig()
    assert export_outputs(folder) == [folder / rel for rel in outcome.outputs]


def test_run_without_textures_or_companions(prepared: ModelContext, ctx: PipelineContext) -> None:
    runner = happy_runner()
    model = with_config(prepared, export={"roundtrip_companion_animations": False})
    outcome = ExportStage().run(model, with_runner(ctx, runner))
    params = ExportRiggedParams.model_validate(runner.requests[0].params)
    assert params.textures_dir is None
    assert runner.call_count("roundtrip_animation") == 0
    assert outcome.outputs == [
        "work/export/cube_rigged.fbx",
        "work/export/rig_fingerprint_out.json",
        "work/export/rig_fidelity_report.json",
    ]
    assert outcome.extra["clips"] == []
    assert read_export_report(model.paths.folder).clips == []


# --------------------------------------------------------------------------- run: RFC flows
def test_rfc_warnings_pass_with_warnings(prepared: ModelContext, ctx: PipelineContext) -> None:
    runner = happy_runner(roundtrip={"rig": extra_bone_rig(["walk"])})
    outcome = ExportStage().run(prepared, with_runner(ctx, runner))
    assert outcome.extra["rig_fidelity"] == "pass_with_warnings"
    assert outcome.warnings == ["walk: bone_names: extra bones: Tail"]
    report = read_export_report(prepared.paths.folder)
    assert report.overall == "pass_with_warnings"
    assert report.model.overall == "pass"
    assert report.clips[0].report.overall == "pass_with_warnings"
    assert report.clips[0].extra_bones == ["Tail"]


def test_rfc_failure_raises_after_writing_report(
    prepared: ModelContext, ctx: PipelineContext
) -> None:
    runner = happy_runner(export={"rig": sample_rig("Rig")})
    with pytest.raises(RigFidelityError) as info:
        ExportStage().run(prepared, with_runner(ctx, runner))
    assert info.value.code == "E_RIG_FIDELITY"
    assert info.value.detail["failures"][0].startswith("model: armature_object:")
    report = read_export_report(prepared.paths.folder)
    assert report.overall == "fail"
    assert report.model.overall == "fail"
    assert runner.call_count("roundtrip_animation") == 1  # clips still round-tripped


def test_rfc_failure_in_a_clip(prepared: ModelContext, ctx: PipelineContext) -> None:
    missing = sample_rig().model_copy(update={"bones": sample_rig().bones[:2], "bone_count": 2})
    runner = happy_runner(roundtrip={"rig": missing})
    with pytest.raises(RigFidelityError, match="walk: bone_names: missing bones: Head"):
        ExportStage().run(prepared, with_runner(ctx, runner))
    assert read_export_report(prepared.paths.folder).clips[0].missing_bones == ["Head"]


def test_model_fingerprint_mismatch_is_a_failure_not_a_warning(
    prepared: ModelContext, ctx: PipelineContext
) -> None:
    # In model mode even a tiny rest-pose drift fails; in animation mode it only warns.
    moved = sample_rig()
    bone = moved.bones[1].model_copy(update={"head": (0.0, 0.0, 0.51)})
    moved = moved.model_copy(update={"bones": [moved.bones[0], bone, moved.bones[2]]})
    runner = happy_runner(export={"rig": moved})
    with pytest.raises(RigFidelityError, match="head_tail"):
        ExportStage().run(prepared, with_runner(ctx, runner))
    runner = happy_runner(roundtrip={"rig": moved.model_copy(update={"actions": ["walk"]})})
    outcome = ExportStage().run(prepared, with_runner(ctx, runner))
    assert outcome.extra["rig_fidelity"] == "pass_with_warnings"


# --------------------------------------------------------------------------- run: errors
def test_missing_prepare_report_is_prereq_error(
    prepared: ModelContext, ctx: PipelineContext
) -> None:
    (prepared.paths.work / "prepare_report.json").unlink()
    with pytest.raises(StagePrereqError, match=r"prepare_report\.json"):
        ExportStage().run(prepared, with_runner(ctx, happy_runner()))
    (prepared.paths.work / "prepare_report.json").write_text("{}", encoding="utf-8")
    with pytest.raises(StagePrereqError):
        read_source_rig(prepared.paths.folder)


def test_job_error_propagates(prepared: ModelContext, ctx: PipelineContext) -> None:
    runner = FakeBlenderRunner({"export_rigged": job_error_response("E_BJ_EXPORT_FAILED", "boom")})
    with pytest.raises(BlenderJobError) as info:
        ExportStage().run(prepared, with_runner(ctx, runner))
    assert info.value.job_code == "E_BJ_EXPORT_FAILED"


@pytest.mark.parametrize(
    ("export_kwargs", "match"),
    [
        ({"data_overrides": {"bogus": 1}}, "does not match ExportRiggedData"),
        ({"data_overrides": {"fbx_sha256": "0" * 64}}, "does not match the sha"),
        ({"write_outputs": False}, "wrote no FBX"),
    ],
)
def test_export_result_drift(
    prepared: ModelContext, ctx: PipelineContext, export_kwargs: dict[str, Any], match: str
) -> None:
    runner = happy_runner(export=export_kwargs)
    with pytest.raises(ProtocolDriftError, match=match):
        ExportStage().run(prepared, with_runner(ctx, runner))


def test_roundtrip_result_drift(prepared: ModelContext, ctx: PipelineContext) -> None:
    runner = happy_runner(roundtrip={"write_outputs": False})
    with pytest.raises(ProtocolDriftError, match="wrote no FBX"):
        ExportStage().run(prepared, with_runner(ctx, runner))


def test_clip_name_collision_is_detected_before_any_job(
    prepared: ModelContext, ctx: PipelineContext
) -> None:
    extra = prepared.animation_fbx[0].with_name("CUBE_RIGGED_walk.fbx")
    extra.write_bytes(b"FBX")
    resolved = prepared.resolved.model_copy(
        update={"animation_fbx": [*prepared.animation_fbx, extra]}
    )
    model = ModelContext(resolved=resolved, paths=prepared.paths, log=prepared.log)
    runner = happy_runner()
    with pytest.raises(AnimNameCollisionError):
        ExportStage().run(model, with_runner(ctx, runner))
    assert runner.requests == []


def test_read_export_report_missing_is_prereq_error(rigged: ModelContext) -> None:
    with pytest.raises(StagePrereqError, match=r"rig_fidelity_report\.json"):
        read_export_report(rigged.paths.folder)
    with pytest.raises(StagePrereqError):
        export_outputs(rigged.paths.folder)
    (rigged.paths.folder / RIG_FIDELITY_REPORT).parent.mkdir(parents=True)
    (rigged.paths.folder / RIG_FIDELITY_REPORT).write_text("{}", encoding="utf-8")
    with pytest.raises(StagePrereqError):
        read_export_report(rigged.paths.folder)


def test_export_report_round_trips(prepared: ModelContext, ctx: PipelineContext) -> None:
    ExportStage().run(prepared, with_runner(ctx, happy_runner()))
    text = (prepared.paths.folder / RIG_FIDELITY_REPORT).read_text(encoding="utf-8")
    report = ExportReport.model_validate_json(text)
    assert json.loads(text) == json.loads(report.model_dump_json())
    assert text.endswith("\n")
