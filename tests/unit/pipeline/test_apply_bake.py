"""Bake route of stage ``apply`` with the scripted Blender runner (plan 11 §4, §5.2)."""

from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from fakes.fake_blender_runner import (
    FakeBlenderRunner,
    bake_textures_response,
    job_error_response,
)
from fakes.glb_builder import cube, write_glb
from grounded_pipeline.blender.protocol import BakeTexturesParams
from grounded_pipeline.domain.errors import BlenderJobError, ProtocolDriftError, StagePrereqError
from grounded_pipeline.domain.textures import TextureFactors
from grounded_pipeline.pipeline.context import ModelContext, PipelineContext
from grounded_pipeline.pipeline.stages.apply import APPLY_REPORT, URP_DIR, ApplyStage
from grounded_pipeline.pipeline.stages.apply_bake import (
    EMISSIVE_MAP,
    MERGED_MAP,
    PREPARED_BLEND,
    RAW_DIR,
    build_params,
    merge_bake_maps,
    read_bbox_diagonal,
    run_bake_route,
    shape_raw_set,
)
from grounded_pipeline.pipeline.stages.verify import PREPARE_REPORT, RESULT_GLB, VERIFY_REPORT

BBOX_DIAGONAL = 1.7320508


# --------------------------------------------------------------------------- fixtures
@pytest.fixture
def bakeable(rigged: ModelContext) -> ModelContext:
    folder = rigged.paths.folder
    write_glb(folder / RESULT_GLB, cube())
    (folder / PREPARED_BLEND).parent.mkdir(parents=True, exist_ok=True)
    (folder / PREPARED_BLEND).write_bytes(b"BLENDER-v405")
    (folder / PREPARE_REPORT).write_text(
        json.dumps({"bbox_diagonal": BBOX_DIAGONAL, "mesh": {}}), encoding="utf-8"
    )
    (folder / VERIFY_REPORT).write_text(json.dumps({"route": "bake"}), encoding="utf-8")
    return rigged


def with_runner(ctx: PipelineContext, runner: FakeBlenderRunner) -> PipelineContext:
    return replace(ctx, blender=runner)


def pixel(path: Path) -> tuple[int, ...]:
    with Image.open(path) as img:
        rgba = img.convert("RGBA")
        return tuple(int(v) for v in np.asarray(rgba)[0, 0])


def grey_png(path: Path, value: int, size: int = 2) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (size, size), (value, value, value)).save(path)
    return path


# --------------------------------------------------------------------------- params
def test_build_params_uses_work_paths_and_prepare_report(bakeable: ModelContext) -> None:
    params = build_params(bakeable)
    folder = bakeable.paths.folder
    assert params == BakeTexturesParams(
        prepared_blend=folder / "work" / "prepared.blend",
        source_glb=folder / "work" / "tripo" / "result.glb",
        out_dir=folder / "work" / "textures" / "raw",
        bake=bakeable.config.bake,
        bbox_diagonal=BBOX_DIAGONAL,
    )
    assert params.out_dir == folder / RAW_DIR


def test_read_bbox_diagonal_errors(rigged: ModelContext) -> None:
    folder = rigged.paths.folder
    with pytest.raises(StagePrereqError, match="missing or unreadable"):
        read_bbox_diagonal(folder)
    report = folder / PREPARE_REPORT
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("{}", encoding="utf-8")
    with pytest.raises(StagePrereqError, match="no valid bbox_diagonal"):
        read_bbox_diagonal(folder)
    report.write_text(json.dumps({"bbox_diagonal": True}), encoding="utf-8")
    with pytest.raises(StagePrereqError, match="no valid bbox_diagonal"):
        read_bbox_diagonal(folder)
    report.write_text(json.dumps({"bbox_diagonal": 0}), encoding="utf-8")
    with pytest.raises(StagePrereqError):
        read_bbox_diagonal(folder)
    report.write_text(json.dumps({"bbox_diagonal": 3}), encoding="utf-8")
    assert read_bbox_diagonal(folder) == 3.0


# --------------------------------------------------------------------------- route
def test_run_bake_route_runs_job_and_shapes_raw_set(
    bakeable: ModelContext, ctx: PipelineContext
) -> None:
    runner = FakeBlenderRunner({"bake_textures": bake_textures_response()})
    folder = bakeable.paths.folder

    raw = run_bake_route(bakeable, with_runner(ctx, runner))

    assert runner.call_count("bake_textures") == 1
    request = runner.requests[0]
    assert request.job == "bake_textures"
    assert request.model_id == "cube_rigged"
    assert len(request.job_id) == 26  # ULID
    assert BakeTexturesParams.model_validate(request.params) == build_params(bakeable)
    assert runner.timeouts == [bakeable.config.blender.job_timeout_s]
    assert runner.log_paths == [bakeable.paths.logs / f"apply-bake-{ctx.run_id}.log"]

    raw_dir = folder / RAW_DIR
    assert raw.source == "bake"
    assert raw.factors == TextureFactors()
    assert raw.basecolor == raw_dir / "basecolor.png"
    assert raw.normal == raw_dir / "normal.png"
    assert raw.occlusion == raw_dir / "occlusion.png"
    assert raw.emissive == raw_dir / EMISSIVE_MAP
    assert raw.metallic_roughness == raw_dir / MERGED_MAP
    assert not (raw_dir / "emission.png").exists()
    # roughness 128 → G, metallic 64 → B, R = 0, A = 255
    assert pixel(raw_dir / MERGED_MAP) == (0, 128, 64, 255)
    assert sorted(p.name for p in raw_dir.iterdir()) == [
        "basecolor.png",
        "emissive.png",
        "metallic.png",
        "metallic_roughness.png",
        "normal.png",
        "occlusion.png",
        "roughness.png",
    ]


def test_apply_stage_bake_route_end_to_end(bakeable: ModelContext, ctx: PipelineContext) -> None:
    runner = FakeBlenderRunner({"bake_textures": bake_textures_response(size=4)})
    outcome = ApplyStage().run(bakeable, with_runner(ctx, runner))
    folder = bakeable.paths.folder
    assert outcome.extra == {"route": "bake", "fallback_to_bake": False, "resolution": [4, 4]}
    names = sorted(p.name for p in (folder / URP_DIR).iterdir())
    assert names == [
        "cube_rigged_basecolor.png",
        "cube_rigged_emission.png",
        "cube_rigged_metallic_smoothness.png",
        "cube_rigged_normal.png",
        "cube_rigged_occlusion.png",
    ]
    report = json.loads((folder / APPLY_REPORT).read_text(encoding="utf-8"))
    assert report["route"] == "bake"
    assert report["source"] == "bake"
    assert report["raw"]["metallic_roughness"] == "work/textures/raw/metallic_roughness.png"
    assert report["raw"]["emissive"] == "work/textures/raw/emissive.png"
    assert report["factors"]["metallic"] == 1.0
    # URP packing of the merged map: R = metallic (64), A = 255 - roughness (128) = 127
    assert pixel(folder / URP_DIR / "cube_rigged_metallic_smoothness.png") == (64, 0, 0, 127)


def test_missing_roughness_merges_with_fully_rough_neutral(
    bakeable: ModelContext, ctx: PipelineContext
) -> None:
    runner = FakeBlenderRunner({"bake_textures": bake_textures_response(skip=("roughness",))})
    raw = run_bake_route(bakeable, with_runner(ctx, runner))
    raw_dir = bakeable.paths.folder / RAW_DIR
    assert raw.metallic_roughness == raw_dir / MERGED_MAP
    assert pixel(raw_dir / MERGED_MAP) == (0, 255, 64, 255)
    assert not (raw_dir / "_neutral_roughness.png").exists()


def test_missing_metallic_merges_with_dielectric_neutral(
    bakeable: ModelContext, ctx: PipelineContext
) -> None:
    runner = FakeBlenderRunner({"bake_textures": bake_textures_response(skip=("metallic",))})
    run_bake_route(bakeable, with_runner(ctx, runner))
    raw_dir = bakeable.paths.folder / RAW_DIR
    assert pixel(raw_dir / MERGED_MAP) == (0, 128, 0, 255)
    assert not (raw_dir / "_neutral_metallic.png").exists()


def test_neither_metallic_nor_roughness_means_no_merged_map(
    bakeable: ModelContext, ctx: PipelineContext
) -> None:
    response = bake_textures_response(skip=("metallic", "roughness", "emission", "occlusion"))
    runner = FakeBlenderRunner({"bake_textures": response})
    raw = run_bake_route(bakeable, with_runner(ctx, runner))
    assert raw.metallic_roughness is None
    assert raw.emissive is None
    assert raw.occlusion is None
    assert raw.basecolor is not None
    assert raw.normal is not None


def test_job_warnings_are_logged_and_route_still_succeeds(
    bakeable: ModelContext, ctx: PipelineContext
) -> None:
    response = bake_textures_response(skip=("occlusion",), warnings=("source aligned",))
    runner = FakeBlenderRunner({"bake_textures": response})
    raw = run_bake_route(bakeable, with_runner(ctx, runner))
    assert raw.occlusion is None


def test_job_error_passes_through(bakeable: ModelContext, ctx: PipelineContext) -> None:
    runner = FakeBlenderRunner(
        {"bake_textures": job_error_response("E_BJ_BAKE_EMPTY", "all black", map="basecolor")}
    )
    with pytest.raises(BlenderJobError) as info:
        run_bake_route(bakeable, with_runner(ctx, runner))
    assert info.value.job_code == "E_BJ_BAKE_EMPTY"
    assert info.value.detail["map"] == "basecolor"


def test_protocol_drift_in_data_or_missing_files(
    bakeable: ModelContext, ctx: PipelineContext
) -> None:
    drift = FakeBlenderRunner(
        {"bake_textures": bake_textures_response(data_overrides={"resolution": "big"})}
    )
    with pytest.raises(ProtocolDriftError, match="BakeTexturesData"):
        run_bake_route(bakeable, with_runner(ctx, drift))
    shutil.rmtree(bakeable.paths.folder / RAW_DIR)  # the drift run above already wrote PNGs
    missing = FakeBlenderRunner({"bake_textures": bake_textures_response(write_outputs=False)})
    with pytest.raises(ProtocolDriftError, match="do not exist") as info:
        run_bake_route(bakeable, with_runner(ctx, missing))
    assert info.value.detail["missing"] == sorted(bakeable.config.bake.maps)


def test_missing_prepare_report_is_prereq_error(
    bakeable: ModelContext, ctx: PipelineContext
) -> None:
    (bakeable.paths.folder / PREPARE_REPORT).unlink()
    runner = FakeBlenderRunner({"bake_textures": bake_textures_response()})
    with pytest.raises(StagePrereqError):
        run_bake_route(bakeable, with_runner(ctx, runner))
    assert runner.call_count("bake_textures") == 0


# --------------------------------------------------------------------------- merge helpers
def test_merge_bake_maps_exact_pixels(tmp_path: Path) -> None:
    metallic = grey_png(tmp_path / "metallic.png", 200)
    roughness = grey_png(tmp_path / "roughness.png", 30)
    merged = merge_bake_maps(metallic, roughness, tmp_path)
    assert merged == tmp_path / MERGED_MAP
    assert pixel(merged) == (0, 30, 200, 255)
    only_metallic = merge_bake_maps(metallic, None, tmp_path / "m")
    assert only_metallic is not None
    assert pixel(only_metallic) == (0, 255, 200, 255)
    only_roughness = merge_bake_maps(None, roughness, tmp_path / "r")
    assert only_roughness is not None
    assert pixel(only_roughness) == (0, 30, 0, 255)
    assert merge_bake_maps(None, None, tmp_path) is None
    assert not any(p.name.startswith("_neutral") for p in tmp_path.rglob("*.png"))


def test_merge_neutral_matches_the_existing_map_size(tmp_path: Path) -> None:
    roughness = grey_png(tmp_path / "roughness.png", 30, size=4)
    merged = merge_bake_maps(None, roughness, tmp_path)
    assert merged is not None
    with Image.open(merged) as img:
        assert img.size == (4, 4)


def test_shape_raw_set_renames_emission_and_checks_files(tmp_path: Path) -> None:
    basecolor = grey_png(tmp_path / "basecolor.png", 10)
    emission = grey_png(tmp_path / "emission.png", 20)
    raw = shape_raw_set({"basecolor": basecolor, "emission": emission}, tmp_path)
    assert raw.emissive == tmp_path / EMISSIVE_MAP
    assert raw.emissive is not None
    assert raw.emissive.is_file()
    assert not emission.exists()
    assert raw.metallic_roughness is None
    assert raw.basecolor == basecolor
    assert raw.source == "bake"
    with pytest.raises(ProtocolDriftError) as info:
        shape_raw_set({"normal": tmp_path / "nope.png"}, tmp_path)
    assert info.value.detail["missing"] == ["normal"]
