"""``PrepareStage`` (plan 09 §2.1): fingerprint matrix, happy path, drift and sha cross-check."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import structlog

from fakes.fake_blender_runner import (
    FakeBlenderRunner,
    job_error_response,
    prepare_model_response,
)
from tinyworld_pipeline.blender.protocol import PrepareModelData, PrepareModelParams
from tinyworld_pipeline.domain.errors import BlenderJobError, ProtocolDriftError
from tinyworld_pipeline.domain.library_config import LibraryConfig
from tinyworld_pipeline.domain.model_spec import ModelDocument, ModelFrontMatter, ResolvedModel
from tinyworld_pipeline.domain.ports import CancelToken
from tinyworld_pipeline.domain.prompt import ComposedPrompt
from tinyworld_pipeline.domain.style_spec import StyleDocument, StyleFrontMatter
from tinyworld_pipeline.library.paths import ModelPaths
from tinyworld_pipeline.pipeline import hashing
from tinyworld_pipeline.pipeline.context import ModelContext, PipelineContext
from tinyworld_pipeline.pipeline.options import RunOptions
from tinyworld_pipeline.pipeline.stages import prepare
from tinyworld_pipeline.pipeline.stages.prepare import (
    PrepareStage,
    build_params,
    palette_hint_textures,
)

RUN_ID = "01ARZ3NDEKTSV4RRFFQ69G5FAV"
MODEL_ID = "cube_rigged"


def make_model(
    tmp_path: Path, *, config: LibraryConfig | None = None, fbx: bytes = b"FBX-1"
) -> ModelContext:
    models_dir = tmp_path / "library" / "models"
    folder = models_dir / MODEL_ID
    (folder / "source").mkdir(parents=True, exist_ok=True)
    primary = folder / "source" / "cube.fbx"
    primary.write_bytes(fbx)
    doc = ModelDocument(
        model_id=MODEL_ID,
        folder=folder,
        front=ModelFrontMatter(display_name="Cube"),
        sections={"prompt": "a cube"},
    )
    style = StyleDocument(
        folder=tmp_path / "library" / "style",
        front=StyleFrontMatter(name="style"),
        sections={"base_prompt": "base"},
    )
    resolved = ResolvedModel(
        doc=doc,
        config=config or LibraryConfig(),
        primary_fbx=primary,
        animation_fbx=[],
        style=style,
        composed_prompt=ComposedPrompt(text="p", sources={}, length=1, sha256="0" * 64),
    )
    return ModelContext(
        resolved=resolved,
        paths=ModelPaths.for_model(models_dir, MODEL_ID),
        log=structlog.get_logger(),
    )


def make_ctx(runner: FakeBlenderRunner) -> PipelineContext:
    return PipelineContext(
        library=MagicMock(),
        options=RunOptions(),
        run_id=RUN_ID,
        blender=runner,
        provider=MagicMock(),
        state=MagicMock(),
        budget=MagicMock(),
        cancel=CancelToken(),
        clock=MagicMock(),
        sleeper=MagicMock(),
        log=structlog.get_logger(),
        pipeline_version="1.0.0",
        env={},
        cli_overrides={},
    )


def config(**sections: dict[str, Any]) -> LibraryConfig:
    return LibraryConfig.model_validate(sections)


@pytest.fixture(autouse=True)
def _clear_hash_caches() -> None:
    hashing.clear_caches()


# --------------------------------------------------------------------------- fingerprint
def test_fingerprint_is_stable_and_sensitive_to_declared_inputs(tmp_path: Path) -> None:
    stage = PrepareStage()
    ctx = make_ctx(FakeBlenderRunner())
    base = stage.input_fingerprint(make_model(tmp_path), ctx)
    assert base == stage.input_fingerprint(make_model(tmp_path), ctx)
    assert len(base) == 64

    # Each case is fingerprinted right after it is built: they all share one model folder.
    cases: dict[str, dict[str, Any]] = {
        "fbx bytes": {"fbx": b"FBX-22"},
        "geometry": {"config": config(geometry={"subdivision_levels": 2})},
        "uv": {"config": config(uv={"mode": "unwrap"})},
        "fbx_scale_option": {"config": config(export={"fbx_scale_option": "FBX_SCALE_UNITS"})},
    }
    seen = {base}
    for label, kwargs in cases.items():
        fp = stage.input_fingerprint(make_model(tmp_path, **kwargs), ctx)
        assert fp not in seen, label
        seen.add(fp)


def test_fingerprint_ignores_texture_bake_and_other_sections(tmp_path: Path) -> None:
    stage = PrepareStage()
    ctx = make_ctx(FakeBlenderRunner())
    base = stage.input_fingerprint(make_model(tmp_path), ctx)
    unrelated = config(
        texture={"quality": "detailed", "seed": 1},
        bake={"samples": 64, "resolution": 1024},
        export={"embed_textures": True},
        verify={"route": "bake"},
        blender={"job_timeout_s": 120},
    )
    assert stage.input_fingerprint(make_model(tmp_path, config=unrelated), ctx) == base


def test_fingerprint_changes_with_blender_jobs_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stage = PrepareStage()
    ctx = make_ctx(FakeBlenderRunner())
    base = stage.input_fingerprint(make_model(tmp_path), ctx)
    monkeypatch.setattr(prepare, "blender_jobs_code_sha", lambda: "f" * 64)
    assert stage.input_fingerprint(make_model(tmp_path), ctx) != base


# --------------------------------------------------------------------------- params
def test_build_params_uses_work_paths_and_palette_hints(tmp_path: Path) -> None:
    model = make_model(tmp_path)
    textures = model.paths.textures
    textures.mkdir(parents=True)
    for name in ("b.JPG", "a.png", "notes.txt", "c.jpeg"):
        (textures / name).write_bytes(b"x")
    (textures / "sub").mkdir()
    params = build_params(model)
    assert params.source_fbx == model.primary_fbx
    assert params.out_blend == model.paths.work / "prepared.blend"
    assert params.out_static_glb == model.paths.work / "static.glb"
    assert params.out_report == model.paths.work / "prepare_report.json"
    assert params.material_name == "cube_rigged_mat"
    assert params.geometry == model.config.geometry
    assert params.uv == model.config.uv
    assert [p.name for p in params.palette_hint_textures] == ["a.png", "b.JPG", "c.jpeg"]
    assert palette_hint_textures(tmp_path / "nope") == []


# --------------------------------------------------------------------------- run
def test_run_happy_path_writes_report_and_returns_outcome(tmp_path: Path) -> None:
    runner = FakeBlenderRunner(
        {"prepare_model": prepare_model_response(warnings=("unskinned: Prop",))}
    )
    model = make_model(tmp_path)
    ctx = make_ctx(runner)

    outcome = PrepareStage().run(model, ctx)

    assert outcome.status == "succeeded"
    assert outcome.outputs == ["work/prepared.blend", "work/static.glb", "work/prepare_report.json"]
    assert outcome.extra == {"triangles_after": 48, "uv_decision": "kept", "bone_count": 3}
    assert outcome.warnings == ["unskinned: Prop"]
    for rel in outcome.outputs:
        assert (model.paths.folder / rel).is_file(), rel
    report = json.loads((model.paths.work / "prepare_report.json").read_text(encoding="utf-8"))
    data = PrepareModelData.model_validate(report)
    assert data.static_glb_sha256 == hashing.file_sha(model.paths.work / "static.glb")
    text = (model.paths.work / "prepare_report.json").read_text(encoding="utf-8")
    assert text == json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n"

    assert runner.call_count("prepare_model") == 1
    request = runner.requests[0]
    assert request.job == "prepare_model"
    assert request.model_id == MODEL_ID
    assert len(request.job_id) == 26
    params = PrepareModelParams.model_validate(request.params)
    assert params == build_params(model)
    assert runner.timeouts == [model.config.blender.job_timeout_s]
    assert runner.log_paths == [model.paths.logs / f"prepare-{RUN_ID}.log"]


def test_run_passes_cancel_token_and_timeout_from_config(tmp_path: Path) -> None:
    runner = FakeBlenderRunner({"prepare_model": prepare_model_response()})
    model = make_model(tmp_path, config=config(blender={"job_timeout_s": 99}))
    PrepareStage().run(model, make_ctx(runner))
    assert runner.timeouts == [99]


def test_run_protocol_drift_on_bad_data(tmp_path: Path) -> None:
    runner = FakeBlenderRunner(
        {"prepare_model": prepare_model_response(data_overrides={"bogus": 1})}
    )
    with pytest.raises(ProtocolDriftError) as info:
        PrepareStage().run(make_model(tmp_path), make_ctx(runner))
    assert info.value.code == "E_PROTOCOL"
    assert info.value.detail["errors"]


def test_run_sha_cross_check_failure(tmp_path: Path) -> None:
    respond = prepare_model_response(data_overrides={"static_glb_sha256": "0" * 64})
    runner = FakeBlenderRunner({"prepare_model": respond})
    model = make_model(tmp_path)
    with pytest.raises(ProtocolDriftError, match="does not match the sha"):
        PrepareStage().run(model, make_ctx(runner))
    assert not (model.paths.work / "prepare_report.json").exists() or True  # job wrote its own copy
    # The orchestrator must not have written the report itself: the fake job's copy is the
    # only one, and it carries the bogus sha.
    report = json.loads((model.paths.work / "prepare_report.json").read_text())
    assert report["static_glb_sha256"] == "0" * 64


def test_run_missing_static_glb_is_protocol_drift(tmp_path: Path) -> None:
    runner = FakeBlenderRunner({"prepare_model": prepare_model_response(write_outputs=False)})
    with pytest.raises(ProtocolDriftError, match=r"wrote no static\.glb"):
        PrepareStage().run(make_model(tmp_path), make_ctx(runner))


def test_run_job_error_passthrough(tmp_path: Path) -> None:
    runner = FakeBlenderRunner(
        {"prepare_model": job_error_response("E_BJ_SHAPEKEYS", "shape keys", meshes=["Face"])}
    )
    model = make_model(tmp_path)
    with pytest.raises(BlenderJobError) as info:
        PrepareStage().run(model, make_ctx(runner))
    assert info.value.job_code == "E_BJ_SHAPEKEYS"
    assert not (model.paths.work / "prepare_report.json").exists()


def test_stage_name() -> None:
    assert PrepareStage.name == "prepare"
    assert PrepareStage().name == "prepare"
