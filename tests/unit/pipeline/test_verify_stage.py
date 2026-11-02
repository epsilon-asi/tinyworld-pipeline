from __future__ import annotations

import json
from pathlib import Path

import pytest

from fakes.glb_builder import cube, textured_set, write_glb
from tinyworld_pipeline.domain.errors import VerifySelfMismatchError
from tinyworld_pipeline.domain.library_config import VerifySection
from tinyworld_pipeline.gltf.fingerprint import surface_fingerprint
from tinyworld_pipeline.gltf.loader import load_glb
from tinyworld_pipeline.library.discovery import Library
from tinyworld_pipeline.pipeline.context import ModelContext, PipelineContext
from tinyworld_pipeline.pipeline.options import RunOptions
from tinyworld_pipeline.pipeline.stages.verify import (
    PREPARE_REPORT,
    RESULT_GLB,
    STATIC_GLB,
    TASK_JSON,
    VERIFY_REPORT,
    VerifyStage,
    effective_verify_section,
    read_normalisation,
    read_prepare_counts,
)

from .conftest import make_context


def _prepare_report(folder: Path, *, triangles: int | None = None) -> None:
    fp = surface_fingerprint(load_glb(folder / STATIC_GLB), q_pos=1e-4, q_uv=1e-4)
    data = {
        "static_glb_triangles": triangles if triangles is not None else fp.triangle_count,
        "static_glb_unique_positions": fp.unique_position_count,
    }
    (folder / PREPARE_REPORT).write_text(json.dumps(data), encoding="utf-8")


@pytest.fixture
def prepared(rigged: ModelContext) -> ModelContext:
    folder = rigged.paths.folder
    images, material = textured_set()
    write_glb(folder / STATIC_GLB, cube())
    write_glb(folder / RESULT_GLB, cube(images=images, materials=[material]))
    _prepare_report(folder)
    return rigged


def test_verify_extract_route_writes_report(prepared: ModelContext, ctx: PipelineContext) -> None:
    outcome = VerifyStage().run(prepared, ctx)
    assert outcome.outputs == ["work/verify_report.json"]
    assert outcome.extra == {
        "route": "extract",
        "position_match": True,
        "uv_match": True,
        "topology_match": True,
    }
    report = json.loads((prepared.paths.folder / VERIFY_REPORT).read_text(encoding="utf-8"))
    assert report["route"] == "extract"
    assert report["forced"] is False
    assert "source=" in report["reason"]


def test_route_override_forces_bake_and_changes_fingerprint(
    library: Library, prepared: ModelContext, ctx: PipelineContext
) -> None:
    stage = VerifyStage()
    fp_auto = stage.input_fingerprint(prepared, ctx)
    forced = make_context(library, options=RunOptions(route_override="bake"))
    assert effective_verify_section(prepared, forced) == VerifySection(route="bake")
    assert effective_verify_section(prepared, ctx) == prepared.config.verify
    assert stage.input_fingerprint(prepared, forced) != fp_auto
    same = make_context(library, options=RunOptions(route_override="auto"))
    assert stage.input_fingerprint(prepared, same) == fp_auto

    outcome = stage.run(prepared, forced)
    assert outcome.extra["route"] == "bake"
    report = json.loads((prepared.paths.folder / VERIFY_REPORT).read_text(encoding="utf-8"))
    assert report["forced"] is True


def test_fingerprint_tracks_result_glb(prepared: ModelContext, ctx: PipelineContext) -> None:
    stage = VerifyStage()
    before = stage.input_fingerprint(prepared, ctx)
    write_glb(prepared.paths.folder / RESULT_GLB, cube(scale=2.0))
    assert stage.input_fingerprint(prepared, ctx) != before


def test_task_json_normalisation_is_used(prepared: ModelContext, ctx: PipelineContext) -> None:
    folder = prepared.paths.folder
    images, material = textured_set()
    write_glb(folder / RESULT_GLB, cube(images=images, materials=[material], scale=2.0))
    (folder / TASK_JSON).parent.mkdir(parents=True, exist_ok=True)
    (folder / TASK_JSON).write_text(
        json.dumps({"raw_output": {"output": {"normalize_scale": 2.0}}}), encoding="utf-8"
    )
    outcome = VerifyStage().run(prepared, ctx)
    assert outcome.extra["route"] == "extract"
    report = json.loads((folder / VERIFY_REPORT).read_text(encoding="utf-8"))
    assert "source=task" in report["reason"]


def test_self_mismatch_raises(prepared: ModelContext, ctx: PipelineContext) -> None:
    _prepare_report(prepared.paths.folder, triangles=3)
    with pytest.raises(VerifySelfMismatchError):
        VerifyStage().run(prepared, ctx)
    assert not (prepared.paths.folder / VERIFY_REPORT).exists()


def test_missing_helpers_are_tolerated(prepared: ModelContext, ctx: PipelineContext) -> None:
    (prepared.paths.folder / PREPARE_REPORT).unlink()
    assert VerifyStage().run(prepared, ctx).extra["route"] == "extract"


def test_read_helpers_are_tolerant(tmp_path: Path) -> None:
    missing = tmp_path / "nope.json"
    assert read_prepare_counts(missing) == (None, None)
    assert read_normalisation(missing) == (None, None)
    bad = tmp_path / "bad.json"
    bad.write_text("{", encoding="utf-8")
    assert read_prepare_counts(bad) == (None, None)
    bad.write_text("[]", encoding="utf-8")
    assert read_normalisation(bad) == (None, None)
    weird = tmp_path / "weird.json"
    weird.write_text(
        json.dumps({"static_glb_triangles": True, "static_glb_unique_positions": "8"}),
        encoding="utf-8",
    )
    assert read_prepare_counts(weird) == (None, None)
    task = tmp_path / "task.json"
    task.write_text(json.dumps({"raw_output": {"output": None}}), encoding="utf-8")
    assert read_normalisation(task) == (None, None)
    task.write_text(
        json.dumps(
            {"raw_output": {"output": {"normalize_scale": "2", "normalize_translation": [1, 2, 3]}}}
        ),
        encoding="utf-8",
    )
    assert read_normalisation(task) == (None, (1.0, 2.0, 3.0))
    task.write_text(
        json.dumps({"raw_output": {"output": {"normalize_translation": [1, 2]}}}),
        encoding="utf-8",
    )
    assert read_normalisation(task) == (None, None)
