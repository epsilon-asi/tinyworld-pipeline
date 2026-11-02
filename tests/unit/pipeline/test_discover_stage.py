from __future__ import annotations

from pathlib import Path

import pytest

from tinyworld_pipeline.config.layering import CliOverrides
from tinyworld_pipeline.domain.state import StageStatus, downstream_of
from tinyworld_pipeline.library.discovery import Library
from tinyworld_pipeline.library.resolve import discover_fingerprint
from tinyworld_pipeline.pipeline.context import ModelContext, PipelineContext
from tinyworld_pipeline.pipeline.stages.discover import (
    INVALIDATED_BY_VERSION,
    DiscoverStage,
    major_version,
)
from tinyworld_pipeline.pipeline.state_store import InMemoryStateStore

from .conftest import make_context, model_context


@pytest.mark.parametrize(
    ("version", "major"), [("1.4.2", "1"), ("0.1.0", "0"), ("v2.0", "2"), ("abc", "abc")]
)
def test_major_version(version: str, major: str) -> None:
    assert major_version(version) == major


def test_fingerprint_matches_library_helper_and_tracks_inputs(
    library: Library, ctx: PipelineContext, rigged: ModelContext, fixture_root: Path
) -> None:
    stage = DiscoverStage()
    assert stage.name == "discover"
    fp = stage.input_fingerprint(rigged, ctx)
    assert fp == discover_fingerprint(
        library, "cube_rigged", env={}, cli=CliOverrides(), pipeline_version="0.1.0"
    )
    # unrelated file → unchanged
    (fixture_root / "models" / "cube_rigged" / "source" / "cube_texture.png").write_bytes(b"x")
    assert stage.input_fingerprint(rigged, ctx) == fp
    # model.md edit → changed
    md = fixture_root / "models" / "cube_rigged" / "model.md"
    md.write_text(md.read_text(encoding="utf-8") + "\n<!-- note -->\n", encoding="utf-8")
    assert stage.input_fingerprint(rigged, ctx) != fp
    # pipeline version → changed
    other = make_context(library, pipeline_version="0.2.0")
    assert stage.input_fingerprint(rigged, other) != stage.input_fingerprint(rigged, ctx)


def test_run_records_baseline(ctx: PipelineContext, rigged: ModelContext) -> None:
    outcome = DiscoverStage().run(rigged, ctx)
    assert outcome.status == "succeeded"
    assert outcome.outputs == []
    assert outcome.warnings == []
    assert outcome.extra["pipeline_version"] == "0.1.0"
    assert outcome.extra["prompt_sha256"] == rigged.composed_prompt.sha256
    assert "invalidated_downstream" not in outcome.extra
    assert ctx.state.load("cube_rigged").pipeline_version == "0.1.0"


def test_major_version_change_invalidates_downstream(library: Library) -> None:
    store = InMemoryStateStore("1.2.0")
    store.update("cube_rigged", "prepare", status=StageStatus.SUCCEEDED, outputs=[])
    store.update("cube_rigged", "texture", status=StageStatus.SUCCEEDED, extra={"task_id": "t"})
    ctx = make_context(library, state=store, pipeline_version="2.0.0")
    model = model_context(library, ctx, "cube_rigged")

    outcome = DiscoverStage().run(model, ctx)

    state = store.load("cube_rigged")
    assert state.pipeline_version == "2.0.0"
    for name in downstream_of("discover"):
        rec = state.stages[name]
        assert rec.status == StageStatus.PENDING
        assert rec.extra["invalidated_by"] == INVALIDATED_BY_VERSION
    assert state.stages["texture"].extra["task_id"] == "t"  # merged, not wiped
    assert outcome.extra["invalidated_downstream"] == list(downstream_of("discover"))


def test_minor_version_change_only_stamps(library: Library) -> None:
    store = InMemoryStateStore("0.0.9")
    store.update("cube_palette", "prepare", status=StageStatus.SUCCEEDED, outputs=[])
    ctx = make_context(library, state=store, pipeline_version="0.1.0")
    model = model_context(library, ctx, "cube_palette")

    DiscoverStage().run(model, ctx)

    state = store.load("cube_palette")
    assert state.pipeline_version == "0.1.0"
    assert state.stages["prepare"].status == StageStatus.SUCCEEDED
