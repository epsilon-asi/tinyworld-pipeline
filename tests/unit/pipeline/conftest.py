"""Fixtures and scripted fakes for the ``pipeline/*`` unit tests.

The runner tests use :class:`FakeStage` (a tiny scripted stage) instead of the real stage
implementations so that WP-10 does not depend on the Blender/Tripo fakes of other work packages.
"""

from __future__ import annotations

import shutil
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pytest
import structlog

from tinyworld_pipeline.blender.runner import BlenderRunner
from tinyworld_pipeline.config.layering import CliOverrides
from tinyworld_pipeline.domain.ports import CancelToken, Clock, Sleeper
from tinyworld_pipeline.domain.state import STAGE_ORDER, StageName, StageOutcome
from tinyworld_pipeline.library.discovery import Library
from tinyworld_pipeline.library.resolve import resolve_model
from tinyworld_pipeline.pipeline.context import ModelContext, PipelineContext
from tinyworld_pipeline.pipeline.graph import StageGraph
from tinyworld_pipeline.pipeline.hashing import sha_text
from tinyworld_pipeline.pipeline.options import RunOptions
from tinyworld_pipeline.pipeline.state_store import InMemoryStateStore, StateStore
from tinyworld_pipeline.pipeline.timing import SystemClock, SystemSleeper
from tinyworld_pipeline.texturing.provider import TextureProvider
from tinyworld_pipeline.tripo.budget import CreditBudget

FIXTURE_LIBRARY = Path(__file__).resolve().parents[2] / "fixtures" / "library_minimal"
PIPELINE_VERSION = "0.1.0"
MODEL_IDS = ("cube_palette", "cube_rigged")

StageFn = Callable[[ModelContext, PipelineContext], StageOutcome]
FingerprintFn = Callable[[ModelContext, PipelineContext], str]


class FakeClock:
    """Settable monotonic clock."""

    def __init__(self, start: float = 100.0) -> None:
        self.now = start

    def monotonic(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FakeStage:
    """A scripted stage: constant (versioned) fingerprint, writes ``work/<name>.txt`` by default.

    ``version`` changes the fingerprint; ``run_fn`` / ``fingerprint_fn`` override behaviour;
    ``calls`` records the model ids run (thread-safe).
    """

    def __init__(
        self,
        name: StageName,
        *,
        version: str = "v1",
        run_fn: StageFn | None = None,
        fingerprint_fn: FingerprintFn | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        self.name = name
        self.version = version
        self.run_fn = run_fn
        self.fingerprint_fn = fingerprint_fn
        self.extra = dict(extra or {})
        self.calls: list[str] = []
        self.fingerprint_calls = 0
        self._lock = threading.Lock()

    def input_fingerprint(self, model: ModelContext, ctx: PipelineContext) -> str:
        with self._lock:
            self.fingerprint_calls += 1
        if self.fingerprint_fn is not None:
            return self.fingerprint_fn(model, ctx)
        return sha_text(f"{self.name}:{self.version}")

    def run(self, model: ModelContext, ctx: PipelineContext) -> StageOutcome:
        with self._lock:
            self.calls.append(model.model_id)
        if self.run_fn is not None:
            return self.run_fn(model, ctx)
        return default_run(self.name, model, extra=self.extra)

    @property
    def run_count(self) -> int:
        with self._lock:
            return len(self.calls)


def default_run(
    name: StageName, model: ModelContext, *, extra: dict[str, Any] | None = None
) -> StageOutcome:
    """Write ``work/<name>.txt`` and declare it as the stage's output."""
    out = model.paths.work / f"{name}.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(f"{name} for {model.model_id}\n", encoding="utf-8")
    return StageOutcome(status="succeeded", outputs=[f"work/{name}.txt"], extra=dict(extra or {}))


def fake_graph(**overrides: FakeStage) -> tuple[StageGraph, dict[StageName, FakeStage]]:
    """A full seven-stage graph of :class:`FakeStage`; ``overrides`` replace stages by name."""
    stages: dict[StageName, FakeStage] = {}
    for name in STAGE_ORDER:
        stage = overrides.get(name)
        stages[name] = stage if stage is not None else FakeStage(name)
    return StageGraph(list(stages.values())), stages


def make_context(
    library: Library,
    *,
    state: StateStore | None = None,
    cancel: CancelToken | None = None,
    options: RunOptions | None = None,
    clock: Clock | None = None,
    sleeper: Sleeper | None = None,
    pipeline_version: str = PIPELINE_VERSION,
    env: dict[str, str] | None = None,
    cli: CliOverrides | None = None,
    budget: CreditBudget | None = None,
) -> PipelineContext:
    """A real :class:`PipelineContext` with in-memory state and inert adapter stand-ins."""
    return PipelineContext(
        library=library,
        options=options or RunOptions(),
        run_id="01ARZ3NDEKTSV4RRFFQ69G5FAV",
        blender=cast(BlenderRunner, object()),
        provider=cast(TextureProvider, object()),
        state=state or InMemoryStateStore(pipeline_version),
        budget=budget or CreditBudget(None),
        cancel=cancel or CancelToken(),
        clock=clock or SystemClock(),
        sleeper=sleeper or SystemSleeper(),
        log=structlog.get_logger("test"),
        pipeline_version=pipeline_version,
        env=dict(env or {}),
        cli_overrides=(cli or CliOverrides()).model_dump(mode="json"),
    )


def model_context(library: Library, ctx: PipelineContext, model_id: str) -> ModelContext:
    """Resolve ``model_id`` the way the runner does."""
    resolved, _ = resolve_model(
        library, model_id, env=ctx.env, cli=CliOverrides.model_validate(ctx.cli_overrides)
    )
    return ModelContext(
        resolved=resolved, paths=library.paths.model(model_id), log=ctx.log.bind(model_id=model_id)
    )


# --------------------------------------------------------------------------- fixtures
@pytest.fixture
def fixture_root(tmp_path: Path) -> Path:
    dest = tmp_path / "library_minimal"
    shutil.copytree(FIXTURE_LIBRARY, dest)
    return dest


@pytest.fixture
def library(fixture_root: Path) -> Library:
    return Library.load(fixture_root)


@pytest.fixture
def ctx(library: Library) -> PipelineContext:
    return make_context(library)


@pytest.fixture
def rigged(library: Library, ctx: PipelineContext) -> ModelContext:
    return model_context(library, ctx, "cube_rigged")


@pytest.fixture
def palette(library: Library, ctx: PipelineContext) -> ModelContext:
    return model_context(library, ctx, "cube_palette")
