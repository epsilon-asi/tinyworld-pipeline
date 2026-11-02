"""Fixtures for the CLI tests: a tmp copy of ``library_minimal`` and a fakes-backed ``gp`` app."""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
import structlog
import typer
from pydantic import SecretStr
from typer.testing import CliRunner, Result

from fakes.fake_blender_runner import FakeBlenderRunner, prepare_model_response
from fakes.fake_texture_provider import FakeTextureProvider
from grounded_pipeline import __version__
from grounded_pipeline.blender.locate import BlenderInfo
from grounded_pipeline.cli.app import build_app
from grounded_pipeline.cli.state import BlenderResolver, CliOutput, TripoClientFactory
from grounded_pipeline.config.layering import CliOverrides
from grounded_pipeline.domain.library_config import BlenderSection, TripoSection
from grounded_pipeline.domain.ports import CancelToken
from grounded_pipeline.observability.console import Console, RecordingConsole
from grounded_pipeline.pipeline.context import PipelineContext
from grounded_pipeline.pipeline.runner import ContextFactory, RunSetup
from grounded_pipeline.pipeline.state_store import JsonStateStore
from grounded_pipeline.pipeline.timing import SystemClock, SystemSleeper
from grounded_pipeline.tripo.client import TripoClient
from grounded_pipeline.tripo.polling import RateLimiter

FIXTURE_LIBRARY = Path(__file__).resolve().parents[2] / "fixtures" / "library_minimal"
FAKE_BLENDER = BlenderInfo(path=Path("/fake/blender"), version=(4, 5, 12))
TEST_KEY = "sk-test-0123456789abcdef"
TRIPO_BASE = "https://openapi.tripo3d.ai/v3"
MODEL_IDS = ("cube_palette", "cube_rigged")


def stub_resolver(info: BlenderInfo = FAKE_BLENDER) -> BlenderResolver:
    """A Blender resolver that never spawns anything."""

    def _resolve(
        explicit: Path | None, env: Mapping[str, str], section: BlenderSection
    ) -> BlenderInfo:
        return info

    return _resolve


def mocked_client_factory(api_key: SecretStr, section: TripoSection) -> TripoClient:
    """Real client (respx mocks the transport) without retries, jitter or pacing."""
    clock, sleeper = SystemClock(), SystemSleeper()
    return TripoClient(
        api_key,
        str(section.base_url),
        sleeper=sleeper,
        clock=clock,
        rate_limiter=RateLimiter(2, 0.0, clock=clock, sleeper=sleeper),
        jitter=lambda: 0.0,
        max_attempts=1,
    )


def fakes_context_factory(
    blender: FakeBlenderRunner, provider: FakeTextureProvider
) -> ContextFactory:
    """The plan 08 §5 injection: fakes + a JSON state store over the run's library."""

    def _factory(setup: RunSetup) -> PipelineContext:
        return PipelineContext(
            library=setup.library,
            options=setup.options,
            run_id=setup.run_id,
            blender=blender,
            provider=provider,
            state=JsonStateStore(setup.library.paths, pipeline_version=__version__),
            budget=setup.budget,
            cancel=CancelToken(),
            clock=SystemClock(),
            sleeper=SystemSleeper(),
            log=structlog.get_logger("gp-test").bind(run_id=setup.run_id),
            pipeline_version=__version__,
            env={},
            cli_overrides=CliOverrides().model_dump(mode="json"),
        )

    return _factory


@dataclass
class Harness:
    """A built app plus handles on the injected fakes."""

    app: typer.Typer
    console: RecordingConsole
    blender: FakeBlenderRunner
    provider: FakeTextureProvider
    env: dict[str, str]
    runner: CliRunner = field(default_factory=CliRunner)

    def invoke(
        self, *args: object, stdin: str | None = None, env: Mapping[str, str] | None = None
    ) -> Result:
        """Run ``gp <args>`` and return the click result."""
        return self.runner.invoke(
            self.app, [str(a) for a in args], input=stdin, env=env, catch_exceptions=False
        )

    def json(self, *args: object, stdin: str | None = None) -> tuple[Result, dict[str, Any]]:
        """Run ``gp --json <args>``; parse and validate the single stdout document."""
        result = self.invoke("--json", *args, stdin=stdin)
        document: dict[str, Any] = json.loads(result.stdout)
        CliOutput.model_validate(document)
        assert document["exit_code"] == result.exit_code
        assert document["ok"] == (result.exit_code == 0)
        return result, document

    @property
    def text(self) -> str:
        """Everything rendered through the recording console so far."""
        return self.console.text


MakeHarness = Callable[..., Harness]


@pytest.fixture
def lib(tmp_path: Path) -> Path:
    """A writable copy of ``tests/fixtures/library_minimal``."""
    dest = tmp_path / "library_minimal"
    shutil.copytree(FIXTURE_LIBRARY, dest)
    return dest


@pytest.fixture
def base_env() -> dict[str, str]:
    """Minimal environment: PATH only (no Tripo key, no GP_* overrides)."""
    return {"PATH": os.environ.get("PATH", "")}


@pytest.fixture
def make_harness(tmp_path: Path, base_env: dict[str, str]) -> MakeHarness:
    """Factory for :class:`Harness` instances with overridable fakes and hooks."""

    def _make(
        *,
        blender: FakeBlenderRunner | None = None,
        provider: FakeTextureProvider | None = None,
        env: dict[str, str] | None = None,
        console: RecordingConsole | None = None,
        resolver: BlenderResolver | None = None,
        tripo_client_factory: TripoClientFactory | None = None,
        cwd: Path | None = None,
        inject_context: bool = True,
    ) -> Harness:
        blender = blender or FakeBlenderRunner({"prepare_model": prepare_model_response()})
        provider = provider or FakeTextureProvider()
        environment = dict(env if env is not None else base_env)
        recording = console or Console.recording()
        app = build_app(
            ctx_factory=fakes_context_factory(blender, provider) if inject_context else None,
            console=recording,
            env=environment,
            tripo_client_factory=tripo_client_factory or mocked_client_factory,
            blender_resolver=resolver or stub_resolver(),
            cwd=cwd or tmp_path,
        )
        return Harness(
            app=app, console=recording, blender=blender, provider=provider, env=environment
        )

    return _make


@pytest.fixture
def harness(make_harness: MakeHarness) -> Harness:
    """Default harness: fake Blender/Tripo, no key, stub Blender resolver."""
    return make_harness()
