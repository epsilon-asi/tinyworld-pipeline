"""Fixtures for the texturing tests: a tiny on-disk library, resolved models, respx-backed client.

The clock/sleeper/client/router helpers mirror ``tests/unit/tripo/conftest.py`` (conftest modules
are not importable across packages, so the small pieces are copied here).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
from pydantic import SecretStr

from fakes.glb_builder import cube, encode_image, solid_image, write_glb
from grounded_pipeline.domain.library_config import LibraryConfig, TextureSection, TripoSection
from grounded_pipeline.domain.model_spec import (
    ModelDocument,
    ModelFrontMatter,
    ResolvedModel,
    TextureOverrides,
)
from grounded_pipeline.domain.prompt import ComposedPrompt
from grounded_pipeline.domain.style_spec import StyleDocument, StyleFrontMatter
from grounded_pipeline.tripo.client import TripoClient
from grounded_pipeline.tripo.polling import RateLimiter

FIXTURE_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "tripo"
BASE_URL = "https://fixture.local/v3"
API_KEY = "sk-test-0123456789abcdefghijklmnopqrstuvwxyz"
TASK_ID = "task_01J9ZK4A2B3C4D5E6F7G8H9J0K"
MODEL_URL = (
    "https://fixture.local/output/task_01J9ZK4A2B3C4D5E6F7G8H9J0K/model_pbr.glb"
    "?Expires=1762085400&Signature=redacted"
)
PREVIEW_URL = (
    "https://fixture.local/output/task_01J9ZK4A2B3C4D5E6F7G8H9J0K/preview.png"
    "?Expires=1762085400&Signature=redacted"
)
GLB_BYTES = b"glTF" + (2).to_bytes(4, "little") + (20).to_bytes(4, "little") + b"\0" * 8
PNG_BYTES = encode_image(solid_image((1, 2, 3, 255)))

DEFAULT_PROMPT = "a red ant, stylised, clean albedo"
MODEL_ID = "ant"


# --------------------------------------------------------------------------- clock / sleeper
class FakeClock:
    """Monotonic clock advanced only by :class:`FakeSleeper` or explicitly."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def monotonic(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FakeSleeper:
    """Records sleeps and advances the fake clock instead of blocking."""

    def __init__(self, clock: FakeClock | None = None) -> None:
        self.calls: list[float] = []
        self._clock = clock

    def sleep(self, seconds: float) -> None:
        self.calls.append(seconds)
        if self._clock is not None:
            self._clock.advance(seconds)


# --------------------------------------------------------------------------- fixtures on disk
def load_fixture(name: str) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((FIXTURE_DIR / f"{name}.json").read_text(encoding="utf-8"))
    return data


def fixture_response(name: str, status: int = 200, **patch: Any) -> httpx.Response:
    """A recorded fixture as a response; ``patch`` overrides keys inside ``data``."""
    payload = load_fixture(name)
    if patch:
        payload["data"] = {**payload["data"], **patch}
    return httpx.Response(status, json=payload)


def write_png(path: Path, rgba: tuple[int, int, int, int] = (200, 10, 10, 255)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encode_image(solid_image(rgba)))
    return path


def prompt_of(text: str) -> ComposedPrompt:
    return ComposedPrompt(
        text=text,
        sources={"base_prompt": "base_prompt", "model_prompt": "prompt"},
        length=len(text),
        sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )


def make_library(root: Path) -> Path:
    """Create ``<root>/style/references/primary.png`` and return ``root``."""
    write_png(root / "style" / "references" / "primary.png", (10, 200, 10, 255))
    return root


def make_resolved(
    root: Path,
    *,
    model_id: str = MODEL_ID,
    prompt: str = DEFAULT_PROMPT,
    texture: dict[str, Any] | None = None,
    tripo: dict[str, Any] | None = None,
    front_texture: TextureOverrides | None = None,
    mesh: bytes | None = None,
) -> ResolvedModel:
    """A :class:`ResolvedModel` for ``<root>/models/<model_id>`` with ``work/static.glb`` written.

    ``texture``/``tripo`` are keyword overrides for the effective config sections;
    ``front_texture`` is what ``model.md`` declared (drives ``model_fields_set`` and guidance
    images).
    """
    folder = root / "models" / model_id
    write_glb(folder / "work" / "static.glb", cube() if mesh is None else mesh)
    config = LibraryConfig(
        texture=TextureSection(**(texture or {})), tripo=TripoSection(**(tripo or {}))
    )
    doc = ModelDocument(
        model_id=model_id,
        folder=folder,
        front=ModelFrontMatter(display_name=model_id.title(), texture=front_texture),
        sections={"prompt": prompt},
    )
    style = StyleDocument(
        folder=root / "style",
        front=StyleFrontMatter(name="test-style"),
        sections={"base_prompt": "stylised low-poly"},
    )
    return ResolvedModel(
        doc=doc,
        config=config,
        primary_fbx=folder / "source" / f"{model_id}.fbx",
        animation_fbx=[],
        style=style,
        composed_prompt=prompt_of(prompt),
    )


@pytest.fixture
def library_root(tmp_path: Path) -> Path:
    return make_library(tmp_path / "library")


# --------------------------------------------------------------------------- client wiring
@pytest.fixture
def fake_clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def fake_sleeper(fake_clock: FakeClock) -> FakeSleeper:
    return FakeSleeper(fake_clock)


MakeClient = Callable[..., TripoClient]


@pytest.fixture
def make_client(fake_clock: FakeClock, fake_sleeper: FakeSleeper) -> Iterator[MakeClient]:
    """Factory for clients with instant sleeps, zero jitter and no request pacing."""
    created: list[TripoClient] = []

    def factory(**overrides: Any) -> TripoClient:
        kwargs: dict[str, Any] = {
            "sleeper": fake_sleeper,
            "clock": fake_clock,
            "rate_limiter": RateLimiter(4, 0.0, clock=fake_clock, sleeper=fake_sleeper),
            "jitter": lambda: 0.0,
        }
        kwargs.update(overrides)
        client = TripoClient(SecretStr(API_KEY), BASE_URL, **kwargs)
        created.append(client)
        return client

    yield factory
    for client in created:
        client.close()


@pytest.fixture
def router() -> Iterator[respx.MockRouter]:
    with respx.mock(assert_all_called=False, assert_all_mocked=True) as mock:
        yield mock
