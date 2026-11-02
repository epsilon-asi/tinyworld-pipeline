"""Shared fixtures for the Tripo client tests: fake clock/sleeper, fixture loader, respx wiring."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
from pydantic import SecretStr

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
GLB_BYTES = b"glTF" + (2).to_bytes(4, "little") + (20).to_bytes(4, "little") + b"\0" * 8


class FakeClock:
    """Monotonic clock advanced only by :class:`FakeSleeper` or explicitly."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def monotonic(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FakeSleeper:
    """Records sleeps and (optionally) advances the fake clock instead of blocking."""

    def __init__(self, clock: FakeClock | None = None) -> None:
        self.calls: list[float] = []
        self._clock = clock

    def sleep(self, seconds: float) -> None:
        self.calls.append(seconds)
        if self._clock is not None:
            self._clock.advance(seconds)


def load_fixture(name: str) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((FIXTURE_DIR / f"{name}.json").read_text(encoding="utf-8"))
    return data


def fixture_response(name: str, status: int = 200) -> httpx.Response:
    return httpx.Response(status, json=load_fixture(name))


@pytest.fixture
def fake_clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def fake_sleeper(fake_clock: FakeClock) -> FakeSleeper:
    return FakeSleeper(fake_clock)


@pytest.fixture
def api_key() -> SecretStr:
    return SecretStr(API_KEY)


MakeClient = Callable[..., TripoClient]


@pytest.fixture
def make_client(
    api_key: SecretStr, fake_clock: FakeClock, fake_sleeper: FakeSleeper
) -> Iterator[MakeClient]:
    """Factory for clients with instant sleeps, zero jitter and no request pacing by default."""
    created: list[TripoClient] = []

    def factory(**overrides: Any) -> TripoClient:
        kwargs: dict[str, Any] = {
            "sleeper": fake_sleeper,
            "clock": fake_clock,
            "rate_limiter": RateLimiter(4, 0.0, clock=fake_clock, sleeper=fake_sleeper),
            "jitter": lambda: 0.0,
        }
        kwargs.update(overrides)
        client = TripoClient(api_key, BASE_URL, **kwargs)
        created.append(client)
        return client

    yield factory
    for client in created:
        client.close()


@pytest.fixture
def router() -> Iterator[respx.MockRouter]:
    with respx.mock(assert_all_called=False, assert_all_mocked=True) as mock:
        yield mock


@dataclass
class HappyPath:
    """Handles on the routes wired by :func:`tripo_happy_path`."""

    router: respx.MockRouter
    upload: respx.Route
    create: respx.Route
    task: respx.Route
    download: respx.Route
    glb_bytes: bytes = field(default=GLB_BYTES)
    task_id: str = TASK_ID
    model_url: str = MODEL_URL


@pytest.fixture
def tripo_happy_path(router: respx.MockRouter) -> HappyPath:
    """upload → create → queued → running → success → download, all from recorded fixtures."""
    upload = router.post(f"{BASE_URL}/files").mock(return_value=fixture_response("upload"))
    create = router.post(f"{BASE_URL}/models/texture").mock(
        return_value=fixture_response("create_task")
    )
    task = router.get(f"{BASE_URL}/tasks/{TASK_ID}").mock(
        side_effect=[
            fixture_response("task_queued"),
            fixture_response("task_running"),
            fixture_response("task_success"),
        ]
    )
    download = router.get(url__eq=MODEL_URL).mock(
        return_value=httpx.Response(
            200, content=GLB_BYTES, headers={"Content-Type": "model/gltf-binary"}
        )
    )
    return HappyPath(router=router, upload=upload, create=create, task=task, download=download)
