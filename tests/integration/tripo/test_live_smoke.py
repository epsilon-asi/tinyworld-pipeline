"""Live Tripo smoke test (``tripo_live``; auto-skipped without ``TRIPO_API_KEY``).

Deliberately cheap: it reads the balance and queries a bogus task id. Uploading and texturing a
mesh (which spends credits) is Spike S1, not this file.
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterator

import pytest
from pydantic import SecretStr

from tinyworld_pipeline.domain.errors import TripoApiError
from tinyworld_pipeline.domain.library_config import TRIPO_DEFAULT_BASE_URL
from tinyworld_pipeline.tripo.client import TripoClient
from tinyworld_pipeline.tripo.polling import RateLimiter

pytestmark = pytest.mark.tripo_live


class _RealClock:
    def monotonic(self) -> float:
        return time.monotonic()


class _RealSleeper:
    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)


@pytest.fixture
def live_client() -> Iterator[TripoClient]:
    key = os.environ.get("TRIPO_API_KEY", "")
    base_url = os.environ.get("GP_TRIPO__BASE_URL", TRIPO_DEFAULT_BASE_URL)
    clock = _RealClock()
    sleeper = _RealSleeper()
    client = TripoClient(
        SecretStr(key),
        base_url,
        sleeper=sleeper,
        clock=clock,
        rate_limiter=RateLimiter(1, 1.0, clock=clock, sleeper=sleeper),
        timeout_s=30.0,
    )
    yield client
    client.close()


def test_balance_is_readable(live_client: TripoClient) -> None:
    balance = live_client.balance()
    assert balance.balance >= 0
    assert balance.frozen >= 0


def test_bogus_task_id_is_an_api_error(live_client: TripoClient) -> None:
    with pytest.raises(TripoApiError) as info:
        live_client.get_task("task_does_not_exist_0000000000")
    assert info.value.detail.get("api_code") not in (None, 0)
