"""``gp tripo balance | task | usage`` against a respx-mocked API."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
import respx

from .conftest import TEST_KEY, TRIPO_BASE, Harness, MakeHarness

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "tripo"


def _fixture(name: str) -> httpx.Response:
    return httpx.Response(200, json=json.loads((FIXTURES / f"{name}.json").read_text()))


@pytest.fixture
def router() -> Iterator[respx.MockRouter]:
    with respx.mock(assert_all_called=False, assert_all_mocked=True) as mock:
        yield mock


@pytest.fixture
def keyed(make_harness: MakeHarness, base_env: dict[str, str]) -> Harness:
    return make_harness(env={**base_env, "TRIPO_API_KEY": TEST_KEY})


def test_balance_without_key_exits_2(harness: Harness) -> None:
    result, document = harness.json("tripo", "balance")
    assert result.exit_code == 2
    assert document["error"]["code"] == "E_USAGE"
    assert "TRIPO_API_KEY" in document["error"]["message"]


def test_balance_with_mocked_client(keyed: Harness, router: respx.MockRouter) -> None:
    route = router.get(f"{TRIPO_BASE}/account/balance").mock(return_value=_fixture("balance"))
    result, document = keyed.json("tripo", "balance")
    assert result.exit_code == 0
    assert route.call_count == 1
    assert (document["balance"], document["frozen"]) == (1240.0, 20.0)
    assert "balance: 1240" in keyed.text


def test_balance_uses_library_base_url_when_found(
    keyed: Harness, router: respx.MockRouter, lib: Path
) -> None:
    toml = lib / "library.toml"
    toml.write_text(
        toml.read_text().replace(
            "max_credits_per_run = 50",
            'max_credits_per_run = 50\nbase_url = "https://example.test/v3"',
        )
    )
    route = router.get("https://example.test/v3/account/balance").mock(
        return_value=_fixture("balance")
    )
    result = keyed.invoke("tripo", "balance", "--library", str(lib))
    assert result.exit_code == 0, keyed.text
    assert route.call_count == 1


def test_balance_auth_error_exits_2(keyed: Harness, router: respx.MockRouter) -> None:
    router.get(f"{TRIPO_BASE}/account/balance").mock(
        return_value=httpx.Response(401, json={"code": 1001, "message": "invalid api key"})
    )
    result, document = keyed.json("tripo", "balance")
    assert result.exit_code == 2
    assert document["error"]["code"] == "E_TRIPO_AUTH"


def test_task_detail(keyed: Harness, router: respx.MockRouter) -> None:
    task_id = "task_01J9ZK4A2B3C4D5E6F7G8H9J0K"
    router.get(f"{TRIPO_BASE}/tasks/{task_id}").mock(return_value=_fixture("task_success"))
    result, document = keyed.json("tripo", "task", task_id)
    assert result.exit_code == 0
    assert document["task"]["task_id"] == task_id
    assert document["task"]["status"] == "success"
    assert task_id in keyed.text
    assert "model_url" in keyed.text


def test_task_running_without_output(keyed: Harness, router: respx.MockRouter) -> None:
    task_id = "task_01J9ZK4A2B3C4D5E6F7G8H9J0K"
    router.get(f"{TRIPO_BASE}/tasks/{task_id}").mock(return_value=_fixture("task_running"))
    result = keyed.invoke("tripo", "task", task_id)
    assert result.exit_code == 0
    assert "45%" in keyed.text


def test_usage(keyed: Harness, router: respx.MockRouter) -> None:
    router.get(f"{TRIPO_BASE}/account/usage").mock(return_value=_fixture("usage"))
    result, document = keyed.json("tripo", "usage")
    assert result.exit_code == 0
    assert document["credits_total"] == 30.0
    assert len(document["entries"]) == 2
    assert "2 task(s), 30 credits" in keyed.text
