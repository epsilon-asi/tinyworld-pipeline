from __future__ import annotations

import hashlib
import json
from pathlib import Path

import httpx
import pytest
import respx
from pydantic import SecretStr
from structlog.testing import capture_logs

from tinyworld_pipeline.domain.errors import (
    TripoAmbiguousCreateError,
    TripoApiError,
    TripoAuthError,
    TripoDownloadError,
    TripoServerError,
    TripoTransientError,
)
from tinyworld_pipeline.domain.ports import CancelToken
from tinyworld_pipeline.tripo.client import TripoClient
from tinyworld_pipeline.tripo.models import TaskDetail, TexturePrompt, TextureTaskRequest
from tinyworld_pipeline.tripo.polling import RateLimiter

from .conftest import (
    API_KEY,
    BASE_URL,
    TASK_ID,
    FakeClock,
    FakeSleeper,
    HappyPath,
    MakeClient,
    fixture_response,
    load_fixture,
)

REQUEST = TextureTaskRequest(
    input="file_01J9ZK3Y5V6W7X8Y9Z0A1B2C3D",
    model="v3.5-20260815",
    texture_prompt=TexturePrompt(text="weathered bronze"),
    texture_seed=1337,
)


def _canonical_sha(req: TextureTaskRequest) -> str:
    body = req.model_dump(mode="json", exclude_none=True)
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


# ---------------------------------------------------------------------------------- happy path
def test_happy_path_upload_create_wait_download(
    make_client: MakeClient, tripo_happy_path: HappyPath, tmp_path: Path, fake_sleeper: FakeSleeper
) -> None:
    mesh = tmp_path / "static.glb"
    mesh.write_bytes(b"glTF")
    client = make_client()
    seen: list[TaskDetail] = []

    token = client.upload(mesh, presign_above_mb=60)
    created = client.create_texture_task(REQUEST.model_copy(update={"input": token}))
    assert created.task_id == TASK_ID
    with capture_logs() as logs:
        detail = client.wait_for_task(
            created.task_id,
            interval_s=2.0,
            timeout_s=900.0,
            cancel=CancelToken(),
            on_progress=seen.append,
        )
        assert detail.output is not None
        assert detail.output.model_url is not None
        dest = client.download(str(detail.output.model_url), tmp_path / "tripo" / "result.glb")

    assert [d.status for d in seen] == ["queued", "running", "success"]
    assert fake_sleeper.calls == [2.0, 2.0]
    assert dest.read_bytes() == tripo_happy_path.glb_bytes
    assert not (tmp_path / "tripo" / "result.glb.tmp").exists()
    assert detail.credits_consumed == 10.0

    sent = json.loads(tripo_happy_path.create.calls.last.request.read())
    assert sent == {
        "input": token,
        "model": "v3.5-20260815",
        "texture_prompt": {"text": "weathered bronze"},
        "pbr": True,
        "texture_seed": 1337,
        "texture_alignment": "original_image",
        "texture_quality": "standard",
        "bake": True,
        "delight": True,
    }
    assert tripo_happy_path.task.calls.last.request.headers["Authorization"] == f"Bearer {API_KEY}"
    assert "authorization" not in tripo_happy_path.download.calls.last.request.headers

    download_events = [e for e in logs if e["event"] == "tripo.download"]
    assert download_events[0]["sha256"] == hashlib.sha256(tripo_happy_path.glb_bytes).hexdigest()
    assert download_events[0]["bytes"] == len(tripo_happy_path.glb_bytes)
    credit_events = [e for e in logs if e["event"] == "tripo.credits"]
    assert credit_events
    assert credit_events[-1]["credits_consumed"] == 10.0


# ---------------------------------------------------------------------------- envelope + status
def test_envelope_error_on_http_200_raises_api_error(
    make_client: MakeClient, router: respx.MockRouter
) -> None:
    router.get(f"{BASE_URL}/tasks/bogus").mock(return_value=fixture_response("error_2000"))
    with pytest.raises(TripoApiError) as info:
        make_client().get_task("bogus")
    assert info.value.detail["api_code"] == 2000
    assert info.value.detail["request_id"] == "req_01J9ZK5B6C7D8E9F0G1H2J3K4L"
    assert info.value.hint == load_fixture("error_2000")["suggestion"]


@pytest.mark.parametrize("status", [401, 403])
def test_auth_failure_is_not_retried(
    make_client: MakeClient, router: respx.MockRouter, fake_sleeper: FakeSleeper, status: int
) -> None:
    route = router.get(f"{BASE_URL}/account/balance").mock(
        return_value=httpx.Response(status, json={"code": 1001, "message": "invalid api key"})
    )
    with pytest.raises(TripoAuthError) as info:
        make_client().balance()
    assert info.value.exit_code == 2
    assert route.call_count == 1
    assert fake_sleeper.calls == []


def test_429_honours_retry_after_then_succeeds(
    make_client: MakeClient, router: respx.MockRouter, fake_sleeper: FakeSleeper
) -> None:
    route = router.get(f"{BASE_URL}/account/balance").mock(
        side_effect=[
            httpx.Response(429, headers={"Retry-After": "7"}),
            fixture_response("balance"),
        ]
    )
    with capture_logs() as logs:
        data = make_client().balance()
    assert data.balance == 1240.0
    assert route.call_count == 2
    assert fake_sleeper.calls == [7.0]
    retry = next(e for e in logs if e["event"] == "tripo.retry")
    assert retry["error"] == "E_TRIPO_RATE_LIMIT"
    assert retry["attempt"] == 1


def test_5xx_retried_with_exponential_backoff_then_success(
    make_client: MakeClient, router: respx.MockRouter, fake_sleeper: FakeSleeper
) -> None:
    route = router.get(f"{BASE_URL}/tasks/{TASK_ID}").mock(
        side_effect=[
            httpx.Response(502),
            httpx.Response(503),
            fixture_response("task_success"),
        ]
    )
    detail = make_client().get_task(TASK_ID)
    assert detail.is_success
    assert route.call_count == 3
    assert fake_sleeper.calls == [1.0, 2.0]


def test_5xx_exhausts_attempts(
    make_client: MakeClient, router: respx.MockRouter, fake_sleeper: FakeSleeper
) -> None:
    route = router.get(f"{BASE_URL}/tasks/{TASK_ID}").mock(return_value=httpx.Response(500))
    with pytest.raises(TripoServerError):
        make_client().get_task(TASK_ID)
    assert route.call_count == 6
    assert fake_sleeper.calls == [1.0, 2.0, 4.0, 8.0, 16.0]


def test_jitter_is_added_to_backoff(
    make_client: MakeClient, router: respx.MockRouter, fake_sleeper: FakeSleeper
) -> None:
    router.get(f"{BASE_URL}/tasks/{TASK_ID}").mock(
        side_effect=[httpx.Response(500), fixture_response("task_success")]
    )
    make_client(jitter=lambda: 0.25).get_task(TASK_ID)
    assert fake_sleeper.calls == [1.25]


def test_timeouts_become_transient_errors_after_retries(
    make_client: MakeClient, router: respx.MockRouter, fake_sleeper: FakeSleeper
) -> None:
    route = router.get(f"{BASE_URL}/tasks/{TASK_ID}").mock(
        side_effect=httpx.ReadTimeout("read timed out")
    )
    with pytest.raises(TripoTransientError) as info:
        make_client(max_attempts=3).get_task(TASK_ID)
    assert isinstance(info.value.__cause__, httpx.ReadTimeout)
    assert route.call_count == 3
    assert fake_sleeper.calls == [1.0, 2.0]


def test_non_json_and_malformed_envelopes(
    make_client: MakeClient, router: respx.MockRouter
) -> None:
    router.get(f"{BASE_URL}/tasks/a").mock(return_value=httpx.Response(200, text="<html>hi</html>"))
    router.get(f"{BASE_URL}/tasks/b").mock(return_value=httpx.Response(200, json={"code": 0}))
    router.get(f"{BASE_URL}/tasks/c").mock(
        return_value=httpx.Response(200, json={"code": 0, "data": {"task_id": "c"}})
    )
    client = make_client()
    with pytest.raises(TripoApiError, match="non-JSON"):
        client.get_task("a")
    with pytest.raises(TripoApiError, match="no data"):
        client.get_task("b")
    with pytest.raises(TripoApiError, match="unexpected response shape"):
        client.get_task("c")


# ----------------------------------------------------------------------------- create policy
def test_create_read_timeout_is_ambiguous_and_not_retried(
    make_client: MakeClient, router: respx.MockRouter, fake_sleeper: FakeSleeper
) -> None:
    route = router.post(f"{BASE_URL}/models/texture").mock(
        side_effect=httpx.ReadTimeout("read timed out")
    )
    with pytest.raises(TripoAmbiguousCreateError) as info:
        make_client().create_texture_task(REQUEST)
    assert info.value.detail["request_sha256"] == _canonical_sha(REQUEST)
    assert info.value.detail["exception"] == "ReadTimeout"
    assert isinstance(info.value.__cause__, httpx.ReadTimeout)
    assert info.value.hint is not None
    assert "tripo task" in info.value.hint
    assert route.call_count == 1
    assert fake_sleeper.calls == []


def test_create_remote_protocol_error_is_ambiguous(
    make_client: MakeClient, router: respx.MockRouter
) -> None:
    router.post(f"{BASE_URL}/models/texture").mock(
        side_effect=httpx.RemoteProtocolError("connection closed")
    )
    with pytest.raises(TripoAmbiguousCreateError):
        make_client().create_texture_task(REQUEST)


def test_create_connect_error_is_retried_then_succeeds(
    make_client: MakeClient, router: respx.MockRouter, fake_sleeper: FakeSleeper
) -> None:
    route = router.post(f"{BASE_URL}/models/texture").mock(
        side_effect=[
            httpx.ConnectError("refused"),
            httpx.ConnectTimeout("slow"),
            fixture_response("create_task"),
        ]
    )
    with capture_logs() as logs:
        created = make_client().create_texture_task(REQUEST)
    assert created.task_id == TASK_ID
    assert route.call_count == 3
    assert fake_sleeper.calls == [1.0, 2.0]
    event = next(e for e in logs if e["event"] == "tripo.task_created")
    assert event["request_sha256"] == _canonical_sha(REQUEST)


def test_create_5xx_is_not_retried_but_carries_request_sha(
    make_client: MakeClient, router: respx.MockRouter, fake_sleeper: FakeSleeper
) -> None:
    route = router.post(f"{BASE_URL}/models/texture").mock(return_value=httpx.Response(500))
    with pytest.raises(TripoServerError) as info:
        make_client().create_texture_task(REQUEST)
    assert route.call_count == 1
    assert fake_sleeper.calls == []
    assert info.value.detail["request_sha256"] == _canonical_sha(REQUEST)


def test_create_429_is_retried(
    make_client: MakeClient, router: respx.MockRouter, fake_sleeper: FakeSleeper
) -> None:
    router.post(f"{BASE_URL}/models/texture").mock(
        side_effect=[
            httpx.Response(429, headers={"Retry-After": "3"}),
            fixture_response("create_task"),
        ]
    )
    assert make_client().create_texture_task(REQUEST).task_id == TASK_ID
    assert fake_sleeper.calls == [3.0]


def test_create_api_error_envelope(make_client: MakeClient, router: respx.MockRouter) -> None:
    router.post(f"{BASE_URL}/models/texture").mock(return_value=fixture_response("error_1004"))
    with pytest.raises(TripoApiError) as info:
        make_client().create_texture_task(REQUEST)
    assert info.value.detail["api_code"] == 1004
    assert info.value.detail["request_sha256"] == _canonical_sha(REQUEST)


# ------------------------------------------------------------------------------ other endpoints
def test_get_tasks_returns_details_and_missed_ids(
    make_client: MakeClient, router: respx.MockRouter
) -> None:
    route = router.post(f"{BASE_URL}/tasks/list").mock(return_value=fixture_response("tasks_list"))
    batch = make_client().get_tasks(
        [TASK_ID, "task_01J9ZK4A2B3C4D5E6F7G8H9J0L", "task_does_not_exist"]
    )
    assert json.loads(route.calls.last.request.read()) == {
        "task_ids": [TASK_ID, "task_01J9ZK4A2B3C4D5E6F7G8H9J0L", "task_does_not_exist"]
    }
    assert set(batch.tasks) == {TASK_ID, "task_01J9ZK4A2B3C4D5E6F7G8H9J0L"}
    assert batch.tasks[TASK_ID].status == "running"
    assert batch.tasks["task_01J9ZK4A2B3C4D5E6F7G8H9J0L"].credits_consumed == 20.0
    assert batch.missed == ["task_does_not_exist"]


def test_get_tasks_chunks_by_hundred_and_skips_empty(
    make_client: MakeClient, router: respx.MockRouter
) -> None:
    route = router.post(f"{BASE_URL}/tasks/list").mock(
        return_value=httpx.Response(200, json={"code": 0, "data": {"tasks": {}}})
    )
    client = make_client()
    assert client.get_tasks([]).tasks == {}
    assert route.call_count == 0
    ids = [f"task_{i:04d}" for i in range(150)] + ["task_0000"]  # duplicate dropped
    batch = client.get_tasks(ids)
    assert batch.missed == []
    assert route.call_count == 2
    sizes = [len(json.loads(c.request.read())["task_ids"]) for c in route.calls]
    assert sizes == [100, 50]


def test_balance_and_usage(make_client: MakeClient, router: respx.MockRouter) -> None:
    router.get(f"{BASE_URL}/account/balance").mock(return_value=fixture_response("balance"))
    router.get(f"{BASE_URL}/account/usage").mock(return_value=fixture_response("usage"))
    client = make_client()
    with capture_logs() as logs:
        balance = client.balance()
    assert (balance.balance, balance.frozen) == (1240.0, 20.0)
    credit_event = next(e for e in logs if e["event"] == "tripo.credits")
    assert credit_event["balance"] == 1240.0
    usage = client.usage()
    assert [u.credits_consumed for u in usage] == [10.0, 20.0]
    assert usage[0].task_id == TASK_ID


def test_requests_are_paced_by_the_rate_limiter(
    make_client: MakeClient, router: respx.MockRouter, fake_clock: FakeClock
) -> None:
    router.get(f"{BASE_URL}/account/balance").mock(return_value=fixture_response("balance"))
    sleeper = FakeSleeper()  # constant clock → every call after the first waits
    limiter = RateLimiter(2, 0.25, clock=fake_clock, sleeper=sleeper)
    client = make_client(rate_limiter=limiter, sleeper=sleeper)
    client.balance()
    client.balance()
    assert sleeper.calls == [0.25]


# ------------------------------------------------------------------------------------ download
def test_download_refuses_html(
    make_client: MakeClient, router: respx.MockRouter, tmp_path: Path
) -> None:
    url = "https://fixture.local/output/expired.glb"
    router.get(url).mock(
        return_value=httpx.Response(
            200, text="<html>expired</html>", headers={"Content-Type": "text/html"}
        )
    )
    dest = tmp_path / "result.glb"
    with pytest.raises(TripoDownloadError, match="HTML"):
        make_client().download(url, dest)
    assert not dest.exists()
    assert not dest.with_name("result.glb.tmp").exists()


def test_download_content_length_mismatch(
    make_client: MakeClient, router: respx.MockRouter, tmp_path: Path
) -> None:
    url = "https://fixture.local/output/short.glb"
    router.get(url).mock(
        return_value=httpx.Response(
            200,
            content=b"glTF",
            headers={"Content-Type": "model/gltf-binary", "Content-Length": "999"},
        )
    )
    dest = tmp_path / "result.glb"
    with pytest.raises(TripoDownloadError) as info:
        make_client().download(url, dest)
    assert info.value.detail == {"expected": 999, "received": 4}
    assert not dest.exists()
    assert list(tmp_path.iterdir()) == []


def test_download_403_is_a_download_error_not_auth(
    make_client: MakeClient, router: respx.MockRouter, tmp_path: Path, fake_sleeper: FakeSleeper
) -> None:
    url = "https://fixture.local/output/expired.glb"
    route = router.get(url).mock(return_value=httpx.Response(403, text="<Error>Expired</Error>"))
    with pytest.raises(TripoDownloadError) as info:
        make_client().download(url, tmp_path / "r.glb")
    assert info.value.detail["http_status"] == 403
    assert "expire" in (info.value.hint or "")
    assert route.call_count == 1
    assert fake_sleeper.calls == []


def test_download_5xx_and_429_are_retried(
    make_client: MakeClient, router: respx.MockRouter, tmp_path: Path, fake_sleeper: FakeSleeper
) -> None:
    url = "https://fixture.local/output/flaky.glb"
    route = router.get(url).mock(
        side_effect=[
            httpx.Response(503),
            httpx.Response(429, headers={"Retry-After": "2"}),
            httpx.Response(200, content=b"glTFok", headers={"Content-Type": "model/gltf-binary"}),
        ]
    )
    dest = make_client().download(url, tmp_path / "nested" / "dir" / "r.glb")
    assert dest.read_bytes() == b"glTFok"
    assert route.call_count == 3
    assert fake_sleeper.calls == [1.0, 2.0]


def test_download_transport_error_mid_request(
    make_client: MakeClient, router: respx.MockRouter, tmp_path: Path
) -> None:
    url = "https://fixture.local/output/broken.glb"
    router.get(url).mock(side_effect=httpx.ReadError("reset"))
    with pytest.raises(TripoTransientError):
        make_client(max_attempts=2).download(url, tmp_path / "r.glb")


# ------------------------------------------------------------------------------------ redaction
def test_key_never_appears_in_logs_repr_or_errors(
    make_client: MakeClient, router: respx.MockRouter, tmp_path: Path
) -> None:
    router.get(f"{BASE_URL}/tasks/{TASK_ID}").mock(return_value=fixture_response("task_success"))
    router.get(f"{BASE_URL}/account/balance").mock(
        side_effect=httpx.ConnectError(f"refused by proxy using {API_KEY}")
    )
    router.post(f"{BASE_URL}/models/texture").mock(
        return_value=httpx.Response(401, json={"code": 1001, "message": f"bad key {API_KEY}"})
    )
    client = make_client(max_attempts=2, record_dir=tmp_path / "rec")
    with capture_logs() as logs:
        client.get_task(TASK_ID)
        with pytest.raises(TripoTransientError) as transient:
            client.balance()
        with pytest.raises(TripoAuthError) as auth:
            client.create_texture_task(REQUEST)
    assert API_KEY not in repr(client)
    assert "****" in repr(client)
    assert API_KEY not in str(transient.value)
    assert API_KEY not in repr(transient.value)
    assert API_KEY not in str(auth.value)
    assert API_KEY not in repr(logs)
    assert logs, "expected structlog events"
    for recorded in (tmp_path / "rec").iterdir():
        assert API_KEY not in recorded.read_text()


# ------------------------------------------------------------------------------------ recording
def test_recording_hook_writes_redacted_pairs(
    make_client: MakeClient, tripo_happy_path: HappyPath, tmp_path: Path
) -> None:
    record_dir = tmp_path / "rec"
    client = make_client(record_dir=record_dir)
    mesh = tmp_path / "static.glb"
    mesh.write_bytes(b"glTF")
    client.upload(mesh, presign_above_mb=60)
    client.create_texture_task(REQUEST)
    client.get_task(TASK_ID)
    client.download(tripo_happy_path.model_url, tmp_path / "r.glb")

    names = sorted(p.name for p in record_dir.iterdir())
    assert names == [
        "001-POST-files.json",
        "002-POST-models_texture.json",
        f"003-GET-tasks_{TASK_ID}.json",
        f"004-GET-output_{TASK_ID}_model_pbr.glb.json",
    ]
    upload = json.loads((record_dir / names[0]).read_text())
    assert upload["request_headers"]["authorization"] == "Bearer ****wxyz"
    assert upload["request_body"]["size"] > 0  # multipart: size only
    assert upload["response_body"] == load_fixture("upload")

    create = json.loads((record_dir / names[1]).read_text())
    assert create["request_body"]["input"] == REQUEST.input
    assert create["status"] == 200

    download = json.loads((record_dir / names[3]).read_text())
    assert download["response_body"] == {"size": len(tripo_happy_path.glb_bytes)}
    assert "authorization" not in download["request_headers"]


# ------------------------------------------------------------------------------- construction
def test_empty_key_rejected(fake_clock: FakeClock, fake_sleeper: FakeSleeper) -> None:
    limiter = RateLimiter(1, 0, clock=fake_clock, sleeper=fake_sleeper)
    with pytest.raises(TripoAuthError):
        TripoClient(
            SecretStr(""), BASE_URL, sleeper=fake_sleeper, clock=fake_clock, rate_limiter=limiter
        )
    with pytest.raises(ValueError, match="max_attempts"):
        TripoClient(
            SecretStr(API_KEY),
            BASE_URL,
            sleeper=fake_sleeper,
            clock=fake_clock,
            rate_limiter=limiter,
            max_attempts=0,
        )


def test_context_manager_closes_clients(make_client: MakeClient) -> None:
    with make_client() as client:
        assert isinstance(client, TripoClient)
    with pytest.raises(RuntimeError):  # httpx refuses to send on a closed client
        client.balance()
