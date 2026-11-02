from __future__ import annotations

from datetime import UTC, datetime

import httpx
import pytest

from grounded_pipeline.domain.errors import (
    TripoApiError,
    TripoAuthError,
    TripoRateLimitedError,
    TripoServerError,
    TripoTransientError,
)
from grounded_pipeline.tripo.errors import (
    map_http_error,
    mask_secret,
    parse_retry_after,
    raise_for_envelope,
    redact,
    wrap_transport_error,
)
from grounded_pipeline.tripo.models import ApiEnvelope

from .conftest import API_KEY, load_fixture


def _response(status: int, **kwargs: object) -> httpx.Response:
    response = httpx.Response(status, **kwargs)  # type: ignore[arg-type]
    response.request = httpx.Request("GET", "https://fixture.local/v3/x")
    return response


# ------------------------------------------------------------------------------------- redaction
def test_redact_masks_key_keeping_last_four() -> None:
    text = f"Authorization: Bearer {API_KEY} and again {API_KEY}"
    out = redact(text, API_KEY)
    assert API_KEY not in out
    assert out.count("****wxyz") == 2


def test_redact_noops_without_secret_or_match() -> None:
    assert redact("hello", None) == "hello"
    assert redact("hello", "") == "hello"
    assert redact("hello", "sk-other-key-123") == "hello"


def test_mask_secret_hides_short_secrets_entirely() -> None:
    assert mask_secret("abc") == "****"
    assert mask_secret("sk-0123456789") == "****6789"


# --------------------------------------------------------------------------------- retry-after
def test_parse_retry_after_delta_seconds_and_http_date() -> None:
    assert parse_retry_after("7") == 7.0
    now = datetime(2026, 11, 2, 12, 0, 0, tzinfo=UTC)
    assert parse_retry_after("Mon, 02 Nov 2026 12:00:30 GMT", now=now) == 30.0
    assert parse_retry_after("Mon, 02 Nov 2026 11:59:00 GMT", now=now) == 0.0


def test_parse_retry_after_rejects_garbage() -> None:
    assert parse_retry_after(None) is None
    assert parse_retry_after("   ") is None
    assert parse_retry_after("soon") is None


# ------------------------------------------------------------------------------- map_http_error
@pytest.mark.parametrize("status", [401, 403])
def test_auth_statuses_map_to_auth_error_with_exit_code_2(status: int) -> None:
    err = map_http_error(_response(status, json=load_fixture("error_2000")), context="GET /x")
    assert isinstance(err, TripoAuthError)
    assert err.exit_code == 2
    assert err.detail["http_status"] == status
    assert err.detail["api_code"] == 2000
    assert str(err).startswith("GET /x: ")


def test_429_maps_to_rate_limited_with_retry_after() -> None:
    err = map_http_error(_response(429, headers={"Retry-After": "12"}))
    assert isinstance(err, TripoRateLimitedError)
    assert err.retry_after_s == 12.0
    assert err.detail["retry_after_s"] == 12.0


def test_429_without_retry_after_header() -> None:
    err = map_http_error(_response(429))
    assert isinstance(err, TripoRateLimitedError)
    assert err.retry_after_s is None


@pytest.mark.parametrize("status", [500, 502, 503])
def test_5xx_maps_to_server_error(status: int) -> None:
    err = map_http_error(_response(status, text="<html>bad gateway</html>"))
    assert isinstance(err, TripoServerError)
    assert err.detail == {"http_status": status}


def test_other_4xx_with_envelope_body_becomes_api_error_with_suggestion() -> None:
    err = map_http_error(_response(404, json=load_fixture("error_2000")), context="GET /tasks/x")
    assert isinstance(err, TripoApiError)
    assert err.detail["api_code"] == 2000
    assert err.detail["request_id"] == "req_01J9ZK5B6C7D8E9F0G1H2J3K4L"
    assert err.hint is not None
    assert "does not exist" in str(err)


def test_other_4xx_without_envelope_body() -> None:
    err = map_http_error(_response(400, text="nope"))
    assert isinstance(err, TripoApiError)
    assert err.detail == {"http_status": 400}
    assert "HTTP 400" in str(err)
    # JSON that is not an envelope is ignored too
    err2 = map_http_error(_response(400, json=[1, 2]))
    assert isinstance(err2, TripoApiError)
    err3 = map_http_error(_response(400, json={"code": "not-an-int"}))
    assert isinstance(err3, TripoApiError)


def test_success_statuses_map_to_none() -> None:
    assert map_http_error(_response(200, json={"code": 0})) is None
    assert map_http_error(_response(302)) is None


# ---------------------------------------------------------------------------- raise_for_envelope
def test_raise_for_envelope_passes_code_zero() -> None:
    raise_for_envelope(ApiEnvelope[dict[str, int]](code=0, data={"a": 1}), context="x")


def test_raise_for_envelope_raises_api_error_with_detail() -> None:
    envelope = ApiEnvelope[dict[str, int]].model_validate(load_fixture("error_2000"))
    with pytest.raises(TripoApiError) as info:
        raise_for_envelope(envelope, context="GET /tasks/bogus")
    err = info.value
    assert err.code == "E_TRIPO_API"
    assert err.detail == {
        "api_code": 2000,
        "message": envelope.message,
        "suggestion": envelope.suggestion,
        "request_id": envelope.request_id,
    }
    assert err.hint == envelope.suggestion
    assert "2000" in str(err)


def test_raise_for_envelope_without_message() -> None:
    with pytest.raises(TripoApiError, match="unspecified"):
        raise_for_envelope(ApiEnvelope[int](code=5), context="x")


# ------------------------------------------------------------------------- wrap_transport_error
def test_wrap_transport_error_sets_cause_and_redacts() -> None:
    exc = httpx.ReadTimeout(f"timed out talking with {API_KEY}")
    err = wrap_transport_error(exc, context="GET /x", secret=API_KEY)
    assert isinstance(err, TripoTransientError)
    assert err.__cause__ is exc
    assert API_KEY not in str(err)
    assert "ReadTimeout" in str(err)
    assert err.detail == {"exception": "ReadTimeout"}


def test_wrap_transport_error_with_empty_message() -> None:
    err = wrap_transport_error(httpx.ConnectError(""), context="GET /x")
    assert "ConnectError" in str(err)
