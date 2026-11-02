from __future__ import annotations

from typing import Any

import pytest

from grounded_pipeline.observability.logging import SecretRedactor, mask_secret

from .conftest import SECRET

OTHER = "tok-ABCDEFGHIJKLMNOP"


def _run(redactor: SecretRedactor, **event: Any) -> dict[str, Any]:
    return dict(redactor(None, "info", dict(event)))


def test_mask_secret_keeps_last_four_and_hides_short_values() -> None:
    assert mask_secret(SECRET) == "****wxyz"
    assert mask_secret("abc") == "****"
    assert mask_secret("1234567") == "****"


def test_secret_in_message_is_masked() -> None:
    out = _run(SecretRedactor([SECRET]), event=f"auth failed for {SECRET}")
    assert out["event"] == "auth failed for ****wxyz"
    assert SECRET not in str(out)


def test_secret_in_nested_detail_is_masked() -> None:
    out = _run(
        SecretRedactor([SECRET]),
        event="tripo.request",
        detail={"headers": {"X-Trace": f"a {SECRET} b"}, "items": [SECRET, ("x", SECRET)]},
    )
    assert out["detail"]["headers"]["X-Trace"] == "a ****wxyz b"
    assert out["detail"]["items"][0] == "****wxyz"
    assert out["detail"]["items"][1] == ("x", "****wxyz")
    assert SECRET not in str(out)


def test_secret_in_exception_text_is_masked() -> None:
    traceback = f"Traceback (most recent call last):\n  ...\nRuntimeError: bad key {SECRET}\n"
    out = _run(SecretRedactor([SECRET]), event="stage.fail", exception=traceback)
    assert SECRET not in out["exception"]
    assert "RuntimeError: bad key ****wxyz" in out["exception"]


def test_bearer_header_is_masked_even_for_unregistered_tokens() -> None:
    out = _run(
        SecretRedactor(),
        event="tripo.request",
        headers={"Authorization": f"Bearer {OTHER}"},
        raw=f"Authorization: Bearer {OTHER}; next",
    )
    assert out["headers"]["Authorization"] == "****MNOP"
    assert out["raw"] == "Authorization: Bearer ****MNOP; next"
    assert OTHER not in str(out)


def test_sensitive_keys_are_masked_wholesale() -> None:
    out = _run(SecretRedactor(), event="x", api_key=OTHER, headers={"x-api-key": OTHER})
    assert out["api_key"] == "****MNOP"
    assert out["headers"]["x-api-key"] == "****MNOP"


def test_registered_secret_inside_bearer_is_not_double_masked() -> None:
    out = _run(SecretRedactor([SECRET]), event=f"Authorization: Bearer {SECRET}")
    assert out["event"] == "Authorization: Bearer ****wxyz"


def test_multiple_secrets_including_overlapping_prefixes() -> None:
    short = SECRET[:12]  # a prefix of the long secret, registered too
    redactor = SecretRedactor([short, SECRET, OTHER])
    out = _run(redactor, event=f"{SECRET} | {OTHER} | {short}")
    assert out["event"] == f"{mask_secret(SECRET)} | {mask_secret(OTHER)} | {mask_secret(short)}"
    assert redactor.secrets[0] == SECRET  # longest first


def test_non_string_values_are_untouched() -> None:
    payload: dict[str, Any] = {
        "event": "stage.done",
        "n": 3,
        "ratio": 0.5,
        "ok": True,
        "nothing": None,
        "ids": frozenset({1, 2}),
        "bytes": b"raw",
    }
    out = _run(SecretRedactor([SECRET]), **payload)
    assert out["n"] == 3
    assert out["ratio"] == 0.5
    assert out["ok"] is True
    assert out["nothing"] is None
    assert out["ids"] == {1, 2}
    assert out["bytes"] == b"raw"


@pytest.mark.parametrize("value", ["", SECRET])
def test_add_ignores_empty_and_duplicates(value: str) -> None:
    redactor = SecretRedactor([SECRET])
    redactor.add(value)
    assert redactor.secrets == (SECRET,)
