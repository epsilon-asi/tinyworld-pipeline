"""HTTP/envelope → :mod:`tinyworld_pipeline.domain.errors` mapping and key redaction (plan 10 §3).

Everything that turns an ``httpx`` outcome into a typed :class:`TripoError` lives here so the
client stays a thin transport wrapper and the mapping is testable without a network.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any

import httpx

from tinyworld_pipeline.domain.errors import (
    TripoApiError,
    TripoAuthError,
    TripoError,
    TripoRateLimitedError,
    TripoServerError,
    TripoTransientError,
)
from tinyworld_pipeline.tripo.models import ApiEnvelope

HTTP_UNAUTHORIZED = 401
HTTP_FORBIDDEN = 403
HTTP_TOO_MANY_REQUESTS = 429
HTTP_SERVER_ERROR_MIN = 500
HTTP_CLIENT_ERROR_MIN = 400

_REDACTED_PREFIX = "****"
_REDACTED_VISIBLE_SUFFIX = 4
_MIN_SECRET_LEN = 8  # shorter "secrets" would make the suffix leak most of the key


def redact(text: str, secret: str | None) -> str:
    """Replace every occurrence of ``secret`` in ``text`` with ``****`` + its last four chars.

    Args:
        text: Any string that may contain the API key (log line, exception message, URL).
        secret: The API key in clear text; ``None``/empty disables redaction.

    Returns:
        ``text`` with the key masked. Keys shorter than eight characters are masked entirely.
    """
    if not secret or secret not in text:
        return text
    return text.replace(secret, mask_secret(secret))


def mask_secret(secret: str) -> str:
    """Return the display form of a secret: ``****`` followed by its last four characters."""
    if len(secret) < _MIN_SECRET_LEN:
        return _REDACTED_PREFIX
    return _REDACTED_PREFIX + secret[-_REDACTED_VISIBLE_SUFFIX:]


def parse_retry_after(value: str | None, *, now: datetime | None = None) -> float | None:
    """Parse a ``Retry-After`` header (delta-seconds or HTTP-date) into seconds from now.

    Args:
        value: Raw header value or ``None``.
        now: Reference time for HTTP-date values (defaults to ``datetime.now(UTC)``).

    Returns:
        Non-negative seconds, or ``None`` when the header is missing or unparsable.
    """
    if value is None or not value.strip():
        return None
    raw = value.strip()
    if raw.isdigit():
        return float(raw)
    try:
        when = parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    reference = now or datetime.now(UTC)
    return max(0.0, (when - reference).total_seconds())


def _envelope_from_body(response: httpx.Response) -> ApiEnvelope[Any] | None:
    """Best-effort parse of an error body into an envelope (``None`` when it is not one)."""
    try:
        payload = response.json()
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(payload, dict) or "code" not in payload:
        return None
    try:
        return ApiEnvelope[Any].model_validate(payload)
    except ValueError:
        return None


def envelope_detail(envelope: ApiEnvelope[Any]) -> dict[str, Any]:
    """The ``detail`` dict recorded on :class:`TripoApiError` for a non-zero envelope."""
    return {
        "api_code": envelope.code,
        "message": envelope.message,
        "suggestion": envelope.suggestion,
        "request_id": envelope.request_id,
    }


def map_http_error(response: httpx.Response, *, context: str = "") -> TripoError | None:
    """Map an HTTP response with ``status >= 400`` to the matching :class:`TripoError`.

    Args:
        response: A completed ``httpx`` response (body may be unread for streams).
        context: Short label of the call (``"GET /tasks/x"``) prefixed to the message.

    Returns:
        ``None`` for 2xx/3xx; otherwise the error to raise. 401/403 → :class:`TripoAuthError`,
        429 → :class:`TripoRateLimitedError` (``retry_after_s`` from ``Retry-After``), 5xx →
        :class:`TripoServerError`, any other 4xx → :class:`TripoApiError` (with the envelope's
        ``api_code``/``suggestion``/``request_id`` when the body is an envelope).
    """
    status = response.status_code
    if status < HTTP_CLIENT_ERROR_MIN:
        return None
    prefix = f"{context}: " if context else ""
    detail: dict[str, Any] = {"http_status": status}
    envelope = _envelope_from_body(response)
    if envelope is not None:
        detail.update(envelope_detail(envelope))
    api_message = envelope.message if envelope is not None and envelope.message else None
    if status in (HTTP_UNAUTHORIZED, HTTP_FORBIDDEN):
        return TripoAuthError(
            f"{prefix}Tripo rejected the API key (HTTP {status})",
            detail=detail,
            hint="Check TRIPO_API_KEY (keys start with 'sk-') and that the key is still active.",
        )
    if status == HTTP_TOO_MANY_REQUESTS:
        retry_after = parse_retry_after(response.headers.get("retry-after"))
        detail["retry_after_s"] = retry_after
        return TripoRateLimitedError(
            f"{prefix}Tripo rate limit hit (HTTP 429)", retry_after_s=retry_after, detail=detail
        )
    if status >= HTTP_SERVER_ERROR_MIN:
        return TripoServerError(f"{prefix}Tripo server error (HTTP {status})", detail=detail)
    message = api_message or f"Tripo request failed (HTTP {status})"
    return TripoApiError(
        f"{prefix}{message}",
        detail=detail,
        hint=envelope.suggestion if envelope is not None else None,
    )


def raise_for_envelope(envelope: ApiEnvelope[Any], *, context: str) -> None:
    """Raise :class:`TripoApiError` when ``envelope.code != 0`` (even on HTTP 200).

    Args:
        envelope: Parsed response envelope.
        context: Short label of the call used as the message prefix.

    Raises:
        TripoApiError: With ``detail = {api_code, message, suggestion, request_id}``.
    """
    if envelope.ok:
        return
    message = envelope.message or "unspecified API error"
    raise TripoApiError(
        f"{context}: Tripo API error {envelope.code}: {message}",
        detail=envelope_detail(envelope),
        hint=envelope.suggestion,
    )


def wrap_transport_error(
    exc: httpx.HTTPError, *, context: str, secret: str | None = None
) -> TripoTransientError:
    """Wrap an ``httpx`` transport/timeout exception as :class:`TripoTransientError` (with cause).

    Args:
        exc: The ``httpx`` exception (``TransportError`` or ``TimeoutException``).
        context: Short label of the call used as the message prefix.
        secret: API key to redact from the exception text, if any.

    Returns:
        The wrapped error; its ``detail`` records the original exception class.
    """
    text = redact(str(exc) or exc.__class__.__name__, secret)
    return TripoTransientError(
        f"{context}: network error: {exc.__class__.__name__}: {text}",
        detail={"exception": exc.__class__.__name__},
        cause=exc,
    )


__all__ = [
    "envelope_detail",
    "map_http_error",
    "mask_secret",
    "parse_retry_after",
    "raise_for_envelope",
    "redact",
    "wrap_transport_error",
]
