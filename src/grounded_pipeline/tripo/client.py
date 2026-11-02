"""``TripoClient`` — the one place that talks HTTP to Tripo (plan 10 §3, normative surface).

Retries, envelope checking, key redaction and the fixture recorder live here; polling,
budgeting and the upload flow are in sibling modules and are composed by this class.
"""

from __future__ import annotations

import hashlib
import json
import secrets
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from types import TracebackType
from typing import Any, Self

import httpx
import structlog
from pydantic import SecretStr
from pydantic import ValidationError as PydanticValidationError

from grounded_pipeline.domain.errors import (
    TripoAmbiguousCreateError,
    TripoApiError,
    TripoAuthError,
    TripoDownloadError,
    TripoError,
    TripoRateLimitedError,
    TripoServerError,
    TripoTransientError,
)
from grounded_pipeline.domain.ports import CancelToken, Clock, Sleeper
from grounded_pipeline.tripo import polling
from grounded_pipeline.tripo.errors import (
    HTTP_CLIENT_ERROR_MIN,
    HTTP_SERVER_ERROR_MIN,
    HTTP_TOO_MANY_REQUESTS,
    map_http_error,
    mask_secret,
    raise_for_envelope,
    redact,
    wrap_transport_error,
)
from grounded_pipeline.tripo.models import (
    BalanceData,
    TaskBatchData,
    TaskCreateData,
    TaskDetail,
    TextureTaskRequest,
    UsageEntry,
    envelope_of,
)
from grounded_pipeline.tripo.polling import RateLimiter
from grounded_pipeline.tripo.upload import MultipartFile, upload_file

EVENT_REQUEST = "tripo.request"
EVENT_RESPONSE = "tripo.response"
EVENT_RETRY = "tripo.retry"
EVENT_CREDITS = "tripo.credits"
EVENT_DOWNLOAD = "tripo.download"

DEFAULT_TIMEOUT_S = 60.0
CONNECT_TIMEOUT_S = 10.0
DEFAULT_MAX_ATTEMPTS = 6
BACKOFF_BASE_S = 1.0
BACKOFF_CAP_S = 60.0
CHUNK_SIZE = 1 << 20
BATCH_QUERY_LIMIT = 100  # POST /tasks/list accepts at most 100 ids (verified 2026-11-02)
USER_AGENT = "grounded-pipeline"

_HTML = "text/html"
_JSON = "application/json"
_RETRYABLE: tuple[type[TripoError], ...] = (
    TripoTransientError,
    TripoServerError,
    TripoRateLimitedError,
)
# Task creation is not idempotent: only errors raised provably before the request left the
# socket (connection refused / connect timeout) or an explicit 429 rejection are retried.
_RETRYABLE_CREATE: tuple[type[TripoError], ...] = (TripoTransientError, TripoRateLimitedError)
_PRE_SEND_ERRORS: tuple[type[httpx.HTTPError], ...] = (httpx.ConnectError, httpx.ConnectTimeout)

ErrorMapper = Callable[[httpx.Response, str], TripoError | None]


def _default_jitter() -> float:
    return secrets.SystemRandom().uniform(0.0, 1.0)


def _iter_file(path: Path) -> Iterator[bytes]:
    with path.open("rb") as fh:
        while chunk := fh.read(CHUNK_SIZE):
            yield chunk


def _content_length(headers: httpx.Headers) -> int | None:
    raw = headers.get("content-length")
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _map_http_error(response: httpx.Response, context: str) -> TripoError | None:
    return map_http_error(response, context=context)


def _map_download_error(response: httpx.Response, context: str) -> TripoError | None:
    """CDN downloads: 429/5xx retry as usual; any other 4xx is a (non-auth) download failure."""
    status = response.status_code
    if status < HTTP_CLIENT_ERROR_MIN:
        return None
    if status == HTTP_TOO_MANY_REQUESTS or status >= HTTP_SERVER_ERROR_MIN:
        return map_http_error(response, context=context)
    return TripoDownloadError(
        f"{context}: download rejected (HTTP {status})",
        detail={"http_status": status, "host": response.url.host},
        hint="Signed output URLs expire ~5 min after the task detail was fetched; re-fetch it.",
    )


class RequestRecorder:
    """Writes one redacted JSON file per request/response pair (``NNN-<method>-<path>.json``).

    Used once with a real key to produce ``tests/fixtures/tripo/*.json``. Bodies that are not
    JSON (uploads, downloads) are recorded as ``{"size": n}`` only.
    """

    def __init__(self, directory: Path, *, secret: str, base_path: str = "") -> None:
        self._dir = directory
        self._dir.mkdir(parents=True, exist_ok=True)
        self._secret = secret
        self._base_path = base_path.rstrip("/")
        self._lock = threading.Lock()
        self._count = 0

    def record(
        self, request: httpx.Request, response: httpx.Response, *, response_size: int | None = None
    ) -> Path:
        """Persist the pair and return the written path."""
        with self._lock:
            self._count += 1
            index = self._count
        path = request.url.path
        if self._base_path and path.startswith(self._base_path + "/"):
            path = path[len(self._base_path) :]
        path_part = path.strip("/").replace("/", "_") or "root"
        entry = {
            "method": request.method,
            "url": str(request.url),
            "status": response.status_code,
            "request_headers": self._headers(request.headers),
            "request_body": self._request_body(request),
            "response_headers": self._headers(response.headers),
            "response_body": self._response_body(response, response_size),
        }
        text = json.dumps(entry, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n"
        out = self._dir / f"{index:03d}-{request.method}-{path_part}.json"
        out.write_text(redact(text, self._secret), encoding="utf-8")
        return out

    def _headers(self, headers: httpx.Headers) -> dict[str, str]:
        out: dict[str, str] = {}
        for key, value in headers.items():
            if key.lower() == "authorization":
                out[key] = f"Bearer {mask_secret(self._secret)}"
            else:
                out[key] = value
        return out

    @staticmethod
    def _request_body(request: httpx.Request) -> object:
        if request.headers.get("content-type", "").startswith(_JSON):
            try:
                return json.loads(request.content)
            except (httpx.RequestNotRead, json.JSONDecodeError, UnicodeDecodeError):
                pass
        return {"size": _content_length(request.headers)}

    @staticmethod
    def _response_body(response: httpx.Response, size: int | None) -> object:
        if size is not None:
            return {"size": size}
        if _JSON in response.headers.get("content-type", ""):
            try:
                return response.json()
            except (httpx.ResponseNotRead, json.JSONDecodeError, UnicodeDecodeError):
                pass
        try:
            return {"size": len(response.content)}
        except httpx.ResponseNotRead:
            return {"size": None}


class TripoClient:
    """Synchronous Tripo API v3 client (plan 10 §3).

    Args:
        api_key: Bearer key (``sk-…``); never logged or repr'd in clear.
        base_url: ``https://openapi.tripo3d.ai/v3`` or a regional host.
        transport: Optional ``httpx`` transport (tests inject ``MockTransport``/``respx``).
        sleeper: Sleep port used for retry backoff and polling.
        clock: Monotonic clock port.
        rate_limiter: Shared :class:`RateLimiter` enforcing request spacing.
        timeout_s: Read/write/pool timeout; connect timeout is fixed at 10 s.
        record_dir: When set, every request/response pair is written there (redacted).
        max_attempts: Retry budget for idempotent calls (default 6).
        jitter: Returns the random seconds added to each backoff (tests pass ``lambda: 0``).
    """

    def __init__(
        self,
        api_key: SecretStr,
        base_url: str,
        *,
        transport: httpx.BaseTransport | None = None,
        sleeper: Sleeper,
        clock: Clock,
        rate_limiter: RateLimiter,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        record_dir: Path | None = None,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        jitter: Callable[[], float] | None = None,
    ) -> None:
        key = api_key.get_secret_value()
        if not key:
            raise TripoAuthError(
                "Tripo API key is empty", hint="Set TRIPO_API_KEY (or GP_TRIPO__API_KEY)."
            )
        if max_attempts < 1:
            msg = f"max_attempts must be >= 1, got {max_attempts}"
            raise ValueError(msg)
        self._secret = key
        self._base_url = base_url.rstrip("/")
        self._sleeper = sleeper
        self._clock = clock
        self._rate_limiter = rate_limiter
        self._max_attempts = max_attempts
        self._jitter = jitter or _default_jitter
        timeout = httpx.Timeout(timeout_s, connect=CONNECT_TIMEOUT_S)
        self._api = httpx.Client(
            base_url=self._base_url,
            headers={"Authorization": f"Bearer {key}", "User-Agent": USER_AGENT},
            transport=transport,
            timeout=timeout,
            http2=False,
        )
        # Presigned PUTs and CDN downloads must not carry our bearer token.
        self._public = httpx.Client(
            headers={"User-Agent": USER_AGENT},
            transport=transport,
            timeout=timeout,
            http2=False,
            follow_redirects=True,
        )
        self._recorder = (
            RequestRecorder(record_dir, secret=key, base_path=httpx.URL(self._base_url).path)
            if record_dir
            else None
        )
        self._log = structlog.get_logger(__name__).bind(base_url=self._base_url)

    # ------------------------------------------------------------------ lifecycle
    def close(self) -> None:
        """Close both underlying HTTP clients."""
        self._api.close()
        self._public.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"TripoClient(base_url={self._base_url!r}, api_key={mask_secret(self._secret)!r})"

    # ------------------------------------------------------------------ public API
    def upload(self, path: Path, *, presign_above_mb: int) -> str:
        """Upload ``path`` (multipart, or presign above ``presign_above_mb`` MiB) → file token."""
        return upload_file(self, path, presign_above_mb=presign_above_mb)

    def create_texture_task(self, req: TextureTaskRequest) -> TaskCreateData:
        """``POST /models/texture``. Never blindly retried (plan 10 §3, R-18).

        Raises:
            TripoAmbiguousCreateError: The request may have reached Tripo (read timeout, protocol
                error, …); ``detail["request_sha256"]`` identifies the body for manual recovery.
        """
        body = req.model_dump(mode="json", exclude_none=True)
        raw = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
        sha = hashlib.sha256(raw).hexdigest()
        context = "POST /models/texture"

        def op() -> TaskCreateData:
            request = self._api.build_request(
                "POST", "/models/texture", content=raw, headers={"Content-Type": _JSON}
            )
            response = self._send(self._api, request, context=context, create_sha=sha)
            return self._parse(response, TaskCreateData, context=context)

        try:
            data = self._retry(op, label=context, retry_on=_RETRYABLE_CREATE)
        except TripoError as exc:
            exc.detail.setdefault("request_sha256", sha)
            raise
        self._log.info("tripo.task_created", task_id=data.task_id, request_sha256=sha)
        return data

    def get_task(self, task_id: str) -> TaskDetail:
        """``GET /tasks/{task_id}``."""
        detail = self.request_envelope("GET", f"/tasks/{task_id}", TaskDetail)
        if detail.credits_consumed is not None:
            self._log.debug(
                EVENT_CREDITS,
                task_id=task_id,
                status=detail.status,
                credits_consumed=detail.credits_consumed,
            )
        return detail

    def get_tasks(self, task_ids: Sequence[str]) -> TaskBatchData:
        """``POST /tasks/list`` in chunks of 100 → details keyed by id plus the ``missed`` ids."""
        ids = list(dict.fromkeys(task_ids))
        tasks: dict[str, TaskDetail] = {}
        missed: list[str] = []
        for start in range(0, len(ids), BATCH_QUERY_LIMIT):
            chunk = ids[start : start + BATCH_QUERY_LIMIT]
            data = self.request_envelope(
                "POST", "/tasks/list", TaskBatchData, json={"task_ids": chunk}
            )
            tasks.update(data.tasks)
            missed.extend(data.missed or [])
        return TaskBatchData(tasks=tasks, missed=missed)

    def wait_for_task(
        self,
        task_id: str,
        *,
        interval_s: float,
        timeout_s: float,
        cancel: CancelToken,
        on_progress: Callable[[TaskDetail], None] | None = None,
    ) -> TaskDetail:
        """Poll until success; see :func:`grounded_pipeline.tripo.polling.wait_for_task`."""
        return polling.wait_for_task(
            self,
            task_id,
            interval_s=interval_s,
            timeout_s=timeout_s,
            cancel=cancel,
            clock=self._clock,
            sleeper=self._sleeper,
            on_progress=on_progress,
        )

    def download(self, url: str, dest: Path) -> Path:
        """Stream ``url`` to ``dest`` (via ``dest.tmp`` + atomic replace); logs the sha256.

        Raises:
            TripoDownloadError: HTML body, ``Content-Length`` mismatch, or a non-retryable 4xx
                (e.g. 403 on an expired signed URL).
        """
        return self._retry(lambda: self._download_once(url, dest), label=f"GET {dest.name}")

    def balance(self) -> BalanceData:
        """``GET /account/balance``."""
        data = self.request_envelope("GET", "/account/balance", BalanceData)
        self._log.info(EVENT_CREDITS, balance=data.balance, frozen=data.frozen)
        return data

    def usage(self) -> list[UsageEntry]:
        """``GET /account/usage`` (``data`` is a list of entries)."""
        return self.request_envelope("GET", "/account/usage", list[UsageEntry])

    # ------------------------------------------------------------------ low-level calls
    def request_envelope[T](
        self,
        method: str,
        path: str,
        data_type: type[T],
        *,
        json: Mapping[str, Any] | None = None,
        upload: MultipartFile | None = None,
        context: str = "",
    ) -> T:
        """Authenticated idempotent call with retries; returns the envelope's ``data``."""
        label = context or f"{method} {path}"

        def op() -> T:
            if upload is not None:
                with upload.path.open("rb") as fh:
                    files = {upload.field: (upload.filename, fh, upload.content_type)}
                    request = self._api.build_request(method, path, files=files)
                    response = self._send(self._api, request, context=label)
            else:
                request = self._api.build_request(method, path, json=json)
                response = self._send(self._api, request, context=label)
            return self._parse(response, data_type, context=label)

        return self._retry(op, label=label)

    def put_unauthenticated(self, url: str, *, path: Path, content_type: str) -> None:
        """Stream ``path`` to a presigned ``url`` without the bearer token (retried)."""
        size = path.stat().st_size
        label = "PUT presigned"

        def op() -> None:
            request = self._public.build_request(
                "PUT",
                url,
                content=_iter_file(path),
                headers={"Content-Type": content_type, "Content-Length": str(size)},
            )
            self._send(self._public, request, context=label)

        self._retry(op, label=label)

    # ------------------------------------------------------------------ internals
    def _retry[T](
        self,
        op: Callable[[], T],
        *,
        label: str,
        retry_on: tuple[type[TripoError], ...] = _RETRYABLE,
    ) -> T:
        attempt = 0
        while True:
            attempt += 1
            try:
                return op()
            except retry_on as exc:
                if attempt >= self._max_attempts:
                    raise
                delay = self._backoff_delay(attempt, exc)
                self._log.warning(
                    EVENT_RETRY, call=label, attempt=attempt, delay_s=delay, error=exc.code
                )
                self._sleeper.sleep(delay)

    def _backoff_delay(self, attempt: int, exc: TripoError) -> float:
        delay = min(BACKOFF_BASE_S * 2.0 ** (attempt - 1), BACKOFF_CAP_S) + self._jitter()
        if isinstance(exc, TripoRateLimitedError) and exc.retry_after_s is not None:
            delay = max(delay, exc.retry_after_s)
        return delay

    def _send(
        self,
        client: httpx.Client,
        request: httpx.Request,
        *,
        context: str,
        create_sha: str | None = None,
        stream: bool = False,
        error_mapper: ErrorMapper = _map_http_error,
    ) -> httpx.Response:
        """One attempt: pace, send, log, record, map HTTP-level errors."""
        self._rate_limiter.pace()
        started = self._clock.monotonic()
        self._log.debug(
            EVENT_REQUEST,
            method=request.method,
            path=request.url.path,
            size=_content_length(request.headers),
        )
        try:
            response = client.send(request, stream=stream)
        except httpx.HTTPError as exc:
            if create_sha is not None and not isinstance(exc, _PRE_SEND_ERRORS):
                raise TripoAmbiguousCreateError(
                    f"{context}: {exc.__class__.__name__} after the request may have been sent; "
                    "not retrying to avoid double billing",
                    detail={"request_sha256": create_sha, "exception": exc.__class__.__name__},
                    hint="Run `gp tripo task --list` to find a task matching request_sha256 "
                    "before re-running.",
                    cause=exc,
                ) from exc
            raise wrap_transport_error(exc, context=context, secret=self._secret) from exc
        self._log.debug(
            EVENT_RESPONSE,
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            elapsed_s=self._clock.monotonic() - started,
        )
        if response.status_code >= HTTP_CLIENT_ERROR_MIN and stream:
            response.read()
        if not stream and self._recorder is not None:
            self._recorder.record(request, response)
        error = error_mapper(response, context)
        if error is not None:
            response.close()
            raise error
        return response

    def _parse[T](self, response: httpx.Response, data_type: type[T], *, context: str) -> T:
        """Validate the envelope (``code`` checked even on HTTP 200) and return ``data``."""
        try:
            payload = response.json()
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise TripoApiError(
                f"{context}: non-JSON response (HTTP {response.status_code})",
                detail={"http_status": response.status_code},
                cause=exc,
            ) from exc
        try:
            envelope = envelope_of(data_type).model_validate(payload)
        except PydanticValidationError as exc:
            raise TripoApiError(
                f"{context}: unexpected response shape: {redact(str(exc), self._secret)}",
                detail={"http_status": response.status_code},
                cause=exc,
            ) from exc
        raise_for_envelope(envelope, context=context)
        if envelope.data is None:
            raise TripoApiError(
                f"{context}: envelope code 0 but no data", detail={"http_status": 200}
            )
        return envelope.data

    def _download_once(self, url: str, dest: Path) -> Path:
        context = f"GET {dest.name}"
        request = self._public.build_request("GET", url)
        response = self._send(
            self._public, request, context=context, stream=True, error_mapper=_map_download_error
        )
        tmp = dest.with_name(dest.name + ".tmp")
        digest = hashlib.sha256()
        size = 0
        try:
            content_type = response.headers.get("content-type", "").lower()
            if content_type.startswith(_HTML):
                raise TripoDownloadError(
                    f"{context}: server returned HTML instead of a file",
                    detail={"content_type": content_type, "http_status": response.status_code},
                    hint="The signed URL probably expired; re-fetch the task detail.",
                )
            expected = _content_length(response.headers)
            dest.parent.mkdir(parents=True, exist_ok=True)
            with tmp.open("wb") as fh:
                try:
                    for chunk in response.iter_bytes(CHUNK_SIZE):
                        fh.write(chunk)
                        digest.update(chunk)
                        size += len(chunk)
                except httpx.HTTPError as exc:
                    raise wrap_transport_error(exc, context=context, secret=self._secret) from exc
            if expected is not None and expected != size:
                raise TripoDownloadError(
                    f"{context}: Content-Length {expected} but received {size} bytes",
                    detail={"expected": expected, "received": size},
                )
            tmp.replace(dest)
        except Exception:
            tmp.unlink(missing_ok=True)
            raise
        finally:
            response.close()
        if self._recorder is not None:
            self._recorder.record(request, response, response_size=size)
        self._log.info(EVENT_DOWNLOAD, path=str(dest), bytes=size, sha256=digest.hexdigest())
        return dest


__all__ = [
    "BACKOFF_BASE_S",
    "BACKOFF_CAP_S",
    "BATCH_QUERY_LIMIT",
    "CHUNK_SIZE",
    "CONNECT_TIMEOUT_S",
    "DEFAULT_MAX_ATTEMPTS",
    "DEFAULT_TIMEOUT_S",
    "EVENT_CREDITS",
    "EVENT_DOWNLOAD",
    "EVENT_REQUEST",
    "EVENT_RESPONSE",
    "EVENT_RETRY",
    "USER_AGENT",
    "RequestRecorder",
    "TripoClient",
]
