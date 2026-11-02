"""Request pacing and task polling (plan 10 §3 "Polling", §4 "Rate limiting").

Both pieces are deterministic under an injected :class:`Clock`/:class:`Sleeper` so the unit
tests never sleep for real.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Protocol

import structlog

from grounded_pipeline.domain.errors import (
    CancelledError,
    TripoTaskFailedError,
    TripoTaskTimeoutError,
)
from grounded_pipeline.domain.ports import CancelToken, Clock, Sleeper
from grounded_pipeline.tripo.models import TaskDetail

EVENT_POLL = "tripo.poll"

BACKOFF_START_AFTER_S = 60.0  # polls stay at interval_s for the first minute (plan 10 §3)
BACKOFF_FACTOR = 1.5
MAX_POLL_INTERVAL_S = 10.0

_log = structlog.get_logger(__name__)


class RateLimiter:
    """Concurrency cap plus minimum spacing between any two requests, shared across threads.

    Args:
        max_concurrent: Number of task slots (``TripoSection.max_concurrent_tasks``).
        min_interval_s: Minimum spacing between two paced requests
            (``TripoSection.min_request_interval_s``); ``0`` disables pacing.
        clock: Monotonic clock port.
        sleeper: Sleep port (instant fake in tests).
    """

    def __init__(
        self, max_concurrent: int, min_interval_s: float, *, clock: Clock, sleeper: Sleeper
    ) -> None:
        if max_concurrent < 1:
            msg = f"max_concurrent must be >= 1, got {max_concurrent}"
            raise ValueError(msg)
        if min_interval_s < 0:
            msg = f"min_interval_s must be >= 0, got {min_interval_s}"
            raise ValueError(msg)
        self._max_concurrent = max_concurrent
        self._min_interval_s = float(min_interval_s)
        self._clock = clock
        self._sleeper = sleeper
        self._lock = threading.Lock()
        self._slots = threading.BoundedSemaphore(max_concurrent)
        self._next_allowed_at = float("-inf")

    @property
    def max_concurrent(self) -> int:
        """Capacity of the slot pool."""
        return self._max_concurrent

    def acquire_slot(self, *, timeout_s: float | None = None) -> bool:
        """Block until a task slot is free.

        Args:
            timeout_s: Give up after this many (real) seconds; ``None`` waits forever.

        Returns:
            ``True`` when a slot was acquired, ``False`` on timeout.
        """
        return self._slots.acquire(timeout=timeout_s)

    def release_slot(self) -> None:
        """Return a slot acquired with :meth:`acquire_slot`.

        Raises:
            ValueError: When released more often than acquired.
        """
        self._slots.release()

    @contextmanager
    def slot(self) -> Iterator[None]:
        """Context manager form of :meth:`acquire_slot`/:meth:`release_slot`."""
        self.acquire_slot()
        try:
            yield
        finally:
            self.release_slot()

    def pace(self) -> None:
        """Sleep as needed so this request starts ≥ ``min_interval_s`` after the previous one.

        The reservation of the next start time happens under the lock, so concurrent callers
        are serialised into non-overlapping windows even though the sleep itself is unlocked.
        """
        if self._min_interval_s <= 0:
            return
        with self._lock:
            now = self._clock.monotonic()
            wait = max(0.0, self._next_allowed_at - now)
            self._next_allowed_at = max(now, self._next_allowed_at) + self._min_interval_s
        if wait > 0:
            self._sleeper.sleep(wait)


class TaskGetter(Protocol):
    """The slice of :class:`TripoClient` that :func:`wait_for_task` needs."""

    def get_task(self, task_id: str) -> TaskDetail:
        """Fetch the current task detail."""
        ...


def next_interval(current_s: float, elapsed_s: float) -> float:
    """Pure poll-interval schedule: constant for 60 s, then x1.5 per poll, capped at 10 s."""
    if elapsed_s < BACKOFF_START_AFTER_S:
        return current_s
    return min(current_s * BACKOFF_FACTOR, MAX_POLL_INTERVAL_S)


def wait_for_task(
    client: TaskGetter,
    task_id: str,
    *,
    interval_s: float,
    timeout_s: float,
    cancel: CancelToken,
    clock: Clock,
    sleeper: Sleeper,
    on_progress: Callable[[TaskDetail], None] | None = None,
) -> TaskDetail:
    """Poll ``GET /tasks/{id}`` until the task reaches a terminal state.

    Args:
        client: Anything with ``get_task(task_id) -> TaskDetail``.
        task_id: Task to wait for.
        interval_s: Initial poll interval (grows x1.5 per poll after 60 s, capped at 10 s).
        timeout_s: Overall budget measured on ``clock``; ``unknown`` polls count toward it.
        cancel: Checked before every request and after every sleep.
        clock: Monotonic clock port.
        sleeper: Sleep port.
        on_progress: Called with every fetched :class:`TaskDetail` (including the final one).

    Returns:
        The detail whose ``status == "success"``.

    Raises:
        TripoTaskFailedError: On ``failed``/``cancelled``/``banned``/``expired``
            (``detail = {status, error_code, error_message}``).
        TripoTaskTimeoutError: When ``timeout_s`` elapses first.
        CancelledError: When ``cancel`` is set (the task id is already persisted for resume).
    """
    if interval_s <= 0:
        msg = f"interval_s must be > 0, got {interval_s}"
        raise ValueError(msg)
    log = _log.bind(task_id=task_id)
    started = clock.monotonic()
    current = float(interval_s)
    polls = 0
    while True:
        if cancel.is_set():
            raise CancelledError(f"cancelled while waiting for task {task_id}")
        detail = client.get_task(task_id)
        polls += 1
        elapsed = clock.monotonic() - started
        log.debug(EVENT_POLL, status=detail.status, progress=detail.progress, elapsed_s=elapsed)
        if on_progress is not None:
            on_progress(detail)
        if detail.is_success:
            return detail
        if detail.is_terminal_failure:
            raise TripoTaskFailedError(
                f"task {task_id} ended with status {detail.status!r}"
                + (f": {detail.error_message}" if detail.error_message else ""),
                detail={
                    "status": detail.status,
                    "error_code": detail.error_code,
                    "error_message": detail.error_message,
                },
            )
        remaining = timeout_s - elapsed
        if remaining <= 0:
            raise TripoTaskTimeoutError(
                f"task {task_id} still {detail.status!r} after {timeout_s:g} s ({polls} polls)",
                detail={"status": detail.status, "timeout_s": timeout_s, "polls": polls},
                hint="Raise [tripo].poll_timeout_s; the task id is persisted and resumable.",
            )
        current = next_interval(current, elapsed)
        sleeper.sleep(min(current, remaining))


__all__ = [
    "BACKOFF_FACTOR",
    "BACKOFF_START_AFTER_S",
    "EVENT_POLL",
    "MAX_POLL_INTERVAL_S",
    "RateLimiter",
    "TaskGetter",
    "next_interval",
    "wait_for_task",
]
