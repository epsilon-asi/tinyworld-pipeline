"""Production ``Clock`` / ``Sleeper`` adapters, UTC timestamps and a stopwatch (plan 13 §5).

Tests inject fake clocks; nothing else in the pipeline calls ``time`` directly.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime

from grounded_pipeline.domain.ports import Clock


class SystemClock:
    """Monotonic wall clock backed by :func:`time.monotonic`."""

    def monotonic(self) -> float:
        """Seconds from an arbitrary fixed origin; never goes backwards."""
        return time.monotonic()


class SystemSleeper:
    """Blocking sleeper backed by :func:`time.sleep`."""

    def sleep(self, seconds: float) -> None:
        """Block for ``seconds`` (negative values return immediately)."""
        if seconds > 0:
            time.sleep(seconds)


def utc_now_iso() -> str:
    """Current UTC time as ``YYYY-MM-DDTHH:MM:SS.mmmZ`` (the form used in state and reports)."""
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def compact_utc_timestamp() -> str:
    """Filesystem-safe UTC timestamp (``YYYYmmddTHHMMSSffffff``) for backup file names."""
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")


class Stopwatch:
    """Elapsed-time helper over an injected :class:`Clock`.

    Args:
        clock: The clock to read; defaults to :class:`SystemClock`.

    Example:
        >>> sw = Stopwatch(SystemClock())
        >>> sw.elapsed_s >= 0.0
        True
    """

    def __init__(self, clock: Clock | None = None) -> None:
        self._clock: Clock = clock if clock is not None else SystemClock()
        self._start = self._clock.monotonic()
        self._stop: float | None = None

    def restart(self) -> None:
        """Reset the origin to now and clear any recorded stop."""
        self._start = self._clock.monotonic()
        self._stop = None

    def stop(self) -> float:
        """Freeze the elapsed time and return it (idempotent)."""
        if self._stop is None:
            self._stop = self._clock.monotonic()
        return self.elapsed_s

    @property
    def elapsed_s(self) -> float:
        """Seconds since start (or until :meth:`stop` when called), rounded to milliseconds."""
        end = self._stop if self._stop is not None else self._clock.monotonic()
        return round(max(0.0, end - self._start), 3)


__all__ = ["Stopwatch", "SystemClock", "SystemSleeper", "compact_utc_timestamp", "utc_now_iso"]
