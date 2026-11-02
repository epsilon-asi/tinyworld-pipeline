"""Small runtime ports shared by every layer: cancellation, clock and sleeper.

These live in ``domain`` (not ``pipeline``) because adapters below the pipeline layer — the Tripo
client, the texture provider, the Blender runner — accept them as parameters.
"""

from __future__ import annotations

import threading
from typing import Protocol


class CancelToken:
    """Thread-safe cancellation flag. Checked between stages, between polls, and while waiting
    on subprocesses."""

    def __init__(self) -> None:
        self._event = threading.Event()

    def set(self) -> None:
        """Request cancellation."""
        self._event.set()

    def is_set(self) -> bool:
        """Return whether cancellation has been requested."""
        return self._event.is_set()

    def wait(self, timeout_s: float) -> bool:
        """Block up to ``timeout_s`` seconds; return True if cancellation was requested."""
        return self._event.wait(timeout_s)


class Clock(Protocol):
    """Monotonic clock port."""

    def monotonic(self) -> float:
        """Seconds from an arbitrary fixed origin; never goes backwards."""
        ...


class Sleeper(Protocol):
    """Blocking sleep port (injected so tests never sleep)."""

    def sleep(self, seconds: float) -> None:
        """Block for ``seconds``."""
        ...
