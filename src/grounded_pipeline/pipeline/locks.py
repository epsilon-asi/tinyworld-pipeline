"""Process and library-level locks (plan 13 §7).

* ``library/.pipeline/run.lock`` — one ``gp run`` per library (:func:`run_lock`).
* ``BoundedSemaphore(blender_parallelism)`` — caps concurrent Blender processes
  (:func:`blender_semaphore`).

The per-model ``state.lock`` lives inside
:class:`~grounded_pipeline.pipeline.state_store.JsonStateStore`.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from filelock import FileLock, Timeout

from grounded_pipeline.domain.errors import LibraryBusyError


@contextmanager
def run_lock(path: Path, *, timeout_s: float = 0) -> Iterator[None]:
    """Hold the library run lock for the duration of the block.

    Args:
        path: Lock file (``library/.pipeline/run.lock``); parents are created.
        timeout_s: Seconds to wait for the lock; ``0`` tries exactly once.

    Raises:
        LibraryBusyError: The lock is held by another run (exit code 2).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(path))
    try:
        lock.acquire(timeout=timeout_s)
    except Timeout as exc:
        raise LibraryBusyError(
            f"another run holds {path}",
            detail={"lock": str(path)},
            hint="wait for the other `gp run` to finish, or delete the lock if it is stale",
            cause=exc,
        ) from exc
    try:
        yield
    finally:
        lock.release()


def blender_semaphore(n: int) -> threading.BoundedSemaphore:
    """Semaphore wrapping every ``BlenderRunner.run`` (``[library].blender_parallelism``).

    Raises:
        ValueError: ``n`` is not positive.
    """
    if n < 1:
        msg = f"blender parallelism must be >= 1, got {n}"
        raise ValueError(msg)
    return threading.BoundedSemaphore(n)


__all__ = ["blender_semaphore", "run_lock"]
