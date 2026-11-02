from __future__ import annotations

import threading
from pathlib import Path

import pytest

from tinyworld_pipeline.domain.errors import LibraryBusyError
from tinyworld_pipeline.pipeline.locks import blender_semaphore, run_lock


def test_run_lock_creates_parent_and_releases(tmp_path: Path) -> None:
    lock_file = tmp_path / ".pipeline" / "run.lock"
    with run_lock(lock_file):
        assert lock_file.parent.is_dir()
    with run_lock(lock_file):  # re-acquirable after release
        pass


def test_run_lock_busy_raises_library_busy(tmp_path: Path) -> None:
    lock_file = tmp_path / "run.lock"
    held = threading.Event()
    release = threading.Event()

    def holder() -> None:
        with run_lock(lock_file):
            held.set()
            release.wait(5)

    thread = threading.Thread(target=holder)
    thread.start()
    assert held.wait(5)
    try:
        with pytest.raises(LibraryBusyError) as excinfo, run_lock(lock_file, timeout_s=0):
            pass
        assert excinfo.value.exit_code == 2
        assert excinfo.value.detail["lock"] == str(lock_file)
    finally:
        release.set()
        thread.join(5)
    with run_lock(lock_file):  # free again
        pass


def test_blender_semaphore_bounds() -> None:
    sem = blender_semaphore(2)
    assert sem.acquire(blocking=False)
    assert sem.acquire(blocking=False)
    assert not sem.acquire(blocking=False)
    sem.release()
    sem.release()
    with pytest.raises(ValueError, match=">= 1"):
        blender_semaphore(0)
