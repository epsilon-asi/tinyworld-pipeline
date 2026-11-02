from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from grounded_pipeline.domain.library_config import VerifySection
from grounded_pipeline.pipeline import hashing
from grounded_pipeline.pipeline.hashing import (
    blender_jobs_code_sha,
    cfg_dump,
    clear_caches,
    file_sha,
    sha_many,
    sha_text,
)

SHA_EMPTY = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


@pytest.fixture(autouse=True)
def _fresh_caches() -> None:
    clear_caches()


def test_file_sha_is_stable_and_cached(tmp_path: Path) -> None:
    f = tmp_path / "a.bin"
    f.write_bytes(b"hello")
    first = file_sha(f)
    assert first == file_sha(f)
    assert len(first) == 64
    (tmp_path / "empty").write_bytes(b"")
    assert file_sha(tmp_path / "empty") == SHA_EMPTY


def test_file_sha_cache_invalidates_on_mtime_change(tmp_path: Path) -> None:
    f = tmp_path / "a.bin"
    f.write_bytes(b"aaaa")
    before = file_sha(f)
    stat = f.stat()
    f.write_bytes(b"bbbb")  # same size: only content + mtime differ
    os.utime(f, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    after = file_sha(f)
    assert after != before
    assert after == sha_text("bbbb")


def test_file_sha_depends_on_content_not_path(tmp_path: Path) -> None:
    a = tmp_path / "one" / "x.glb"
    b = tmp_path / "two" / "deeper" / "y.glb"
    a.parent.mkdir(parents=True)
    b.parent.mkdir(parents=True)
    a.write_bytes(b"same bytes")
    b.write_bytes(b"same bytes")
    assert file_sha(a) == file_sha(b)
    assert sha_many(file_sha(a), "x") == sha_many(file_sha(b), "x")


def test_sha_many_is_order_sensitive_and_separator_safe() -> None:
    assert sha_many("a", "b") != sha_many("b", "a")
    assert sha_many("ab", "c") != sha_many("a", "bc")
    assert sha_many("a") == sha_text("a")


def test_cfg_dump_is_canonical() -> None:
    a = cfg_dump(VerifySection(route="auto", position_quantum=1e-4))
    b = cfg_dump(VerifySection())
    assert a == b
    assert cfg_dump(VerifySection(route="bake")) != a
    assert a.startswith('{"allow_geometry_change":false')


def test_blender_jobs_code_sha_changes_when_a_job_file_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline = blender_jobs_code_sha()
    assert baseline == blender_jobs_code_sha()  # cached
    src_root = Path(hashing.__file__).resolve().parents[1]
    fake_pkg = tmp_path / "grounded_pipeline"
    shutil.copytree(src_root / "blender_jobs", fake_pkg / "blender_jobs")
    (fake_pkg / "pipeline").mkdir()
    monkeypatch.setattr(hashing, "__file__", str(fake_pkg / "pipeline" / "hashing.py"))
    clear_caches()
    copied = blender_jobs_code_sha()
    assert copied == baseline  # identical contents, different absolute location
    job = next(iter(sorted((fake_pkg / "blender_jobs").rglob("*.py"))))
    job.write_text(job.read_text(encoding="utf-8") + "\n# changed\n", encoding="utf-8")
    clear_caches()
    assert blender_jobs_code_sha() != baseline
