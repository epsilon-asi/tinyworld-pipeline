"""Fingerprint composition helpers (plan 13 §3).

Fingerprints MUST NOT include timestamps, run ids, absolute paths, or environment-dependent values
other than configuration.
"""

from __future__ import annotations

import hashlib
import threading
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel

from tinyworld_pipeline.domain.ids import Sha256

_CHUNK = 1 << 20
_SEPARATOR = "\x1f"

_file_sha_cache: dict[tuple[str, int, int], str] = {}
_file_sha_lock = threading.Lock()


def file_sha(path: Path) -> Sha256:
    """Streamed SHA-256 of a file, cached per process by ``(path, mtime_ns, size)``."""
    stat = path.stat()
    key = (str(path.resolve()), stat.st_mtime_ns, stat.st_size)
    with _file_sha_lock:
        cached = _file_sha_cache.get(key)
    if cached is not None:
        return cached
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    value = digest.hexdigest()
    with _file_sha_lock:
        _file_sha_cache[key] = value
    return value


def cfg_dump(section: BaseModel) -> str:
    """Stable JSON of a config section (sorted keys, ``exclude_none``)."""
    data = section.model_dump(mode="json", exclude_none=True)
    return _canonical_json(data)


def sha_text(text: str) -> Sha256:
    """SHA-256 of UTF-8 text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha_many(*parts: str) -> Sha256:
    """SHA-256 of the parts joined with an ASCII unit separator (order matters)."""
    return sha_text(_SEPARATOR.join(parts))


def _canonical_json(data: object) -> str:
    import json  # noqa: PLC0415 - local to keep module import cheap

    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


@lru_cache(maxsize=1)
def blender_jobs_code_sha() -> Sha256:
    """SHA over the sorted contents of ``blender_jobs/**/*.py`` (lazy, cached; plan 19 §2).

    Changing Blender-side code therefore invalidates ``prepare`` and everything downstream.
    """
    root = Path(__file__).resolve().parents[1] / "blender_jobs"
    digest = hashlib.sha256()
    for file in sorted(root.rglob("*.py")):
        digest.update(file.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(file.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def clear_caches() -> None:
    """Forget cached hashes (tests)."""
    with _file_sha_lock:
        _file_sha_cache.clear()
    blender_jobs_code_sha.cache_clear()
