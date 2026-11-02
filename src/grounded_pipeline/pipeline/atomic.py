"""Atomic file and directory writes (plan 19 §4).

Every output is written to a temporary sibling and then :func:`os.replace`d into place, so a
crash never leaves a half-written ``state.json`` or report behind.
"""

from __future__ import annotations

import json
import os
import secrets
import shutil
from pathlib import Path
from typing import Any

from pydantic import BaseModel


def _tmp_sibling(path: Path) -> Path:
    return path.with_name(f".{path.name}.tmp-{os.getpid()}-{secrets.token_hex(4)}")


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` atomically (temp sibling + ``os.replace``).

    Args:
        path: Destination; parent directories are created.
        data: File contents.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_sibling(path)
    try:
        with tmp.open("wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    """Write ``text`` to ``path`` atomically."""
    atomic_write_bytes(path, text.encode(encoding))


def to_jsonable(obj: object) -> Any:  # noqa: ANN401 - JSON is dynamically typed by nature
    """Convert pydantic models, paths and nested containers into plain JSON data."""
    if isinstance(obj, BaseModel):
        return obj.model_dump(mode="json")
    if isinstance(obj, Path):
        return obj.as_posix()
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple | set | frozenset):
        return [to_jsonable(v) for v in obj]
    return obj


def dump_json(obj: object) -> str:
    """Canonical pretty JSON: ``indent=2``, sorted keys, UTF-8, trailing newline."""
    return json.dumps(to_jsonable(obj), indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def atomic_write_json(path: Path, obj: object) -> None:
    """Write ``obj`` (pydantic model, dict, list, …) as canonical JSON atomically."""
    atomic_write_text(path, dump_json(obj))


def atomic_replace_dir(src_tmp: Path, dest: Path, *, keep_old: bool = False) -> Path | None:
    """Swap directory ``src_tmp`` into place at ``dest``.

    An existing ``dest`` is first renamed to ``<dest>.old-<token>``; ``src_tmp`` is then renamed to
    ``dest``; finally the old directory is deleted unless ``keep_old`` is set. If the second rename
    fails the previous directory is restored.

    Args:
        src_tmp: Fully written staging directory (same filesystem as ``dest``).
        dest: Final directory path.
        keep_old: Keep the previous directory under its ``.old-<token>`` name and return it.

    Returns:
        The path of the retained old directory when ``keep_old`` is set and one existed, else
        ``None``.

    Raises:
        FileNotFoundError: ``src_tmp`` does not exist.
        OSError: A rename failed (the previous ``dest`` is restored when possible).
    """
    if not src_tmp.is_dir():
        msg = f"staging directory does not exist: {src_tmp}"
        raise FileNotFoundError(msg)
    dest.parent.mkdir(parents=True, exist_ok=True)
    old: Path | None = None
    if dest.exists():
        old = dest.with_name(f"{dest.name}.old-{secrets.token_hex(4)}")
        dest.replace(old)
    try:
        src_tmp.replace(dest)
    except OSError:
        if old is not None and not dest.exists():
            old.replace(dest)
        raise
    if old is None:
        return None
    if keep_old:
        return old
    shutil.rmtree(old, ignore_errors=True)
    return None


__all__ = [
    "atomic_replace_dir",
    "atomic_write_bytes",
    "atomic_write_json",
    "atomic_write_text",
    "dump_json",
    "to_jsonable",
]
