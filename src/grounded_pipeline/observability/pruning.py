"""Retention of per-stage logs and kept job directories (plan 14 §8: last 5 per stage).

Pure path logic plus ``unlink``/``rmtree``; nothing here logs or prints, so callers decide what to
report.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Final

__all__ = ["STAGE_LOG_RE", "prune_job_dirs", "prune_stage_logs"]

STAGE_LOG_RE: Final[re.Pattern[str]] = re.compile(
    r"^(?P<stage>[a-z][a-z0-9_]*)-(?P<run_id>[0-9A-HJKMNP-TV-Z]{26})\.log$"
)
"""``<stage>-<run_id>.log`` where ``run_id`` is a Crockford-base32 ULID."""

DEFAULT_KEEP: Final[int] = 5


def prune_stage_logs(logs_dir: Path, *, keep_per_stage: int = DEFAULT_KEEP) -> list[Path]:
    """Delete stage logs older than the newest ``keep_per_stage`` per stage.

    Files that do not match :data:`STAGE_LOG_RE` are ignored. Newest is decided by ``run_id``
    (ULIDs sort lexicographically by creation time) with file mtime as the tiebreak.

    Args:
        logs_dir: ``<model>/.pipeline/logs``. A missing directory is treated as empty.
        keep_per_stage: How many logs to keep per stage (``0`` removes all).

    Returns:
        The deleted paths, sorted.

    Raises:
        ValueError: ``keep_per_stage`` is negative.
    """
    if keep_per_stage < 0:
        msg = f"keep_per_stage must be >= 0, got {keep_per_stage}"
        raise ValueError(msg)
    if not logs_dir.is_dir():
        return []

    groups: dict[str, list[tuple[str, float, Path]]] = {}
    for path in logs_dir.iterdir():
        match = STAGE_LOG_RE.match(path.name)
        if match is None or not path.is_file():
            continue
        groups.setdefault(match["stage"], []).append((match["run_id"], path.stat().st_mtime, path))

    deleted: list[Path] = []
    for entries in groups.values():
        entries.sort(key=lambda e: (e[0], e[1]), reverse=True)
        for _run_id, _mtime, path in entries[keep_per_stage:]:
            path.unlink(missing_ok=True)
            deleted.append(path)
    return sorted(deleted)


def prune_job_dirs(
    parent: Path, *, keep_per_job: int = DEFAULT_KEEP, prefix: str = ""
) -> list[Path]:
    """Remove kept temp job directories beyond the newest ``keep_per_job`` per job kind.

    Directories are named ``<prefix><job>-<suffix>`` (``gp-blender-prepare-ab12cd`` from
    ``tempfile.mkdtemp``, or ``prepare-<run_id>`` under a logs folder). The group key is everything
    between ``prefix`` and the last ``-``; newest is decided by directory mtime, then name.

    Args:
        parent: Directory containing the job directories. A missing directory is treated as empty.
        keep_per_job: How many directories to keep per job kind (``0`` removes all).
        prefix: Only directories whose name starts with ``prefix`` are considered.

    Returns:
        The removed directories, sorted.

    Raises:
        ValueError: ``keep_per_job`` is negative.
    """
    if keep_per_job < 0:
        msg = f"keep_per_job must be >= 0, got {keep_per_job}"
        raise ValueError(msg)
    if not parent.is_dir():
        return []

    groups: dict[str, list[tuple[float, str, Path]]] = {}
    for path in parent.iterdir():
        if not path.is_dir() or not path.name.startswith(prefix):
            continue
        rest = path.name[len(prefix) :]
        job, sep, _suffix = rest.rpartition("-")
        if not sep or not job:
            continue
        groups.setdefault(job, []).append((path.stat().st_mtime, path.name, path))

    removed: list[Path] = []
    for entries in groups.values():
        entries.sort(reverse=True)
        for _mtime, _name, path in entries[keep_per_job:]:
            shutil.rmtree(path, ignore_errors=True)
            removed.append(path)
    return sorted(removed)
