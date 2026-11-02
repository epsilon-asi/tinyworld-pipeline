"""``gp unity sync``: copy ``output/<id>`` into a Unity project folder, keeping ``.meta`` GUIDs.

Semantics (plan 08 §2, 12 §2 step 4, appendix C §7):

1. Stage into ``<dest_root>/<model_id>.tmp-<run_token>`` (a full copy of the output folder).
2. For every staged file or directory that also exists at the same relative path in the current
   ``<dest_root>/<model_id>``, carry its sibling ``.meta`` over so Unity keeps the asset GUID and
   scene/prefab references survive re-runs.
3. Atomic swap: rename the current folder to ``<model_id>.old-<run_token>``, rename the staged
   folder into place, delete the old one. Any failure restores the previous folder and removes the
   staging folder. ``.meta`` files of removed assets disappear together with the old folder.

Symlinks inside the output folder are never followed (they could point outside it) and are not
copied. ADR 0002: Unity lives on another machine; the destination is a local or synced folder.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from tinyworld_pipeline.domain.errors import HandoffError, Problem

META_SUFFIX = ".meta"
E_UNITY_DIR = "E_UNITY_DIR"


@dataclass(frozen=True)
class SyncReport:
    """What one :func:`copy_model` call did."""

    copied: int  # regular files copied from output_dir
    meta_preserved: int  # .meta files carried over from the previous destination folder
    removed: int  # files of the previous folder with no counterpart in the new one (.meta excluded)
    bytes_copied: int  # payload bytes copied (excluding carried-over .meta files)
    skipped_symlinks: int = 0  # symlinked files/dirs in output_dir that were not copied


def validate_dest(dest_root: Path) -> list[Problem]:
    """Check that ``dest_root`` (``unity_import_dir``) is an existing, writable directory.

    Returns:
        ``[]`` when usable, otherwise one ``E_UNITY_DIR`` error.
    """
    if not dest_root.exists():
        message = "unity_import_dir does not exist"
        hint = "create it or fix [library].unity_import_dir"
    elif not dest_root.is_dir():
        message = "unity_import_dir is not a directory"
        hint = "point it at a folder inside the Unity project's Assets/"
    elif not os.access(dest_root, os.W_OK):
        message = "unity_import_dir is not writable"
        hint = "fix the folder permissions"
    else:
        return []
    problem = Problem(
        code=E_UNITY_DIR, severity="error", path=str(dest_root), message=message, hint=hint
    )
    return [problem]


def copy_model(output_dir: Path, dest_root: Path, model_id: str, *, run_token: str) -> SyncReport:
    """Copy ``output_dir`` to ``<dest_root>/<model_id>`` atomically, preserving ``.meta`` files.

    Args:
        output_dir: ``output/<model_id>`` produced by the handoff stage.
        dest_root: Unity import folder (``unity_import_dir``).
        model_id: Destination folder name.
        run_token: Unique token (run id) used to name the staging and backup folders.

    Returns:
        A :class:`SyncReport`.

    Raises:
        HandoffError: The destination is unusable (``E_UNITY_DIR``), the output folder is missing,
            or copying / swapping failed (previous destination left intact).
    """
    if not output_dir.is_dir():
        raise HandoffError(f"output folder {output_dir} does not exist")
    problems = validate_dest(dest_root)
    if problems:
        raise HandoffError(
            problems[0].message,
            detail={"code": E_UNITY_DIR, "path": str(dest_root)},
            hint=problems[0].hint,
        )
    final = dest_root / model_id
    tmp = dest_root / f"{model_id}.tmp-{run_token}"
    old = dest_root / f"{model_id}.old-{run_token}"
    _rmtree_if_exists(tmp)
    try:
        copied, bytes_copied, skipped = _stage_copy(output_dir, tmp)
        meta_preserved = _carry_meta(final, tmp) if final.is_dir() else 0
        removed = _count_removed(final, tmp) if final.is_dir() else 0
    except OSError as exc:
        _rmtree_if_exists(tmp)
        raise HandoffError(f"copying {output_dir} to {tmp} failed: {exc}", cause=exc) from exc
    _swap(final, tmp, old)
    return SyncReport(
        copied=copied,
        meta_preserved=meta_preserved,
        removed=removed,
        bytes_copied=bytes_copied,
        skipped_symlinks=skipped,
    )


# ------------------------------------------------------------------------------------ helpers
def _stage_copy(src: Path, dst: Path) -> tuple[int, int, int]:
    """Copy ``src`` into ``dst`` without following symlinks; returns (files, bytes, skipped)."""
    copied = total = skipped = 0
    dst.mkdir(parents=True, exist_ok=False)
    for root, dirnames, filenames in os.walk(src, followlinks=False):
        rel = Path(root).relative_to(src)
        kept_dirs: list[str] = []
        for name in dirnames:
            if (Path(root) / name).is_symlink():
                skipped += 1
            else:
                kept_dirs.append(name)
                (dst / rel / name).mkdir()
        dirnames[:] = kept_dirs  # prunes os.walk; symlinked dirs are never descended
        for name in sorted(filenames):
            source = Path(root) / name
            if source.is_symlink():
                skipped += 1
                continue
            shutil.copy2(source, dst / rel / name)
            copied += 1
            total += source.stat().st_size
    return copied, total, skipped


def _carry_meta(previous: Path, staged: Path) -> int:
    """Carry ``<rel>.meta`` from ``previous`` for each ``<rel>`` that exists in ``staged``."""
    preserved = 0
    for entry in sorted(staged.rglob("*")):
        rel = entry.relative_to(staged)
        if entry.name.endswith(META_SUFFIX):
            continue
        meta = previous / rel.with_name(rel.name + META_SUFFIX)
        target = staged / rel.with_name(rel.name + META_SUFFIX)
        if meta.is_file() and not target.exists() and (previous / rel).exists():
            shutil.copy2(meta, target)
            preserved += 1
    return preserved


def _count_removed(previous: Path, staged: Path) -> int:
    """Files (not ``.meta``) present in ``previous`` but absent from ``staged``."""
    return sum(
        1
        for entry in previous.rglob("*")
        if entry.is_file()
        and not entry.name.endswith(META_SUFFIX)
        and not (staged / entry.relative_to(previous)).exists()
    )


def _swap(final: Path, tmp: Path, old: Path) -> None:
    """Rename ``final`` aside, move ``tmp`` into place, delete the old copy; restore on failure."""
    had_previous = final.exists()
    try:
        if had_previous:
            final.rename(old)
        try:
            tmp.rename(final)
        except OSError:
            if had_previous:
                old.rename(final)
            raise
    except OSError as exc:
        _rmtree_if_exists(tmp)
        raise HandoffError(f"replacing {final} failed: {exc}", cause=exc) from exc
    if had_previous:
        shutil.rmtree(old, ignore_errors=True)


def _rmtree_if_exists(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path, ignore_errors=True)


__all__ = ["E_UNITY_DIR", "META_SUFFIX", "SyncReport", "copy_model", "validate_dest"]
