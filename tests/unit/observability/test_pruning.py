from __future__ import annotations

import os
from pathlib import Path

import pytest

from tinyworld_pipeline.observability.pruning import (
    STAGE_LOG_RE,
    prune_job_dirs,
    prune_stage_logs,
)

# Lexicographically increasing ULIDs (the first 10 chars are the timestamp component).
ULIDS = [f"01HZX3V6Y{i}K8Q4M9N7P5R1S0TAB"[:26] for i in range(10)]


def _touch(path: Path, mtime: float | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}", encoding="utf-8")
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


def test_stage_log_regex() -> None:
    match = STAGE_LOG_RE.match(f"prepare-{ULIDS[0]}.log")
    assert match is not None
    assert match["stage"] == "prepare"
    assert match["run_id"] == ULIDS[0]
    assert STAGE_LOG_RE.match("prepare-notaulid.log") is None
    assert STAGE_LOG_RE.match(f"prepare-{ULIDS[0]}.json") is None


def test_prune_keeps_newest_per_stage_by_run_id(tmp_path: Path) -> None:
    files = [_touch(tmp_path / f"prepare-{u}.log") for u in ULIDS[:8]]
    deleted = prune_stage_logs(tmp_path)
    assert deleted == sorted(files[:3])
    remaining = sorted(p.name for p in tmp_path.iterdir())
    assert remaining == sorted(f"prepare-{u}.log" for u in ULIDS[3:8])


def test_prune_handles_mixed_stages_and_ignores_other_files(tmp_path: Path) -> None:
    for u in ULIDS[:7]:
        _touch(tmp_path / f"prepare-{u}.log")
    for u in ULIDS[:3]:
        _touch(tmp_path / f"texture-{u}.log")
    keepers = [
        _touch(tmp_path / "prepare_report.json"),
        _touch(tmp_path / "notes.log"),
        _touch(tmp_path / f"prepare-{ULIDS[0]}.log.bak"),
    ]
    (tmp_path / f"apply-{ULIDS[0]}.log").mkdir()  # a directory with a matching name is ignored

    deleted = prune_stage_logs(tmp_path, keep_per_stage=5)

    assert [p.name for p in deleted] == [f"prepare-{u}.log" for u in ULIDS[:2]]
    assert all(p.exists() for p in keepers)
    assert all((tmp_path / f"texture-{u}.log").exists() for u in ULIDS[:3])
    assert (tmp_path / f"apply-{ULIDS[0]}.log").is_dir()


def test_prune_orders_by_run_id_before_mtime(tmp_path: Path) -> None:
    # An older ULID with a *newer* mtime is still the one pruned: the id is the primary key.
    older_id_newer_mtime = _touch(tmp_path / f"prepare-{ULIDS[0]}.log", mtime=9_000)
    newer_id_older_mtime = _touch(tmp_path / f"prepare-{ULIDS[1]}.log", mtime=1_000)
    assert prune_stage_logs(tmp_path, keep_per_stage=1) == [older_id_newer_mtime]
    assert newer_id_older_mtime.exists()


def test_prune_keep_zero_removes_all_matching(tmp_path: Path) -> None:
    files = [_touch(tmp_path / f"export-{u}.log") for u in ULIDS[:3]]
    assert prune_stage_logs(tmp_path, keep_per_stage=0) == sorted(files)
    assert list(tmp_path.iterdir()) == []


def test_prune_empty_and_missing_dirs(tmp_path: Path) -> None:
    assert prune_stage_logs(tmp_path) == []
    assert prune_stage_logs(tmp_path / "missing") == []
    assert prune_job_dirs(tmp_path) == []
    assert prune_job_dirs(tmp_path / "missing") == []


def test_prune_rejects_negative_keep(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="keep_per_stage"):
        prune_stage_logs(tmp_path, keep_per_stage=-1)
    with pytest.raises(ValueError, match="keep_per_job"):
        prune_job_dirs(tmp_path, keep_per_job=-1)


def test_prune_job_dirs_by_mtime_per_job_kind(tmp_path: Path) -> None:
    def mkdir(name: str, mtime: float) -> Path:
        d = tmp_path / name
        d.mkdir()
        (d / "job.json").write_text("{}", encoding="utf-8")
        os.utime(d, (mtime, mtime))
        return d

    prepares = [mkdir(f"gp-blender-prepare-{i:02d}xyz", mtime=1_000 + i) for i in range(4)]
    bakes = [mkdir(f"gp-blender-bake-{i:02d}xyz", mtime=5_000 + i) for i in range(2)]
    stray_file = _touch(tmp_path / "gp-blender-prepare-file")
    unrelated = tmp_path / "other-dir"
    unrelated.mkdir()
    nodash = tmp_path / "gp-blender-nodash"
    nodash.mkdir()

    removed = prune_job_dirs(tmp_path, keep_per_job=2, prefix="gp-blender-")

    assert removed == sorted(prepares[:2])
    assert all(not p.exists() for p in prepares[:2])
    assert all(p.exists() for p in prepares[2:] + bakes)
    assert stray_file.exists()
    assert unrelated.exists()
    assert nodash.exists()


def test_prune_job_dirs_without_prefix_groups_by_name(tmp_path: Path) -> None:
    dirs = []
    for i, u in enumerate(ULIDS[:3]):
        d = tmp_path / f"prepare-{u}"
        d.mkdir()
        os.utime(d, (1_000 + i, 1_000 + i))
        dirs.append(d)
    assert prune_job_dirs(tmp_path, keep_per_job=1) == sorted(dirs[:2])
    assert dirs[2].exists()
