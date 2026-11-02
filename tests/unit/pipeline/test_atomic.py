from __future__ import annotations

import json
from pathlib import Path

import pytest

from grounded_pipeline.domain.state import StageRecord
from grounded_pipeline.pipeline.atomic import (
    atomic_replace_dir,
    atomic_write_bytes,
    atomic_write_json,
    atomic_write_text,
    dump_json,
    to_jsonable,
)


def _leftovers(folder: Path) -> list[Path]:
    return [p for p in folder.iterdir() if ".tmp-" in p.name]


def test_atomic_write_bytes_creates_parents_and_leaves_no_tmp(tmp_path: Path) -> None:
    target = tmp_path / "a" / "b" / "file.bin"
    atomic_write_bytes(target, b"\x00\x01")
    assert target.read_bytes() == b"\x00\x01"
    assert _leftovers(target.parent) == []


def test_atomic_write_text_overwrites(tmp_path: Path) -> None:
    target = tmp_path / "t.txt"
    atomic_write_text(target, "one")
    atomic_write_text(target, "two")
    assert target.read_text(encoding="utf-8") == "two"


def test_atomic_write_json_canonical_form(tmp_path: Path) -> None:
    target = tmp_path / "x.json"
    atomic_write_json(target, {"b": 1, "a": {"z": Path("p/q"), "é": [1, (2, 3)]}})
    text = target.read_text(encoding="utf-8")
    assert text.endswith("\n")
    assert text.startswith('{\n  "a": {')
    assert "é" in text  # ensure_ascii=False
    assert json.loads(text) == {"b": 1, "a": {"z": "p/q", "é": [1, [2, 3]]}}


def test_atomic_write_json_accepts_pydantic_models(tmp_path: Path) -> None:
    target = tmp_path / "rec.json"
    atomic_write_json(target, StageRecord(attempts=2))
    assert json.loads(target.read_text())["attempts"] == 2
    assert dump_json(StageRecord()) == dump_json(StageRecord().model_dump())


def test_to_jsonable_passes_scalars_through() -> None:
    assert to_jsonable(3) == 3
    assert to_jsonable({1: {"a", "b"}}) in ({"1": ["a", "b"]}, {"1": ["b", "a"]})


def test_crash_between_tmp_write_and_replace_keeps_previous_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "state.json"
    atomic_write_text(target, "old")

    def boom(self: Path, other: Path) -> Path:
        raise OSError("simulated crash")

    monkeypatch.setattr(Path, "replace", boom)
    with pytest.raises(OSError, match="simulated"):
        atomic_write_text(target, "new")
    monkeypatch.undo()
    assert target.read_text() == "old"
    assert _leftovers(tmp_path) == []


def test_atomic_replace_dir_fresh_destination(tmp_path: Path) -> None:
    staging = tmp_path / "out.tmp"
    staging.mkdir()
    (staging / "f.txt").write_text("x")
    assert atomic_replace_dir(staging, tmp_path / "deep" / "out") is None
    assert (tmp_path / "deep" / "out" / "f.txt").read_text() == "x"
    assert not staging.exists()


def test_atomic_replace_dir_swaps_and_deletes_old(tmp_path: Path) -> None:
    dest = tmp_path / "out"
    dest.mkdir()
    (dest / "stale.txt").write_text("stale")
    staging = tmp_path / "out.tmp"
    staging.mkdir()
    (staging / "new.txt").write_text("new")
    assert atomic_replace_dir(staging, dest) is None
    assert sorted(p.name for p in dest.iterdir()) == ["new.txt"]
    assert [p.name for p in tmp_path.iterdir()] == ["out"]


def test_atomic_replace_dir_keep_old(tmp_path: Path) -> None:
    dest = tmp_path / "out"
    dest.mkdir()
    (dest / "stale.txt").write_text("stale")
    staging = tmp_path / "out.tmp"
    staging.mkdir()
    old = atomic_replace_dir(staging, dest, keep_old=True)
    assert old is not None
    assert old.name.startswith("out.old-")
    assert (old / "stale.txt").exists()


def test_atomic_replace_dir_missing_source(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        atomic_replace_dir(tmp_path / "nope", tmp_path / "out")


def test_atomic_replace_dir_restores_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dest = tmp_path / "out"
    dest.mkdir()
    (dest / "keep.txt").write_text("keep")
    staging = tmp_path / "out.tmp"
    staging.mkdir()
    real_replace = Path.replace
    calls = {"n": 0}

    def flaky(self: Path, other: Path) -> Path:
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("second rename fails")
        return real_replace(self, other)

    monkeypatch.setattr(Path, "replace", flaky)
    with pytest.raises(OSError, match="second rename"):
        atomic_replace_dir(staging, dest)
    monkeypatch.undo()
    assert (dest / "keep.txt").read_text() == "keep"
    assert staging.exists()
