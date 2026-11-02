from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import pytest

from tinyworld_pipeline.domain.errors import HandoffError
from tinyworld_pipeline.unity.sync import SyncReport, copy_model, validate_dest

FILES = {
    "cube.fbx": b"fbx-v1",
    "manifest.json": b"{}",
    "textures/cube_basecolor.png": b"png-base",
    "textures/cube_normal.png": b"png-normal",
    "animations/cube@walk.fbx": b"anim-walk",
}


def make_output(root: Path, files: dict[str, bytes] | None = None) -> Path:
    out = root / "output" / "cube"
    for rel, data in (files or FILES).items():
        path = out / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return out


def listing(folder: Path) -> dict[str, bytes]:
    return {
        p.relative_to(folder).as_posix(): p.read_bytes()
        for p in sorted(folder.rglob("*"))
        if p.is_file()
    }


def entries(root: Path) -> set[str]:
    return {p.name for p in root.iterdir()}


# ------------------------------------------------------------------------------ fresh copy
def test_fresh_copy_creates_folder_and_reports_counts(tmp_path: Path) -> None:
    out = make_output(tmp_path)
    dest = tmp_path / "Assets"
    dest.mkdir()
    report = copy_model(out, dest, "cube", run_token="run1")
    assert report == SyncReport(
        copied=5,
        meta_preserved=0,
        removed=0,
        bytes_copied=sum(len(v) for v in FILES.values()),
        skipped_symlinks=0,
    )
    assert listing(dest / "cube") == FILES
    assert entries(dest) == {"cube"}  # no tmp / old leftovers


# ------------------------------------------------------------------------------ .meta preservation
def test_resync_preserves_meta_for_same_relative_paths_and_directories(tmp_path: Path) -> None:
    out = make_output(tmp_path)
    dest = tmp_path / "Assets"
    dest.mkdir()
    copy_model(out, dest, "cube", run_token="run1")
    # Unity would now have created .meta files (for files, folders and the model folder itself).
    final = dest / "cube"
    metas = {
        "cube.fbx.meta": b"guid: fbx",
        "manifest.json.meta": b"guid: manifest",
        "textures.meta": b"guid: textures-dir",
        "textures/cube_basecolor.png.meta": b"guid: base",
        "textures/cube_normal.png.meta": b"guid: normal",
        "animations.meta": b"guid: anim-dir",
        "animations/cube@walk.fbx.meta": b"guid: walk",
        "Materials.meta": b"guid: materials-dir",
        "Materials/cube_mat.mat": b"mat",
        "Materials/cube_mat.mat.meta": b"guid: mat",
    }
    for rel, data in metas.items():
        (final / rel).parent.mkdir(parents=True, exist_ok=True)
        (final / rel).write_bytes(data)
    (dest / "cube.meta").write_bytes(b"guid: model-dir")

    # Second pipeline run: normal map removed, basecolor changed, occlusion added.
    new_files = dict(FILES)
    del new_files["textures/cube_normal.png"]
    new_files["textures/cube_basecolor.png"] = b"png-base-v2"
    new_files["textures/cube_occlusion.png"] = b"png-occ"
    shutil.rmtree(out)
    out = make_output(tmp_path, new_files)

    report = copy_model(out, dest, "cube", run_token="run2")

    assert report.copied == 5
    assert report.removed == 2  # cube_normal.png and Materials/cube_mat.mat (Unity-generated)
    # Four file metas (fbx, manifest, basecolor, walk) and two folder metas (textures, animations).
    assert report.meta_preserved == 6
    got = listing(final)
    assert got["textures/cube_basecolor.png"] == b"png-base-v2"
    assert got["textures/cube_basecolor.png.meta"] == b"guid: base"
    assert got["cube.fbx.meta"] == b"guid: fbx"
    assert got["manifest.json.meta"] == b"guid: manifest"
    assert got["animations/cube@walk.fbx.meta"] == b"guid: walk"
    assert got["textures.meta"] == b"guid: textures-dir"
    assert got["animations.meta"] == b"guid: anim-dir"
    assert "textures/cube_normal.png" not in got
    assert "textures/cube_normal.png.meta" not in got  # removed file's .meta gone with old folder
    assert "textures/cube_occlusion.png.meta" not in got  # new file: Unity assigns a fresh GUID
    assert "Materials.meta" not in got  # Unity-generated folder is not pipeline output
    assert (
        dest / "cube.meta"
    ).read_bytes() == b"guid: model-dir"  # folder meta outside is untouched
    assert entries(dest) == {"cube", "cube.meta"}


def test_meta_files_in_output_are_not_overwritten_by_previous(tmp_path: Path) -> None:
    files = dict(FILES)
    files["cube.fbx.meta"] = b"guid: from-output"
    out = make_output(tmp_path, files)
    dest = tmp_path / "Assets"
    (dest / "cube").mkdir(parents=True)
    (dest / "cube" / "cube.fbx").write_bytes(b"old")
    (dest / "cube" / "cube.fbx.meta").write_bytes(b"guid: previous")
    report = copy_model(out, dest, "cube", run_token="t")
    assert report.meta_preserved == 0
    assert (dest / "cube" / "cube.fbx.meta").read_bytes() == b"guid: from-output"


# ------------------------------------------------------------------------------ atomicity
def test_failure_mid_copy_leaves_previous_output_intact_and_no_tmp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = make_output(tmp_path)
    dest = tmp_path / "Assets"
    dest.mkdir()
    copy_model(out, dest, "cube", run_token="run1")
    before = listing(dest / "cube")
    (dest / "cube" / "cube.fbx.meta").write_bytes(b"guid")
    (out / "cube.fbx").write_bytes(b"fbx-v2")

    real_copy2 = shutil.copy2
    calls = {"n": 0}

    def flaky(src: Path, dst: Path) -> Path:
        calls["n"] += 1
        if calls["n"] == 3:
            raise OSError("disk full")
        return Path(real_copy2(src, dst))

    monkeypatch.setattr(shutil, "copy2", flaky)
    with pytest.raises(HandoffError, match="disk full"):
        copy_model(out, dest, "cube", run_token="run2")
    assert listing(dest / "cube") == {**before, "cube.fbx.meta": b"guid"}
    assert entries(dest) == {"cube"}


def test_failed_swap_restores_previous_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = make_output(tmp_path)
    dest = tmp_path / "Assets"
    dest.mkdir()
    copy_model(out, dest, "cube", run_token="run1")
    before = listing(dest / "cube")
    (out / "cube.fbx").write_bytes(b"fbx-v2")

    real_rename = Path.rename

    def failing_rename(self: Path, target: Path | str) -> Path:
        if Path(target) == dest / "cube" and self.name.startswith("cube.tmp-"):
            raise OSError("rename blocked")
        return real_rename(self, target)

    monkeypatch.setattr(Path, "rename", failing_rename)
    with pytest.raises(HandoffError, match="rename blocked"):
        copy_model(out, dest, "cube", run_token="run2")
    assert listing(dest / "cube") == before
    assert entries(dest) == {"cube"}


def test_stale_tmp_from_same_token_is_replaced(tmp_path: Path) -> None:
    out = make_output(tmp_path)
    dest = tmp_path / "Assets"
    stale = dest / "cube.tmp-run1"
    stale.mkdir(parents=True)
    (stale / "junk.txt").write_bytes(b"junk")
    copy_model(out, dest, "cube", run_token="run1")
    assert listing(dest / "cube") == FILES
    assert entries(dest) == {"cube"}


# ------------------------------------------------------------------------------ symlinks
@pytest.mark.skipif(sys.platform == "win32", reason="symlinks need privileges on Windows")
def test_symlinks_inside_output_are_skipped_not_followed(tmp_path: Path) -> None:
    out = make_output(tmp_path)
    secret_dir = tmp_path / "secret"
    secret_dir.mkdir()
    (secret_dir / "key.txt").write_bytes(b"secret")
    (out / "linked_dir").symlink_to(secret_dir, target_is_directory=True)
    (out / "linked_file.txt").symlink_to(secret_dir / "key.txt")
    dest = tmp_path / "Assets"
    dest.mkdir()
    report = copy_model(out, dest, "cube", run_token="t")
    assert report.skipped_symlinks == 2
    assert report.copied == 5
    assert listing(dest / "cube") == FILES
    assert not (dest / "cube" / "linked_dir").exists()


# ------------------------------------------------------------------------------ validation
def test_validate_dest_reports_missing_non_dir_and_unwritable(tmp_path: Path) -> None:
    assert validate_dest(tmp_path) == []

    missing = validate_dest(tmp_path / "nope")
    assert [p.code for p in missing] == ["E_UNITY_DIR"]
    assert missing[0].severity == "error"
    assert "does not exist" in missing[0].message
    assert missing[0].path == str(tmp_path / "nope")

    file_path = tmp_path / "file.txt"
    file_path.write_text("x")
    assert "not a directory" in validate_dest(file_path)[0].message

    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0o500)
    try:
        if os.access(locked, os.W_OK):  # running as root: cannot provoke the condition
            pytest.skip("filesystem permissions not enforced for this user")
        assert "not writable" in validate_dest(locked)[0].message
    finally:
        locked.chmod(0o700)


def test_copy_model_rejects_bad_destination_and_missing_output(tmp_path: Path) -> None:
    out = make_output(tmp_path)
    with pytest.raises(HandoffError, match="does not exist") as info:
        copy_model(out, tmp_path / "missing", "cube", run_token="t")
    assert info.value.detail["code"] == "E_UNITY_DIR"
    dest = tmp_path / "Assets"
    dest.mkdir()
    with pytest.raises(HandoffError, match="output folder"):
        copy_model(tmp_path / "no-output", dest, "cube", run_token="t")
