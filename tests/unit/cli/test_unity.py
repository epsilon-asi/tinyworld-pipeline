"""``gp unity sync``."""

from __future__ import annotations

from pathlib import Path

from .conftest import Harness


def _fake_output(lib: Path, model_id: str) -> Path:
    out = lib / "models" / model_id / "output"
    out.mkdir(parents=True)
    (out / f"{model_id}.fbx").write_bytes(b"FBX")
    (out / "manifest.json").write_text("{}", encoding="utf-8")
    return out


def test_sync_without_destination_exits_2(harness: Harness, lib: Path) -> None:
    result, document = harness.json("unity", "sync", "--library", str(lib))
    assert result.exit_code == 2
    assert document["error"]["code"] == "E_USAGE"
    assert "--dest" in document["error"]["hint"]


def test_sync_missing_destination_dir_exits_3(harness: Harness, lib: Path, tmp_path: Path) -> None:
    result, document = harness.json(
        "unity", "sync", "--library", str(lib), "--dest", str(tmp_path / "nope")
    )
    assert result.exit_code == 3
    assert document["error"]["code"] == "E_VALIDATION"


def test_sync_copies_output_and_skips_models_without_output(
    harness: Harness, lib: Path, tmp_path: Path
) -> None:
    _fake_output(lib, "cube_rigged")
    dest = tmp_path / "Assets" / "Imported"
    dest.mkdir(parents=True)
    (dest / "cube_rigged").mkdir()
    (dest / "cube_rigged" / "stale.txt").write_text("old")
    (dest / "cube_rigged" / "cube_rigged.fbx").write_bytes(b"OLD")
    (dest / "cube_rigged" / "cube_rigged.fbx.meta").write_text("guid")
    result, document = harness.json("unity", "sync", "--library", str(lib), "--dest", str(dest))
    assert result.exit_code == 0, result.output
    rows = {r["model_id"]: r for r in document["models"]}
    assert rows["cube_palette"]["synced"] is False
    assert rows["cube_rigged"]["synced"] is True
    assert rows["cube_rigged"]["copied"] == 2
    assert rows["cube_rigged"]["meta_preserved"] == 1
    assert rows["cube_rigged"]["removed"] == 1
    assert (dest / "cube_rigged" / "cube_rigged.fbx").is_file()
    assert (dest / "cube_rigged" / "cube_rigged.fbx.meta").is_file()
    assert not (dest / "cube_rigged" / "stale.txt").exists()
    assert "skipped (no output/)" in harness.text
    assert "synced to" in harness.text


def test_sync_uses_unity_import_dir_from_config(
    harness: Harness, lib: Path, tmp_path: Path
) -> None:
    _fake_output(lib, "cube_palette")
    dest = tmp_path / "unity_dest"
    dest.mkdir()
    toml = lib / "library.toml"
    toml.write_text(
        toml.read_text().replace(
            'name = "library_minimal"', f'name = "library_minimal"\nunity_import_dir = "{dest}"'
        )
    )
    result = harness.invoke("unity", "sync", "--library", str(lib), "--model", "cube_palette")
    assert result.exit_code == 0, harness.text
    assert (dest / "cube_palette" / "cube_palette.fbx").is_file()
