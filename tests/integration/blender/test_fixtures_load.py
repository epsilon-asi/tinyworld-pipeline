"""Smoke test: the committed FBX fixtures load in a real Blender with the pipeline's import kwargs.

One Blender process runs ``tools/fixture_probe.py`` over all four files and writes a JSON summary;
the assertions below pin the structure every later Blender job test relies on (identical rig across
files, action names, counts, textures resolved via the relative path).
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.blender

ROOT = Path(__file__).resolve().parents[3]
FIXTURE_DIR = ROOT / "tests" / "fixtures" / "fbx"
PROBE = ROOT / "tools" / "fixture_probe.py"
FILES = ("rigged_cube.fbx", "rigged_cube@walk.fbx", "palette_cube.fbx", "shapekey_cube.fbx")
MAX_FIXTURE_BYTES = 200 * 1024

# The one rig every fixture shares (what the rig-fidelity round trip compares).
EXPECTED_RIG = {
    "bones": ["Root", "Spine", "Head"],
    "bone_parents": {"Root": None, "Spine": "Root", "Head": "Spine"},
    "bone_heads": {"Root": [0.0, 0.0, 0.0], "Spine": [0.0, 0.0, 0.5], "Head": [0.0, 0.0, 1.0]},
    "bone_tails": {"Root": [0.0, 0.0, 0.5], "Spine": [0.0, 0.0, 1.0], "Head": [0.0, 0.0, 1.5]},
}
EXPECTED_CUBE = {
    "vertices": 26,
    "faces": 24,
    "triangles": 48,
    "uv_layers": ["UVMap"],
    "vertex_groups": ["Head", "Root", "Spine"],
    "weighted_vertices": 26,
    "armature_modifier_targets": ["Armature"],
    "parent": "Armature",
}


@pytest.fixture(scope="module")
def summaries(blender_exe: Path, tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    out = tmp_path_factory.mktemp("probe") / "summary.json"
    cmd = [
        str(blender_exe),
        "-b",
        "--factory-startup",
        "--python-exit-code",
        "2",
        "--python",
        str(PROBE),
        "--",
        *(str(FIXTURE_DIR / name) for name in FILES),
        "--out",
        str(out),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300, check=False)  # noqa: S603  # trusted local executable + repo paths
    assert proc.returncode == 0, f"probe failed\n{proc.stdout[-4000:]}\n{proc.stderr[-4000:]}"
    result: dict[str, Any] = json.loads(out.read_text(encoding="utf-8"))
    return result


def _cube(summary: dict[str, Any]) -> dict[str, Any]:
    cube: dict[str, Any] = summary["meshes"]["Cube"]
    return cube


def test_fixture_files_are_small() -> None:
    for name in FILES:
        assert (FIXTURE_DIR / name).stat().st_size < MAX_FIXTURE_BYTES, name
    for png in ("cube_texture.png", "palette.png"):
        assert (FIXTURE_DIR / png).stat().st_size < 1024, png


@pytest.mark.parametrize("name", FILES)
def test_every_fixture_has_the_same_rig(summaries: dict[str, Any], name: str) -> None:
    assert summaries[name]["armatures"] == {"Armature": EXPECTED_RIG}


def test_rigged_cube(summaries: dict[str, Any]) -> None:
    s = summaries["rigged_cube.fbx"]
    assert s["objects"] == [["Armature", "ARMATURE"], ["Cube", "MESH"]]
    # The importer names actions "<object>|<take>"; the take is already "<object>|<action>".
    assert s["actions"] == {"Armature|Armature|idle": {"frame_range": [1.0, 24.0], "fcurves": 39}}
    cube = _cube(s)
    assert {k: cube[k] for k in EXPECTED_CUBE} == EXPECTED_CUBE
    assert cube["shape_keys"] == []
    assert cube["materials"] == ["cube_mat"]
    assert (
        cube["uv_unique_points"] == 39
    )  # default cube cross, subdivided once: no collapsed islands
    assert cube["uv_bbox"] == [0.125, 0.0, 0.875, 1.0]
    assert s["images"] == {
        "cube_texture.png": {"size": [4, 4], "filename": "cube_texture.png", "found": True}
    }


def test_walk_clip_is_armature_only(summaries: dict[str, Any]) -> None:
    s = summaries["rigged_cube@walk.fbx"]
    assert s["objects"] == [["Armature", "ARMATURE"]]
    assert s["meshes"] == {}
    assert s["images"] == {}
    assert s["actions"] == {"Armature|Armature|walk": {"frame_range": [1.0, 24.0], "fcurves": 39}}


def test_palette_cube_has_collapsed_uvs(summaries: dict[str, Any]) -> None:
    s = summaries["palette_cube.fbx"]
    assert list(s["actions"]) == ["Armature|Armature|idle"]
    cube = _cube(s)
    assert {k: cube[k] for k in EXPECTED_CUBE} == EXPECTED_CUBE
    assert cube["materials"] == ["palette_mat"]
    assert (
        cube["uv_unique_points"] == 24
    )  # 6 sides x 4 corners of one tiny quad each (4 faces share it)
    u0, v0, u1, v1 = cube["uv_bbox"]
    assert 0.0 < u0 < u1 < 1.0
    assert 0.0 < v0 < v1 < 1.0
    assert s["images"] == {
        "palette.png": {"size": [16, 4], "filename": "palette.png", "found": True}
    }


def test_shapekey_cube(summaries: dict[str, Any]) -> None:
    s = summaries["shapekey_cube.fbx"]
    cube = _cube(s)
    assert {k: cube[k] for k in EXPECTED_CUBE} == EXPECTED_CUBE
    assert cube["shape_keys"] == ["Basis", "blink"]
    assert cube["materials"] == ["cube_mat"]
    # The importer also creates a 1-curve action for the baked blend-shape channel ("Key|...").
    assert set(s["actions"]) == {"Armature|Armature|idle", "Key|Armature|idle"}
    assert s["actions"]["Armature|Armature|idle"]["frame_range"] == [1.0, 24.0]
