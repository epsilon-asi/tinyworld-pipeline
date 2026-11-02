"""WP-04 acceptance: ``prepare_model`` under a real Blender through the real runner (plan 09 §2.10).

Every test runs one full job on a committed FBX fixture (``tests/fixtures/fbx/README.md``) and
checks the numbers the rest of the pipeline relies on: Catmull-Clark counts, the UV decision, the
rig fingerprint, and that ``static.glb`` re-derives to the same triangle / unique-position counts
with the orchestrator's own ``gltf`` tools.
"""

from __future__ import annotations

import json
import os
import subprocess
import textwrap
import threading
from pathlib import Path
from typing import Any

import pytest
from ulid import ULID

from grounded_pipeline.blender.protocol import (
    JobRequest,
    JobResult,
    PrepareModelData,
    PrepareModelParams,
)
from grounded_pipeline.blender.runner import SubprocessBlenderRunner, default_entry_script
from grounded_pipeline.domain.errors import BlenderJobError
from grounded_pipeline.domain.library_config import GeometrySection, UvSection
from grounded_pipeline.domain.ports import CancelToken
from grounded_pipeline.gltf.fingerprint import surface_fingerprint
from grounded_pipeline.gltf.loader import load_glb
from grounded_pipeline.pipeline.hashing import file_sha

pytestmark = pytest.mark.blender

ROOT = Path(__file__).resolve().parents[3]
FIXTURE_DIR = ROOT / "tests" / "fixtures" / "fbx"
BONE_PATHS = ["Armature/Root", "Armature/Root/Spine", "Armature/Root/Spine/Head"]

# Fixture cube: 26 verts / 24 quads. Catmull-Clark on a closed quad mesh: V' = V + E + F with
# E = 2F, so one level gives 26 + 48 + 24 = 98 verts and 96 quads (192 tris).
LEVEL1 = {"vertices": 98, "faces": 96, "triangles": 192}

# A Blender-side check that the saved .blend reopens with the subdivided, still-weighted mesh.
REOPEN_SCRIPT = """
import json, sys
import bpy
mesh_obj = bpy.data.objects["Cube"]
me = mesh_obj.data
out = {
    "vertices": len(me.vertices),
    "faces": len(me.polygons),
    "unweighted": sum(1 for v in me.vertices if not any(g.weight > 0.0 for g in v.groups)),
    "modifiers": [m.type for m in mesh_obj.modifiers],
    "smooth": all(p.use_smooth for p in me.polygons),
    "materials": [m.name for m in me.materials],
    "bones": [b.name for b in bpy.data.objects["Armature"].data.bones],
    "actions": sorted(a.name for a in bpy.data.actions),
}
print("GP_REOPEN_JSON_BEGIN")
print(json.dumps(out))
print("GP_REOPEN_JSON_END")
"""


@pytest.fixture(scope="module")
def runner(blender_exe: Path) -> SubprocessBlenderRunner:
    return SubprocessBlenderRunner(
        blender_exe,
        default_entry_script(),
        threading.BoundedSemaphore(1),
        os.environ,
        poll_interval_s=0.2,
    )


def _params(tmp_path: Path, fixture: str, **sections: dict[str, Any]) -> PrepareModelParams:
    work = tmp_path / "work"
    return PrepareModelParams(
        source_fbx=FIXTURE_DIR / fixture,
        out_blend=work / "prepared.blend",
        out_static_glb=work / "static.glb",
        out_report=work / "prepare_report.json",
        geometry=GeometrySection.model_validate(sections.get("geometry", {})),
        uv=UvSection.model_validate(sections.get("uv", {})),
        material_name="cube_rigged_mat",
        palette_hint_textures=[],
    )


def _run(
    runner: SubprocessBlenderRunner, tmp_path: Path, fixture: str, **sections: dict[str, Any]
) -> tuple[PrepareModelData, JobResult, PrepareModelParams]:
    params = _params(tmp_path, fixture, **sections)
    req = JobRequest(
        job="prepare_model",
        job_id=str(ULID()),
        model_id="cube_rigged",
        params=params.model_dump(mode="json"),
    )
    result = runner.run(req, timeout_s=300, log_path=tmp_path / "prepare.log", cancel=CancelToken())
    assert result.ok, result.error
    return PrepareModelData.model_validate(result.data), result, params


def _reopen(blender_exe: Path, blend: Path, tmp_path: Path) -> dict[str, Any]:
    script = tmp_path / "reopen.py"
    script.write_text(textwrap.dedent(REOPEN_SCRIPT))
    cmd = [str(blender_exe), "-b", str(blend), "--python-exit-code", "2", "--python", str(script)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300, check=False)  # noqa: S603  # trusted local executable
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    payload = proc.stdout.split("GP_REOPEN_JSON_BEGIN")[1].split("GP_REOPEN_JSON_END")[0]
    out: dict[str, Any] = json.loads(payload)
    return out


# --------------------------------------------------------------------------- tests
def test_rigged_cube_level_one(runner: SubprocessBlenderRunner, tmp_path: Path) -> None:
    # The fixture texture is 4x4 px, which the palette heuristic would (correctly) flag; lower
    # the palette threshold so this test exercises the "kept" path on the source UVs.
    data, result, params = _run(
        runner, tmp_path, "rigged_cube.fbx", uv={"palette_texture_max_px": 2}
    )

    assert result.blender_version.startswith("4.5")
    assert data.mesh.object_names == ["Cube"]
    assert (data.mesh.vertices_before, data.mesh.faces_before) == (26, 24)
    assert data.mesh.vertices_after == LEVEL1["vertices"]
    assert data.mesh.faces_after == LEVEL1["faces"]
    assert data.mesh.triangles_after == LEVEL1["triangles"]
    assert data.mesh.shapekey_meshes == []
    assert data.mesh.shapekeys_policy_applied == "none"
    assert (data.mesh.material_slots_before, data.mesh.material_slots_after) == (1, 1)

    assert data.uv.decision == "kept"
    assert data.uv.reason == "uvs usable"
    assert data.uv.uv_layers == ["UVMap"]
    assert data.uv.overlap_ratio == 0.0
    assert data.uv.island_count == 1  # the default cube cross is one connected island
    assert data.uv.source_texture_max_px == 4

    assert data.rig.armature_object == "Armature"
    assert data.rig.armature_data == "Armature"
    assert data.rig.bone_count == 3
    assert [b.name for b in data.rig.bones] == ["Root", "Spine", "Head"]
    assert [b.transform_path for b in data.rig.bones] == BONE_PATHS
    assert [b.parent for b in data.rig.bones] == [None, "Root", "Spine"]
    assert data.rig.bones[1].head == pytest.approx((0.0, 0.0, 0.5))
    assert data.rig.actions == ["idle"]  # "Armature|Armature|idle" with prefixes stripped

    assert params.out_blend.is_file()
    assert params.out_report.is_file()
    assert PrepareModelData.model_validate_json(params.out_report.read_text()) == data
    assert file_sha(params.out_static_glb) == data.static_glb_sha256
    assert data.bbox_diagonal > 0

    doc = load_glb(params.out_static_glb)
    fp = surface_fingerprint(doc, q_pos=1e-4, q_uv=1e-4)
    assert fp.triangle_count == data.static_glb_triangles == LEVEL1["triangles"]
    assert fp.unique_position_count == data.static_glb_unique_positions == LEVEL1["vertices"]
    assert fp.has_uv is True
    assert doc.images == []  # unrigged, untextured static copy

    log = (tmp_path / "prepare.log").read_text()
    assert "[entry] job=prepare_model" in log
    assert "[entry] ok=True exit=0" in log


def test_prepared_blend_reopens_with_weights_intact(
    runner: SubprocessBlenderRunner, blender_exe: Path, tmp_path: Path
) -> None:
    data, _result, params = _run(runner, tmp_path, "rigged_cube.fbx")
    # Default auto mode: the 4x4 fixture texture is a palette → re-unwrapped on the base mesh.
    assert data.uv.decision == "unwrapped"
    assert data.uv.reason == "palette texture 4px"
    reopened = _reopen(blender_exe, params.out_blend, tmp_path)
    assert reopened["vertices"] == LEVEL1["vertices"]
    assert reopened["faces"] == LEVEL1["faces"]
    assert reopened["unweighted"] == 0  # appendix B §3: weights survive subdivision
    assert reopened["modifiers"] == ["ARMATURE"]  # GP_Subdiv applied, Armature kept
    assert reopened["smooth"] is True
    assert reopened["materials"] == ["cube_rigged_mat"]
    assert reopened["bones"] == ["Root", "Spine", "Head"]
    assert reopened["actions"] == ["Armature|Armature|idle"]  # untouched by prepare


def test_palette_cube_is_unwrapped(runner: SubprocessBlenderRunner, tmp_path: Path) -> None:
    data, _result, params = _run(runner, tmp_path, "palette_cube.fbx")
    assert data.uv.decision == "unwrapped"
    assert "palette" in data.uv.reason
    assert data.uv.source_texture_max_px == 16
    assert data.uv.overlap_ratio == pytest.approx(1.0)
    assert data.uv.island_count == 6
    assert data.mesh.faces_after == LEVEL1["faces"]
    fp = surface_fingerprint(load_glb(params.out_static_glb), q_pos=1e-4, q_uv=1e-4)
    assert fp.triangle_count == data.static_glb_triangles
    assert fp.unique_position_count == data.static_glb_unique_positions
    # Re-unwrapped UVs: many points, no longer collapsed onto one swatch.
    uvs = {
        tuple(round(float(c), 3) for c in uv)
        for m in load_glb(params.out_static_glb).meshes
        if m.uvs is not None
        for uv in m.uvs
    }
    assert len(uvs) > 24


def test_shapekey_cube_fail_policy(runner: SubprocessBlenderRunner, tmp_path: Path) -> None:
    with pytest.raises(BlenderJobError) as info:
        _run(runner, tmp_path, "shapekey_cube.fbx")
    assert info.value.job_code == "E_BJ_SHAPEKEYS"
    assert info.value.detail["meshes"] == ["Cube"]


def test_shapekey_cube_drop_policy(runner: SubprocessBlenderRunner, tmp_path: Path) -> None:
    data, result, _params = _run(
        runner, tmp_path, "shapekey_cube.fbx", geometry={"shapekeys": "drop"}
    )
    assert data.mesh.shapekey_meshes == ["Cube"]
    assert data.mesh.shapekeys_policy_applied == "dropped"
    assert data.mesh.faces_after == LEVEL1["faces"]
    assert data.rig.actions == ["idle"]  # the "Key|…" action is not an object action
    assert any("dropped" in w for w in result.warnings)


def test_levels_zero_keeps_counts(runner: SubprocessBlenderRunner, tmp_path: Path) -> None:
    data, _result, params = _run(
        runner, tmp_path, "rigged_cube.fbx", geometry={"subdivision_levels": 0}
    )
    assert (data.mesh.vertices_after, data.mesh.faces_after, data.mesh.triangles_after) == (
        26,
        24,
        48,
    )
    fp = surface_fingerprint(load_glb(params.out_static_glb), q_pos=1e-4, q_uv=1e-4)
    assert (fp.triangle_count, fp.unique_position_count) == (48, 26)
    assert data.static_glb_unique_positions == 26
