"""WP-09 acceptance under a real Blender (plan 12 §1, 16 §WP-09).

One module-scoped pipeline: ``prepare_model`` on ``rigged_cube.fbx`` (levels 0, UVs kept) →
``export_rigged`` with synthetic URP PNGs → ``roundtrip_animation`` on ``rigged_cube@walk.fbx``
→ Rig Fidelity Check in both modes → a second Blender run re-imports both exported files through
``tools/fixture_probe.py`` and pins what Unity will see: identical bone names/order/paths and rest
pose (appendix B §8.3), the actions, no ``_end`` leaf bones, the material name with its image
references, and unit armature scale (``FBX_SCALE_ALL``).
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
from PIL import Image
from ulid import ULID

from grounded_pipeline.blender.protocol import (
    ExportRiggedData,
    ExportRiggedParams,
    JobName,
    JobRequest,
    PrepareModelData,
    PrepareModelParams,
    RoundtripAnimationData,
    RoundtripAnimationParams,
)
from grounded_pipeline.blender.runner import SubprocessBlenderRunner, default_entry_script
from grounded_pipeline.domain.library_config import ExportSection, GeometrySection, UvSection
from grounded_pipeline.domain.ports import CancelToken
from grounded_pipeline.pipeline.hashing import file_sha
from grounded_pipeline.pipeline.rig_fidelity import RfcTolerances, rig_fidelity_check

pytestmark = pytest.mark.blender

ROOT = Path(__file__).resolve().parents[3]
FIXTURE_DIR = ROOT / "tests" / "fixtures" / "fbx"
TOOLS_DIR = ROOT / "tools"
MODEL = "cube_rigged"
MATERIAL = f"{MODEL}_mat"
BONES = ["Root", "Spine", "Head"]
BONE_PATHS = ["Armature/Root", "Armature/Root/Spine", "Armature/Root/Spine/Head"]
EXPECTED_RIG = {  # same table as test_fixtures_load.py: rest pose must survive the round trip
    "bones": BONES,
    "bone_parents": {"Root": None, "Spine": "Root", "Head": "Spine"},
    "bone_heads": {"Root": [0.0, 0.0, 0.0], "Spine": [0.0, 0.0, 0.5], "Head": [0.0, 0.0, 1.0]},
    "bone_tails": {"Root": [0.0, 0.0, 0.5], "Spine": [0.0, 0.0, 1.0], "Head": [0.0, 0.0, 1.5]},
}
URP_PNGS = (f"{MODEL}_basecolor.png", f"{MODEL}_normal.png", f"{MODEL}_metallic_smoothness.png")

# Re-import both exported files with the fixture probe (same FBX_IMPORT_KWARGS as the pipeline)
# and add what the probe does not report: object scale and the material → image wiring.
PROBE_SCRIPT = """
import json
import sys
from pathlib import Path

sys.path.insert(0, {tools_dir!r})
import bpy
import fixture_probe

out = {{}}
for raw in {paths!r}:
    path = Path(raw)
    summary = fixture_probe.summarize_fbx(path)
    summary["object_scales"] = {{
        o.name: [round(float(v), 4) for v in o.matrix_world.to_scale()] for o in bpy.data.objects
    }}
    summary["material_images"] = {{
        m.name: sorted(
            Path(bpy.path.abspath(n.image.filepath)).name
            for n in m.node_tree.nodes
            if n.type == "TEX_IMAGE" and n.image is not None
        )
        for m in bpy.data.materials
        if m.use_nodes and m.node_tree is not None
    }}
    out[path.name] = summary
Path({out!r}).write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
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


def _run(runner: SubprocessBlenderRunner, job: JobName, params: Any, log: Path) -> Any:
    req = JobRequest(
        job=job, job_id=str(ULID()), model_id=MODEL, params=params.model_dump(mode="json")
    )
    result = runner.run(req, timeout_s=300, log_path=log, cancel=CancelToken())
    assert result.ok, result.error
    return result


def _write_urp_pngs(urp: Path) -> None:
    urp.mkdir(parents=True, exist_ok=True)
    Image.new("RGBA", (2, 2), (180, 120, 60, 255)).save(urp / URP_PNGS[0])
    Image.new("RGB", (2, 2), (128, 128, 255)).save(urp / URP_PNGS[1])
    Image.new("RGBA", (2, 2), (255, 0, 0, 64)).save(urp / URP_PNGS[2])


def _probe(blender_exe: Path, files: list[Path], tmp: Path) -> dict[str, Any]:
    script = tmp / "probe_exports.py"
    out = tmp / "probe_exports.json"
    script.write_text(
        textwrap.dedent(
            PROBE_SCRIPT.format(
                tools_dir=str(TOOLS_DIR), paths=[str(f) for f in files], out=str(out)
            )
        ),
        encoding="utf-8",
    )
    cmd = [
        str(blender_exe),
        "-b",
        "--factory-startup",
        "--python-exit-code",
        "2",
        "--python",
        str(script),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300, check=False)  # noqa: S603  # trusted local executable + repo paths
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-4000:]
    summaries: dict[str, Any] = json.loads(out.read_text(encoding="utf-8"))
    return summaries


@pytest.fixture(scope="module")
def pipeline(
    runner: SubprocessBlenderRunner, blender_exe: Path, tmp_path_factory: pytest.TempPathFactory
) -> dict[str, Any]:
    tmp = tmp_path_factory.mktemp("export")
    work = tmp / "work"
    log = tmp / "jobs.log"

    prepare_params = PrepareModelParams(
        source_fbx=FIXTURE_DIR / "rigged_cube.fbx",
        out_blend=work / "prepared.blend",
        out_static_glb=work / "static.glb",
        out_report=work / "prepare_report.json",
        geometry=GeometrySection(subdivision_levels=0),
        uv=UvSection(palette_texture_max_px=2),  # the 4x4 fixture texture is not a palette here
        material_name=MATERIAL,
    )
    prepare = PrepareModelData.model_validate(
        _run(runner, "prepare_model", prepare_params, log).data
    )

    _write_urp_pngs(work / "textures" / "urp")
    export_params = ExportRiggedParams(
        prepared_blend=work / "prepared.blend",
        textures_dir=work / "textures" / "urp",
        out_fbx=work / "export" / f"{MODEL}.fbx",
        out_rig_fingerprint=work / "export" / "rig_fingerprint_out.json",
        export=ExportSection(),
        material_name=MATERIAL,
    )
    export_result = _run(runner, "export_rigged", export_params, log)
    export = ExportRiggedData.model_validate(export_result.data)

    clip_params = RoundtripAnimationParams(
        source_fbx=FIXTURE_DIR / "rigged_cube@walk.fbx",
        out_fbx=work / "export" / "animations" / f"{MODEL}@walk.fbx",
        export=ExportSection(),
        expected_bone_names=[b.name for b in prepare.rig.bones],
    )
    clip_result = _run(runner, "roundtrip_animation", clip_params, log)
    clip = RoundtripAnimationData.model_validate(clip_result.data)

    probe = _probe(blender_exe, [export_params.out_fbx, clip_params.out_fbx], tmp)
    return {
        "prepare": prepare,
        "export": export,
        "export_warnings": export_result.warnings,
        "clip": clip,
        "clip_warnings": clip_result.warnings,
        "export_params": export_params,
        "clip_params": clip_params,
        "probe": probe,
        "log": log,
    }


# --------------------------------------------------------------------------- job results
def test_export_rig_equals_prepared_rig(pipeline: dict[str, Any]) -> None:
    prepare: PrepareModelData = pipeline["prepare"]
    export: ExportRiggedData = pipeline["export"]
    params: ExportRiggedParams = pipeline["export_params"]
    assert prepare.mesh.faces_after == 24  # levels 0: the fixture cube unchanged
    assert prepare.uv.decision == "kept"
    assert export.rig == prepare.rig
    assert export.rig.fingerprint_sha256 == prepare.rig.fingerprint_sha256
    assert [b.transform_path for b in export.rig.bones] == BONE_PATHS
    assert export.actions_exported == ["idle"]
    assert export.rig.actions == ["idle"]
    assert pipeline["export_warnings"] == []
    assert params.out_fbx.is_file()
    assert file_sha(params.out_fbx) == export.fbx_sha256
    written = json.loads(params.out_rig_fingerprint.read_text(encoding="utf-8"))
    assert written == export.rig.model_dump(mode="json")


def test_roundtrip_clip_rig_matches_and_keeps_its_action(pipeline: dict[str, Any]) -> None:
    prepare: PrepareModelData = pipeline["prepare"]
    clip: RoundtripAnimationData = pipeline["clip"]
    params: RoundtripAnimationParams = pipeline["clip_params"]
    assert [b.name for b in clip.rig.bones] == BONES
    assert [b.transform_path for b in clip.rig.bones] == BONE_PATHS
    assert clip.missing_bones == []
    assert clip.extra_bones == []
    assert clip.actions_exported == ["walk"]
    assert clip.rig.actions == ["walk"]
    assert pipeline["clip_warnings"] == []
    assert params.out_fbx.is_file()
    for ours, theirs in zip(prepare.rig.bones, clip.rig.bones, strict=True):
        assert ours.head == pytest.approx(theirs.head, abs=1e-5)
        assert ours.tail == pytest.approx(theirs.tail, abs=1e-5)
        for row_a, row_b in zip(ours.matrix_local, theirs.matrix_local, strict=True):
            assert row_a == pytest.approx(row_b, abs=1e-5)


def test_rig_fidelity_check_passes_in_both_modes(pipeline: dict[str, Any]) -> None:
    prepare: PrepareModelData = pipeline["prepare"]
    tol = RfcTolerances()
    model = rig_fidelity_check(prepare.rig, pipeline["export"].rig, tol, "model")
    assert model.overall == "pass", model.messages("fail") + model.messages("warn")
    anim = rig_fidelity_check(prepare.rig, pipeline["clip"].rig, tol, "animation")
    assert anim.overall == "pass", anim.messages("fail") + anim.messages("warn")


# --------------------------------------------------------------------------- re-import in Blender
def test_reimported_model_has_identical_rig_actions_material_and_scale(
    pipeline: dict[str, Any],
) -> None:
    summary = pipeline["probe"][f"{MODEL}.fbx"]
    assert summary["objects"] == [["Armature", "ARMATURE"], ["Cube", "MESH"]]
    # Bone names, order, parents and rest pose identical to the vendor file (B §8.3 honoured).
    assert summary["armatures"] == {"Armature": EXPECTED_RIG}
    assert not any(name.endswith("_end") for name in summary["armatures"]["Armature"]["bones"])
    # The one action survives bake_anim_use_all_actions with the pose reset before export, and
    # the take is "Armature|idle" like the vendor's (no prefix pile-up; appendix B §8.1).
    assert summary["actions"] == {
        "Armature|Armature|idle": {"frame_range": [1.0, 24.0], "fcurves": 39}
    }
    cube = summary["meshes"]["Cube"]
    assert cube["materials"] == [MATERIAL]
    assert (cube["vertices"], cube["faces"], cube["weighted_vertices"]) == (26, 24, 26)
    assert cube["armature_modifier_targets"] == ["Armature"]
    assert cube["uv_layers"] == ["UVMap"]
    # URP maps referenced by the material (path_mode=RELATIVE, resolved on import). Only images
    # wired straight into Principled inputs become FBX texture slots: basecolor (twice — the
    # exporter also writes an RGBA base colour as the transparency texture) and the normal map.
    # Metallic/roughness sit behind SeparateColor/Invert (appendix B §6) and are not exported;
    # Unity builds the material from the manifest anyway (plan 12 §4.2).
    assert set(summary["material_images"][MATERIAL]) == {URP_PNGS[0], URP_PNGS[1]}
    found = {img["filename"] for img in summary["images"].values() if img["found"]}
    assert found == {URP_PNGS[0], URP_PNGS[1]}
    # FBX_SCALE_ALL: unit scale on every object after re-import (no x100 armature).
    assert summary["object_scales"] == {"Armature": [1.0, 1.0, 1.0], "Cube": [1.0, 1.0, 1.0]}


def test_reimported_clip_has_identical_rig_and_only_its_action(pipeline: dict[str, Any]) -> None:
    summary = pipeline["probe"][f"{MODEL}@walk.fbx"]
    assert summary["objects"] == [["Armature", "ARMATURE"]]
    assert summary["armatures"] == {"Armature": EXPECTED_RIG}
    assert not any(name.endswith("_end") for name in summary["armatures"]["Armature"]["bones"])
    assert summary["actions"] == {
        "Armature|Armature|walk": {"frame_range": [1.0, 24.0], "fcurves": 39}
    }
    assert summary["meshes"] == {}
    assert summary["images"] == {}
    assert summary["object_scales"] == {"Armature": [1.0, 1.0, 1.0]}


def test_job_log_mentions_both_jobs(pipeline: dict[str, Any]) -> None:
    log = Path(pipeline["log"]).read_text(encoding="utf-8")
    assert "[entry] job=export_rigged" in log
    assert "[entry] job=roundtrip_animation" in log
    assert log.count("[entry] ok=True exit=0") == 3
