"""Credit-free end-to-end run (plan 01 success criteria 2, 3 and 5; plan 16 §WP-17).

``discover → prepare → texture → verify → apply → export → handoff`` is driven through the real
``gp`` CLI (``build_app`` + ``CliRunner``), the real :class:`PipelineRunner`, the real production
context (``build_production_context`` → ``SubprocessBlenderRunner`` + ``JsonStateStore``) and real
Blender jobs on a temporary copy of ``tests/fixtures/library_minimal``. Only the texture provider
is replaced: a :class:`FakeTextureProvider` whose GLB factory re-emits the geometry *it was sent*
(``work/static.glb``) with a known-value PBR texture set, so no Tripo credit is ever spent.

Two provider behaviours are simulated:

* ``extract`` — identical positions and UVs (the provider kept our unwrap);
* ``bake`` — identical positions returned at ``normalize_scale = 2`` with re-unwrapped UVs (the
  provider's documented normalisation + re-unwrap). ``verify`` undoes the scale from
  ``task.json`` (``source=task``), sees matching positions but different UVs and picks ``bake``;
  the real ``bake_textures`` job then aligns the source onto the prepared mesh by bounding box.
  No ``allow_geometry_change`` override is needed. ``[bake]`` is lowered to 512 px / 4 samples.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fakes.fake_texture_provider import FakeTextureProvider
from grounded_pipeline.domain.manifest import Manifest, json_utility_violations
from grounded_pipeline.domain.state import STAGE_ORDER

from .e2e_helpers import (
    BAKE_RUN_JOBS,
    BASECOLOR_RGB,
    CLIP,
    COLOUR_TOLERANCE,
    CREDITS,
    E2E,
    FAST_BAKE_TOML,
    FULL_RUN_JOBS,
    MODEL,
    PROVIDER_SCALE,
    URP_MAPS,
    copy_library,
    glb_scaled_reunwrapped,
    glb_textured_copy,
    median_rgb,
)

pytestmark = pytest.mark.blender


# --------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def extract_e2e(blender_exe: Path, tmp_path_factory: pytest.TempPathFactory) -> E2E:
    """The extract-route library after its first full run (shared by tests 1, 3, 4 and 6)."""
    lib = copy_library(tmp_path_factory.mktemp("e2e_extract") / "library_minimal")
    provider = FakeTextureProvider(result_glb_factory=glb_textured_copy, credits=CREDITS)
    e2e = E2E.create(lib, blender_exe, provider)
    e2e.first = e2e.run()
    e2e.first_jobs = list(e2e.jobs)
    return e2e


# --------------------------------------------------------------------------- tests
def test_full_pipeline_extract_route(extract_e2e: E2E) -> None:
    e2e, document, folder = extract_e2e, extract_e2e.first, extract_e2e.folder
    assert e2e.first_jobs == FULL_RUN_JOBS
    assert e2e.provider.calls[0].resume_task_id is None
    assert document["credits_total"] == CREDITS
    assert document["models"][0]["result"] == "succeeded"

    report = e2e.report(document)
    assert report["exit_code"] == 0
    assert report["credits_total"] == CREDITS
    assert e2e.report_stages(document) == dict.fromkeys(STAGE_ORDER, "succeeded")
    assert e2e.stage_statuses() == dict.fromkeys(STAGE_ORDER, "succeeded")

    verify = e2e.work_json("work/verify_report.json")
    assert verify["route"] == "extract"
    assert verify["position_match"] is True
    assert verify["uv_match"] is True
    assert verify["forced"] is False
    assert e2e.work_json("work/apply_report.json")["route"] == "extract"
    for name in URP_MAPS:
        assert (folder / "work" / "textures" / "urp" / f"{MODEL}_{name}.png").is_file()
    assert (folder / "work" / "export" / f"{MODEL}.fbx").is_file()
    assert (folder / "work" / "export" / "animations" / f"{MODEL}@{CLIP}.fbx").is_file()
    assert e2e.work_json("work/export/rig_fidelity_report.json")["overall"] == "pass"

    out = folder / "output" / MODEL
    assert (out / f"{MODEL}.fbx").is_file()
    assert (out / "animations" / f"{MODEL}@{CLIP}.fbx").is_file()
    delivered = sorted(p.name for p in (out / "textures").iterdir())
    assert delivered == sorted(f"{MODEL}_{name}.png" for name in URP_MAPS)
    assert sorted(p.name for p in out.iterdir()) == [
        "animations",
        f"{MODEL}.fbx",
        "manifest.json",
        "textures",
    ]

    manifest = e2e.manifest()
    Manifest.model_validate(manifest)
    assert json_utility_violations(manifest) == []
    assert manifest["model_id"] == MODEL
    assert manifest["fbx"] == f"{MODEL}.fbx"
    assert manifest["rig"]["root_bone"] == "Root"
    (material,) = manifest["materials"]
    assert material["name"] == f"{MODEL}_mat"
    assert sorted(t["file"] for t in material["textures"]) == [
        f"textures/{name}" for name in delivered
    ]
    assert manifest["animations"] == [{"file": f"animations/{MODEL}@{CLIP}.fbx", "clip_name": CLIP}]
    provenance = manifest["provenance"]
    assert provenance["texture_route"] == "extract"
    assert provenance["rig_fidelity"] == "pass"
    assert provenance["texture_task_id"] == e2e.provider.calls[0].task_id
    assert provenance["credits_consumed"] == CREDITS
    assert provenance["subdivision_levels"] == 1
    assert provenance["uv_decision"] == "unwrapped"  # 4x4 fixture texture is a palette
    assert len(provenance["prompt_sha256"]) == 64


def test_full_pipeline_bake_route(
    blender_exe: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    lib = copy_library(
        tmp_path_factory.mktemp("e2e_bake") / "library_minimal", extra_toml=FAST_BAKE_TOML
    )
    provider = FakeTextureProvider(
        result_glb_factory=glb_scaled_reunwrapped,
        credits=CREDITS,
        normalize_scale=PROVIDER_SCALE,
        normalize_translation=(0.0, 0.0, 0.0),
    )
    e2e = E2E.create(lib, blender_exe, provider)
    document = e2e.run()

    assert e2e.jobs == BAKE_RUN_JOBS
    assert e2e.report_stages(document) == dict.fromkeys(STAGE_ORDER, "succeeded")
    verify = e2e.work_json("work/verify_report.json")
    assert verify["route"] == "bake"
    assert verify["position_match"] is True
    assert verify["uv_match"] is False
    assert "source=task" in verify["reason"]
    apply = e2e.work_json("work/apply_report.json")
    assert (apply["route"], apply["source"], apply["fallback_to_bake"]) == ("bake", "bake", False)
    assert apply["resolution"] == [512, 512]
    raw_basecolor = e2e.folder / "work" / "textures" / "raw" / "basecolor.png"
    assert raw_basecolor.is_file()
    assert median_rgb(raw_basecolor) == pytest.approx(BASECOLOR_RGB, abs=COLOUR_TOLERANCE)
    manifest = e2e.manifest()
    assert manifest["provenance"]["texture_route"] == "bake"
    assert manifest["provenance"]["rig_fidelity"] == "pass"
    assert (e2e.folder / "output" / MODEL / "textures" / f"{MODEL}_basecolor.png").is_file()


def test_second_run_is_fully_cached(extract_e2e: E2E) -> None:
    e2e = extract_e2e
    calls, jobs, logs = len(e2e.provider.calls), list(e2e.jobs), e2e.log_files()
    manifest_before = e2e.manifest()

    document = e2e.run()

    assert e2e.report_stages(document) == dict.fromkeys(STAGE_ORDER, "skipped")
    assert document["models"][0]["result"] == "skipped"
    assert document["credits_total"] == 0.0
    assert len(e2e.provider.calls) == calls, "provider must not be called again"
    assert e2e.jobs == jobs, "no Blender job may be launched"
    assert e2e.log_files() == logs
    assert e2e.manifest() == manifest_before
    assert e2e.stage_statuses() == dict.fromkeys(STAGE_ORDER, "succeeded")


def test_prompt_edit_reruns_only_texture_and_downstream(extract_e2e: E2E) -> None:
    e2e = extract_e2e
    calls, jobs = len(e2e.provider.calls), list(e2e.jobs)
    before = e2e.manifest()["provenance"]
    model_md = e2e.folder / "model.md"
    text = model_md.read_text(encoding="utf-8")
    edited = text.replace("A small wooden crate cube", "A small weathered wooden crate cube")
    assert edited != text
    model_md.write_text(edited, encoding="utf-8")

    document = e2e.run()

    stages = e2e.report_stages(document)
    assert stages == {
        "discover": "succeeded",
        "prepare": "skipped",
        "texture": "succeeded",
        "verify": "succeeded",
        "apply": "succeeded",
        "export": "succeeded",
        "handoff": "succeeded",
    }
    assert e2e.jobs == [*jobs, "export_rigged", "roundtrip_animation"]  # no prepare_model
    assert len(e2e.provider.calls) == calls + 1
    assert e2e.provider.last_call.resume_task_id is None  # request changed → fresh task
    assert document["credits_total"] == CREDITS
    after = e2e.manifest()["provenance"]
    assert after["prompt_sha256"] != before["prompt_sha256"]
    assert after["texture_task_id"] == e2e.provider.last_call.task_id
    assert after["texture_task_id"] != before["texture_task_id"]
    assert after["texture_route"] == "extract"
    assert e2e.stage_statuses() == dict.fromkeys(STAGE_ORDER, "succeeded")


def test_dry_run_estimates_credits_without_work(blender_exe: Path, tmp_path: Path) -> None:
    lib = copy_library(tmp_path / "library_minimal")
    provider = FakeTextureProvider(result_glb_factory=glb_textured_copy, credits=CREDITS)
    e2e = E2E.create(lib, blender_exe, provider)

    result, document = e2e.gp("run", "--model", MODEL, "--dry-run")

    assert result.exit_code == 0, document
    assert document["dry_run"] is True
    assert document["estimated_credits"] > 0
    assert document["stage_runs"] == len(STAGE_ORDER)
    assert {row["action"] for row in document["rows"]} == {"run"}
    assert "TOTAL estimated credits" in e2e.console.text
    assert e2e.jobs == []
    assert e2e.provider.calls == []
    assert not (e2e.folder / "work").exists()
    assert not (e2e.folder / "output").exists()
    assert not (e2e.folder / ".pipeline" / "state.json").exists()


def test_status_shows_every_stage_done(extract_e2e: E2E) -> None:
    e2e = extract_e2e
    _, document = e2e.gp("status", "--model", MODEL)
    (model,) = document["models"]
    assert model["stages"] == dict.fromkeys(STAGE_ORDER, "succeeded")
    assert model["last_error"] is None
    assert model["credits_total"] >= CREDITS
    before = e2e.console.text
    result = e2e.runner.invoke(
        e2e.app, ["--no-color", "status", "--library", str(e2e.lib), "--model", MODEL]
    )
    assert result.exit_code == 0
    table = e2e.console.text[len(before) :].split("stages:")[0]  # before the legend lines
    assert MODEL in table
    assert table.count("✓") == len(STAGE_ORDER)
    assert not {"✗", "○", "⟳"} & set(table)
