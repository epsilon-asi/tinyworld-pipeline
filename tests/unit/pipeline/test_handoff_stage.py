"""``HandoffStage`` (plan 12 §2-§3): golden manifest, atomic swap, Unity sync, JsonUtility lint."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import structlog

from fakes.fake_blender_runner import sample_prepare_data, sample_rig
from tinyworld_pipeline.blender.protocol import RigFingerprint
from tinyworld_pipeline.domain.errors import HandoffError, StagePrereqError
from tinyworld_pipeline.domain.library_config import LibraryConfig
from tinyworld_pipeline.domain.manifest import Manifest, json_utility_violations
from tinyworld_pipeline.domain.model_spec import (
    ModelDocument,
    ModelFrontMatter,
    ResolvedModel,
    SourceSpec,
    UnityOverrides,
)
from tinyworld_pipeline.domain.ports import CancelToken
from tinyworld_pipeline.domain.prompt import ComposedPrompt
from tinyworld_pipeline.domain.style_spec import StyleDocument, StyleFrontMatter
from tinyworld_pipeline.library.paths import ModelPaths
from tinyworld_pipeline.pipeline import hashing
from tinyworld_pipeline.pipeline.context import ModelContext, PipelineContext
from tinyworld_pipeline.pipeline.options import RunOptions
from tinyworld_pipeline.pipeline.rig_fidelity import RfcTolerances, rig_fidelity_check
from tinyworld_pipeline.pipeline.stages.export import (
    RIG_FIDELITY_REPORT,
    RIG_FINGERPRINT_OUT,
    ClipRecord,
    ExportReport,
)
from tinyworld_pipeline.pipeline.stages.handoff import (
    HandoffStage,
    iso_utc,
    read_apply_material,
    read_output_rig,
    read_route,
    read_texture_task,
    read_uv_decision,
    utc_now,
)
from tinyworld_pipeline.unity.manifest_writer import UrpMaterialInputs

ROOT = Path(__file__).resolve().parents[3]
GOLDEN = ROOT / "tests" / "fixtures" / "manifest_golden.json"
RUN_ID = "01ARZ3NDEKTSV4RRFFQ69G5FAV"
MODEL = "cube_rigged"
FIXED_NOW = datetime(2026, 11, 2, 12, 0, 0, tzinfo=UTC)
MAPS = ("basecolor", "normal", "metallic_smoothness")


# --------------------------------------------------------------------------- synthetic model
def make_model(
    tmp_path: Path,
    *,
    config: LibraryConfig | None = None,
    source_asset_path: str | None = "Assets/Vendor/Cubes/cube_rigged.fbx",
) -> ModelContext:
    """A ``cube_rigged`` whose spec values match ``tests/fixtures/manifest_golden.json``."""
    models_dir = tmp_path / "library" / "models"
    folder = models_dir / MODEL
    (folder / "source" / "animations").mkdir(parents=True, exist_ok=True)
    primary = folder / "source" / "cube_rigged.fbx"
    primary.write_bytes(b"FBX-source")
    companion = folder / "source" / "animations" / "cube_rigged@walk.fbx"
    companion.write_bytes(b"FBX-walk")
    front = ModelFrontMatter(
        display_name="Rigged Cube",
        source=SourceSpec(
            primary=Path("source/cube_rigged.fbx"), license="CC0 — synthetic fixture"
        ),
        unity=UnityOverrides(source_asset_path=source_asset_path) if source_asset_path else None,
    )
    doc = ModelDocument(model_id=MODEL, folder=folder, front=front, sections={"prompt": "crate"})
    style = StyleDocument(
        folder=tmp_path / "library" / "style",
        front=StyleFrontMatter(name="style"),
        sections={"base_prompt": "base"},
    )
    resolved = ResolvedModel(
        doc=doc,
        config=config or LibraryConfig.model_validate({"texture": {"seed": 4242}}),
        primary_fbx=primary,
        animation_fbx=[companion],
        style=style,
        composed_prompt=ComposedPrompt(text="p", sources={}, length=1, sha256="a7" * 32),
    )
    return ModelContext(
        resolved=resolved, paths=ModelPaths.for_model(models_dir, MODEL), log=structlog.get_logger()
    )


def make_ctx(pipeline_version: str = "0.1.0") -> PipelineContext:
    return PipelineContext(
        library=MagicMock(),
        options=RunOptions(),
        run_id=RUN_ID,
        blender=MagicMock(),
        provider=MagicMock(),
        state=MagicMock(),
        budget=MagicMock(),
        cancel=CancelToken(),
        clock=MagicMock(),
        sleeper=MagicMock(),
        log=structlog.get_logger(),
        pipeline_version=pipeline_version,
        env={},
        cli_overrides={},
    )


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_reports(
    model: ModelContext,
    *,
    maps: tuple[str, ...] = MAPS,
    apply: bool = True,
    clips: bool = True,
    overall: str = "pass",
) -> None:
    """The synthetic report set behind the golden manifest."""
    folder = model.paths.folder
    work = model.paths.work
    write_json(
        work / "prepare_report.json",
        json.loads(sample_prepare_data(static_glb_sha256="0" * 64).model_dump_json()),
    )
    write_json(work / "verify_report.json", {"route": "extract", "reason": "uvs preserved"})
    write_json(
        work / "tripo" / "task.json",
        {"task_id": "task_cube_rigged_0001", "credits_consumed": 20.0, "provider": "tripo"},
    )
    if apply:
        urp: dict[str, str] = {}
        for name in maps:
            rel = f"work/textures/urp/{MODEL}_{name}.png"
            (folder / rel).parent.mkdir(parents=True, exist_ok=True)
            (folder / rel).write_bytes(b"\x89PNG" + name.encode())
            urp[name] = rel
        write_json(
            work / "apply_report.json",
            {
                "route": "extract",
                "urp": urp,
                "material": {
                    "bump_scale": 1.0,
                    "occlusion_strength": 1.0,
                    "emission_color": [0.0, 0.0, 0.0],
                    "metallic_fallback": 0.0,
                    "smoothness_fallback": 1.0,
                },
            },
        )
    fbx = work / "export" / f"{MODEL}.fbx"
    fbx.parent.mkdir(parents=True, exist_ok=True)
    fbx.write_bytes(b"FBX-export")
    rig = sample_rig()
    write_json(folder / RIG_FINGERPRINT_OUT, json.loads(rig.model_dump_json()))
    rfc = rig_fidelity_check(rig, rig, RfcTolerances(), "model")
    records: list[ClipRecord] = []
    if clips:
        clip = work / "export" / "animations" / f"{MODEL}@walk.fbx"
        clip.parent.mkdir(parents=True, exist_ok=True)
        clip.write_bytes(b"FBX-walk-roundtrip")
        records.append(
            ClipRecord(
                clip_name="walk",
                file=f"work/export/animations/{MODEL}@walk.fbx",
                source="source/animations/cube_rigged@walk.fbx",
                actions_exported=["walk"],
                missing_bones=[],
                extra_bones=[],
                report=rig_fidelity_check(rig, rig, RfcTolerances(), "animation"),
            )
        )
    report = ExportReport(
        overall=overall,
        fbx=f"work/export/{MODEL}.fbx",
        fbx_sha256=hashing.file_sha(fbx),
        actions_exported=["idle"],
        model=rfc,
        clips=records,
    )
    write_json(folder / RIG_FIDELITY_REPORT, json.loads(report.model_dump_json()))


def golden_manifest(model: ModelContext, **overrides: Any) -> dict[str, Any]:
    """The golden document with the one value that cannot be synthesised: the source sha."""
    doc: dict[str, Any] = json.loads(GOLDEN.read_text(encoding="utf-8"))
    doc["provenance"]["source_fbx_sha256"] = hashing.file_sha(model.primary_fbx)
    doc["provenance"].update(overrides)
    return doc


@pytest.fixture(autouse=True)
def _clear_hash_caches() -> None:
    hashing.clear_caches()


@pytest.fixture
def model(tmp_path: Path) -> ModelContext:
    m = make_model(tmp_path)
    write_reports(m)
    return m


@pytest.fixture
def stage() -> HandoffStage:
    return HandoffStage(now=lambda: FIXED_NOW)


# --------------------------------------------------------------------------- golden
def test_golden_manifest_and_output_layout(model: ModelContext, stage: HandoffStage) -> None:
    outcome = stage.run(model, make_ctx())

    out = model.paths.output / MODEL
    manifest_path = out / "manifest.json"
    text = manifest_path.read_text(encoding="utf-8")
    assert json.loads(text) == golden_manifest(model)
    assert Manifest.model_validate_json(text).provenance.rig_fidelity == "pass"
    assert json_utility_violations(json.loads(text)) == []
    # Field order and formatting are the writer's (schema order, indent 2, trailing newline).
    assert list(json.loads(text)) == list(Manifest.model_fields)
    assert text.endswith("\n")

    expected_files = [
        f"output/{MODEL}/animations/{MODEL}@walk.fbx",
        f"output/{MODEL}/{MODEL}.fbx",
        f"output/{MODEL}/manifest.json",
        f"output/{MODEL}/textures/{MODEL}_basecolor.png",
        f"output/{MODEL}/textures/{MODEL}_metallic_smoothness.png",
        f"output/{MODEL}/textures/{MODEL}_normal.png",
    ]
    assert outcome.status == "succeeded"
    assert outcome.outputs == expected_files
    assert outcome.warnings == []
    assert (out / f"{MODEL}.fbx").read_bytes() == b"FBX-export"
    assert (out / "animations" / f"{MODEL}@walk.fbx").read_bytes() == b"FBX-walk-roundtrip"
    assert (out / "textures" / f"{MODEL}_normal.png").read_bytes() == b"\x89PNGnormal"
    total = sum((model.paths.folder / rel).stat().st_size for rel in expected_files)
    assert outcome.extra == {
        "output_bytes": total,
        "synced": False,
        "manifest": f"output/{MODEL}/manifest.json",
    }
    assert not any(p.name.startswith(f"{MODEL}.tmp-") for p in model.paths.output.iterdir())
    assert not any(p.name.startswith(f"{MODEL}.old-") for p in model.paths.output.iterdir())


def test_manifest_reflects_rfc_warnings_pipeline_version_and_missing_avatar_path(
    tmp_path: Path, stage: HandoffStage
) -> None:
    model = make_model(tmp_path, source_asset_path=None)
    write_reports(model, overall="pass_with_warnings")
    stage.run(model, make_ctx(pipeline_version="9.9.9"))
    doc = json.loads((model.paths.output / MODEL / "manifest.json").read_text(encoding="utf-8"))
    assert doc["pipeline_version"] == "9.9.9"
    assert doc["generated_at"] == "2026-11-02T12:00:00Z"
    assert doc["rig"]["source_avatar_asset_path"] == ""
    assert doc["rig"]["root_bone"] == "Root"
    assert doc["provenance"]["rig_fidelity"] == "pass_with_warnings"
    assert json_utility_violations(doc) == []


def test_without_apply_report_delivers_no_textures_with_a_warning(
    tmp_path: Path, stage: HandoffStage
) -> None:
    model = make_model(tmp_path)
    write_reports(model, apply=False, clips=False)
    outcome = stage.run(model, make_ctx())
    assert outcome.warnings == ["work/apply_report.json not found; delivering without textures"]
    doc = json.loads((model.paths.output / MODEL / "manifest.json").read_text(encoding="utf-8"))
    assert doc["materials"][0]["textures"] == []
    assert doc["materials"][0]["keywords"] == []
    assert doc["animations"] == []
    assert {f["name"] for f in doc["materials"][0]["floats"]} == {
        "_WorkflowMode",
        "_Surface",
        "_Metallic",
        "_Smoothness",
    }
    assert outcome.outputs == [f"output/{MODEL}/{MODEL}.fbx", f"output/{MODEL}/manifest.json"]


# --------------------------------------------------------------------------- atomic swap
def test_failure_mid_copy_keeps_previous_output(model: ModelContext, stage: HandoffStage) -> None:
    previous = model.paths.output / MODEL
    previous.mkdir(parents=True)
    (previous / "old.txt").write_text("keep me", encoding="utf-8")
    (model.paths.folder / "work" / "textures" / "urp" / f"{MODEL}_normal.png").unlink()

    with pytest.raises(HandoffError, match="delivered file is missing"):
        stage.run(model, make_ctx())

    assert (previous / "old.txt").read_text(encoding="utf-8") == "keep me"
    assert sorted(p.name for p in model.paths.output.iterdir()) == [MODEL]


def test_success_replaces_previous_output_entirely(
    model: ModelContext, stage: HandoffStage
) -> None:
    previous = model.paths.output / MODEL
    previous.mkdir(parents=True)
    (previous / "old.txt").write_text("stale", encoding="utf-8")
    stage.run(model, make_ctx())
    assert not (previous / "old.txt").exists()
    assert (previous / "manifest.json").is_file()
    assert sorted(p.name for p in model.paths.output.iterdir()) == [MODEL]


def test_run_is_idempotent(model: ModelContext, stage: HandoffStage) -> None:
    first = stage.run(model, make_ctx())
    second = stage.run(model, make_ctx())
    assert first == second


# --------------------------------------------------------------------------- unity sync
def test_sync_invoked_when_unity_import_dir_is_set(tmp_path: Path, stage: HandoffStage) -> None:
    unity_dir = tmp_path / "UnityProject" / "Assets" / "Art"
    unity_dir.mkdir(parents=True)
    config = LibraryConfig.model_validate(
        {"texture": {"seed": 4242}, "library": {"unity_import_dir": str(unity_dir)}}
    )
    model = make_model(tmp_path, config=config)
    write_reports(model)
    # A previous Unity import with a .meta that must survive the re-sync (GUID stability).
    old = unity_dir / MODEL
    old.mkdir()
    (old / f"{MODEL}.fbx").write_bytes(b"older")
    (old / f"{MODEL}.fbx.meta").write_text("guid: 1234", encoding="utf-8")

    outcome = stage.run(model, make_ctx())

    assert outcome.extra["synced"] is True
    synced = unity_dir / MODEL
    assert (synced / f"{MODEL}.fbx").read_bytes() == b"FBX-export"
    assert (synced / f"{MODEL}.fbx.meta").read_text(encoding="utf-8") == "guid: 1234"
    assert (synced / "manifest.json").read_text(encoding="utf-8") == (
        model.paths.output / MODEL / "manifest.json"
    ).read_text(encoding="utf-8")
    assert (synced / "textures" / f"{MODEL}_basecolor.png").is_file()
    assert (synced / "animations" / f"{MODEL}@walk.fbx").is_file()


def test_sync_failure_is_handoff_error_after_local_output_exists(
    tmp_path: Path, stage: HandoffStage
) -> None:
    config = LibraryConfig.model_validate(
        {"library": {"unity_import_dir": str(tmp_path / "missing")}}
    )
    model = make_model(tmp_path, config=config)
    write_reports(model)
    with pytest.raises(HandoffError, match="unity_import_dir does not exist"):
        stage.run(model, make_ctx())
    assert (model.paths.output / MODEL / "manifest.json").is_file()


# --------------------------------------------------------------------------- fingerprint
def test_fingerprint_matrix(tmp_path: Path, stage: HandoffStage) -> None:
    model = make_model(tmp_path)
    write_reports(model)
    ctx = make_ctx()
    base = stage.input_fingerprint(model, ctx)
    assert base == stage.input_fingerprint(model, ctx)
    seen = {base}

    def differs(label: str, m: ModelContext = model) -> None:
        fp = stage.input_fingerprint(m, ctx)
        assert fp not in seen, label
        seen.add(fp)

    (model.paths.work / "export" / f"{MODEL}.fbx").write_bytes(b"FBX-export-2")
    differs("export fbx bytes")
    (model.paths.work / "export" / "animations" / f"{MODEL}@walk.fbx").write_bytes(b"clip-2")
    differs("clip bytes")
    unity_cfg = LibraryConfig.model_validate({"unity": {"avatar_mode": "create_from_model"}})
    differs("unity section", make_model(tmp_path, config=unity_cfg))
    sync_cfg = LibraryConfig.model_validate({"library": {"unity_import_dir": str(tmp_path)}})
    differs("unity import dir", make_model(tmp_path, config=sync_cfg))
    current = stage.input_fingerprint(model, ctx)
    unrelated = LibraryConfig.model_validate({"texture": {"quality": "detailed"}, "bake": {}})
    assert stage.input_fingerprint(make_model(tmp_path, config=unrelated), ctx) == current


def test_fingerprint_requires_export_report(tmp_path: Path, stage: HandoffStage) -> None:
    model = make_model(tmp_path)
    with pytest.raises(StagePrereqError, match=r"rig_fidelity_report\.json"):
        stage.input_fingerprint(model, make_ctx())
    with pytest.raises(StagePrereqError):
        stage.run(model, make_ctx())


# --------------------------------------------------------------------------- readers
def test_readers_are_tolerant_where_the_plan_allows(tmp_path: Path) -> None:
    folder = tmp_path / "m"
    folder.mkdir()
    assert read_route(folder) == ""
    assert read_texture_task(folder) == ("", 0.0)
    assert read_apply_material(folder) is None
    with pytest.raises(StagePrereqError, match=r"prepare_report\.json"):
        read_uv_decision(folder)
    with pytest.raises(StagePrereqError, match=r"rig_fingerprint_out\.json"):
        read_output_rig(folder)

    write_json(folder / "work" / "verify_report.json", {"route": 7})
    write_json(folder / "work" / "tripo" / "task.json", {"task_id": "t1", "credits_consumed": "x"})
    write_json(folder / "work" / "apply_report.json", {"urp": {"basecolor": "work/x.png"}})
    write_json(folder / "work" / "prepare_report.json", {"uv": {"decision": "unwrapped"}})
    write_json(folder / RIG_FINGERPRINT_OUT, json.loads(sample_rig().model_dump_json()))
    assert read_route(folder) == ""
    assert read_texture_task(folder) == ("t1", 0.0)
    maps, material = read_apply_material(folder) or ({}, None)
    assert maps == {"basecolor": folder / "work" / "x.png"}
    assert material == UrpMaterialInputs()
    assert read_uv_decision(folder) == "unwrapped"
    assert read_output_rig(folder) == RigFingerprint.model_validate_json(
        sample_rig().model_dump_json()
    )


def test_default_clock_is_timezone_aware_utc() -> None:
    assert HandoffStage().now is utc_now
    now = utc_now()
    assert now.tzinfo is not None
    assert now.utcoffset() is not None
    assert now.utcoffset().total_seconds() == 0  # type: ignore[union-attr]  # checked above
    assert iso_utc(now).endswith("Z")


def test_malformed_helper_reports_are_treated_as_absent(tmp_path: Path) -> None:
    folder = tmp_path / "m"
    (folder / "work").mkdir(parents=True)
    (folder / "work" / "verify_report.json").write_text("{not json", encoding="utf-8")
    (folder / "work" / "apply_report.json").write_text("[]", encoding="utf-8")
    assert read_route(folder) == ""
    assert read_apply_material(folder) is None


def test_iso_utc_formats_seconds_in_utc() -> None:
    assert iso_utc(FIXED_NOW) == "2026-11-02T12:00:00Z"
    assert iso_utc(datetime(2026, 11, 2, 13, 30, 15, 999, tzinfo=UTC)) == "2026-11-02T13:30:15Z"


def test_subdivision_type_is_recorded_with_blender_name(
    tmp_path: Path, stage: HandoffStage
) -> None:
    config = LibraryConfig.model_validate(
        {"geometry": {"subdivision_type": "simple", "subdivision_levels": 2}}
    )
    model = make_model(tmp_path, config=config)
    write_reports(model)
    stage.run(model, make_ctx())
    doc = json.loads((model.paths.output / MODEL / "manifest.json").read_text(encoding="utf-8"))
    assert doc["provenance"]["subdivision_type"] == "simple"
    assert doc["provenance"]["subdivision_levels"] == 2
