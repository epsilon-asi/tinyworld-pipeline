from __future__ import annotations

import json
from pathlib import Path

from tinyworld_pipeline.domain.manifest import (
    Manifest,
    ManifestAnimation,
    ManifestColor,
    ManifestFloat,
    ManifestMaterial,
    ManifestProvenance,
    ManifestRig,
    ManifestTexture,
    json_utility_violations,
    manifest_json_schema,
)

ROOT = Path(__file__).resolve().parents[3]


def sample_manifest() -> Manifest:
    return Manifest(
        pipeline_version="0.1.0",
        generated_at="2026-11-02T00:00:00Z",
        model_id="ant_worker",
        display_name="Worker Ant",
        fbx="ant_worker.fbx",
        rig=ManifestRig(
            animation_type="Generic",
            avatar_mode="copy_from_source",
            source_avatar_asset_path="",
            root_bone="Root",
        ),
        materials=[
            ManifestMaterial(
                name="ant_worker_mat",
                shader="Universal Render Pipeline/Lit",
                renderers=[],
                textures=[
                    ManifestTexture(
                        slot="_BaseMap",
                        file="textures/ant_worker_basecolor.png",
                        srgb=True,
                        kind="default",
                    )
                ],
                floats=[ManifestFloat(name="_Smoothness", value=1.0)],
                colors=[ManifestColor(name="_BaseColor", r=1, g=1, b=1, a=1)],
                keywords=["_NORMALMAP"],
            )
        ],
        animations=[ManifestAnimation(file="animations/ant_worker@walk.fbx", clip_name="walk")],
        provenance=ManifestProvenance(
            source_fbx_sha256="a" * 64,
            source_license="CC0",
            subdivision_levels=1,
            subdivision_type="catmull",
            uv_decision="unwrapped",
            texture_provider="tripo",
            texture_task_id="task_1",
            texture_route="extract",
            texture_seed=1337,
            prompt_sha256="b" * 64,
            credits_consumed=10.0,
            rig_fidelity="pass",
        ),
    )


def test_sample_manifest_is_json_utility_compatible() -> None:
    doc = json.loads(sample_manifest().model_dump_json())
    assert json_utility_violations(doc) == []
    assert doc["schema_version"] == 1


def test_json_utility_violations_reports_nulls_and_mixed_arrays() -> None:
    doc = {"a": None, "b": [1, "x"], "c": {"d": [None]}}
    problems = json_utility_violations(doc)
    assert "$.a: null is not representable" in problems
    assert any(p.startswith("$.b: heterogeneous array") for p in problems)
    assert "$.c.d[0]: null is not representable" in problems


def test_manifest_models_declare_no_dict_or_optional_fields() -> None:
    schema = manifest_json_schema()
    for name, definition in schema.get("$defs", {}).items():
        for prop, spec in definition.get("properties", {}).items():
            assert "anyOf" not in spec, f"{name}.{prop} is nullable/union"
            assert spec.get("type") != "object" or "$ref" in spec, f"{name}.{prop} is a dict"


def test_committed_manifest_schema_is_not_stale() -> None:
    committed = ROOT / "docs" / "schemas" / "manifest.schema.json"
    assert committed.exists(), "run tools/export_schemas.py"
    assert json.loads(committed.read_text(encoding="utf-8")) == manifest_json_schema()
