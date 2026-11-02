"""Plan 06: pydantic protocol (orchestrator) and dataclass protocol (Blender) must agree."""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest
from pydantic import BaseModel

from tinyworld_pipeline.blender import protocol as orch
from tinyworld_pipeline.blender_jobs import protocol as bj
from tinyworld_pipeline.domain.library_config import GeometrySection, UvSection


@pytest.mark.parametrize("name", bj.MIRRORED_TYPES)
def test_mirrored_type_field_names_match(name: str) -> None:
    pyd: type[BaseModel] = getattr(orch, name)
    dc = getattr(bj, name)

    pyd_fields = set(pyd.model_json_schema()["properties"])
    dc_fields = {f.name for f in dataclasses.fields(dc)}
    assert pyd_fields == dc_fields, f"{name}: {pyd_fields ^ dc_fields}"


def test_job_tables_agree() -> None:
    assert (
        set(orch.PARAMS_BY_JOB) == set(bj.PARAMS_BY_JOB) == set(orch.JOB_NAMES) == set(bj.JOB_NAMES)
    )
    assert set(orch.DATA_BY_JOB) == set(bj.DATA_BY_JOB)
    assert orch.PROTOCOL_VERSION == bj.PROTOCOL_VERSION == 1
    assert (orch.EXIT_OK, orch.EXIT_UNHANDLED, orch.EXIT_BAD_JOB_FILE) == (0, 2, 3)
    assert (bj.EXIT_OK, bj.EXIT_UNHANDLED, bj.EXIT_BAD_JOB_FILE) == (0, 2, 3)


def test_config_section_defaults_agree() -> None:
    for name in ("GeometrySection", "UvSection", "BakeSection", "ExportSection"):
        pyd_defaults = getattr(orch, name).model_validate({}).model_dump(mode="json")
        dc_defaults = bj.to_dict(getattr(bj, name)())
        assert pyd_defaults == dc_defaults, name


def test_all_bj_codes_are_unique_and_prefixed() -> None:
    assert len(bj.ALL_BJ_CODES) == len(set(bj.ALL_BJ_CODES))
    assert all(code.startswith("E_BJ_") for code in bj.ALL_BJ_CODES)
    assert "E_BJ_SHAPEKEYS" in bj.ALL_BJ_CODES


def _sample_params() -> orch.PrepareModelParams:
    return orch.PrepareModelParams(
        source_fbx="/m/source/a.fbx",
        out_blend="/m/work/prepared.blend",
        out_static_glb="/m/work/static.glb",
        out_report="/m/work/prepare_report.json",
        geometry=GeometrySection(subdivision_levels=2),
        uv=UvSection(mode="unwrap"),
        material_name="a_mat",
        palette_hint_textures=["/m/source/textures/p.png"],
    )


def test_pydantic_dump_loads_into_dataclass_and_back() -> None:
    dumped = _sample_params().model_dump(mode="json")

    dc = bj.from_dict(bj.PrepareModelParams, dumped)
    assert dc.geometry.subdivision_levels == 2
    assert dc.uv.mode == "unwrap"
    assert dc.palette_hint_textures == ["/m/source/textures/p.png"]
    assert bj.to_dict(dc) == dumped
    assert orch.PrepareModelParams.model_validate(bj.to_dict(dc)) == _sample_params()


def test_dataclass_result_validates_as_pydantic_result() -> None:
    bone = bj.Bone(
        name="Root",
        parent=None,
        head=(0.0, 0.0, 0.0),
        tail=(0.0, 1.0, 0.0),
        matrix_local=[[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0], [0, 0, 0, 1.0]],
        transform_path="Armature/Root",
    )
    rig = bj.RigFingerprint(
        armature_object="Armature",
        armature_data="Armature",
        bone_count=1,
        bones=[bone],
        actions=["idle"],
        fingerprint_sha256="f" * 64,
    )
    result = bj.JobResult(
        job="export_rigged",
        job_id="01ARZ3NDEKTSV4RRFFQ69G5FAV",
        ok=True,
        blender_version="4.5.12",
        duration_s=1.0,
        data=bj.to_dict(
            bj.ExportRiggedData(rig=rig, fbx_sha256="e" * 64, actions_exported=["idle"])
        ),
    )

    parsed = orch.JobResult.model_validate(bj.to_dict(result))
    data = orch.ExportRiggedData.model_validate(parsed.data)
    assert data.rig.bones[0].head == (0.0, 0.0, 0.0)
    assert data.rig.bones[0].transform_path == "Armature/Root"


def test_from_dict_rejects_unknown_keys_and_handles_optionals() -> None:
    with pytest.raises(KeyError, match="unknown keys"):
        bj.from_dict(bj.JobError, {"code": "E", "message": "m", "bogus": 1})
    err = bj.from_dict(bj.JobError, {"code": "E_BJ_X", "message": "m"})
    assert err.detail == {}
    assert err.traceback is None
    nested: dict[str, Any] = {"job": "probe", "job_id": "j", "params": {}, "model_id": None}
    assert bj.from_dict(bj.JobRequest, nested).model_id is None


def test_to_dict_converts_tuples_and_nested_dicts() -> None:
    data = bj.BakeTexturesData(
        maps={"basecolor": "raw/basecolor.png"},
        resolution=2048,
        samples=16,
        bake_seconds={"basecolor": 1.5},
    )
    assert bj.to_dict(data) == {
        "maps": {"basecolor": "raw/basecolor.png"},
        "resolution": 2048,
        "samples": 16,
        "bake_seconds": {"basecolor": 1.5},
    }
    assert bj.to_dict((1, 2)) == [1, 2]
    assert bj.to_dict("x") == "x"
