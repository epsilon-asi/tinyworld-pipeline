"""The scripted Blender runner must record, write outputs consistently, and raise on demand."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from fakes.fake_blender_runner import (
    FakeBlenderRunner,
    job_error_response,
    prepare_model_response,
    sample_prepare_data,
)
from tinyworld_pipeline.blender.protocol import JobRequest, JobResult, PrepareModelData
from tinyworld_pipeline.domain.errors import BlenderJobError, BlenderTimeoutError, CancelledError
from tinyworld_pipeline.domain.library_config import GeometrySection, UvSection
from tinyworld_pipeline.domain.ports import CancelToken

JOB_ID = "01FAKERUNNER00000000000000"


def _request(tmp_path: Path) -> JobRequest:
    work = tmp_path / "work"
    params = {
        "source_fbx": str(tmp_path / "source" / "cube.fbx"),
        "out_blend": str(work / "prepared.blend"),
        "out_static_glb": str(work / "static.glb"),
        "out_report": str(work / "prepare_report.json"),
        "geometry": GeometrySection().model_dump(mode="json"),
        "uv": UvSection().model_dump(mode="json"),
        "material_name": "cube_mat",
        "palette_hint_textures": [],
    }
    return JobRequest(job="prepare_model", job_id=JOB_ID, model_id="cube", params=params)


def test_prepare_response_writes_outputs_and_matching_sha(tmp_path: Path) -> None:
    runner = FakeBlenderRunner({"prepare_model": prepare_model_response(static_glb_bytes=b"abc")})
    log = tmp_path / "logs" / "p.log"

    result = runner.run(_request(tmp_path), timeout_s=30, log_path=log, cancel=CancelToken())

    assert isinstance(result, JobResult)
    assert result.ok is True
    assert result.job_id == JOB_ID
    data = PrepareModelData.model_validate(result.data)
    assert data.static_glb_sha256 == hashlib.sha256(b"abc").hexdigest()
    assert (tmp_path / "work" / "static.glb").read_bytes() == b"abc"
    assert (tmp_path / "work" / "prepared.blend").exists()
    assert json.loads((tmp_path / "work" / "prepare_report.json").read_text()) == result.data
    assert "job=prepare_model" in log.read_text()
    assert runner.call_count("prepare_model") == 1
    assert runner.call_count("probe") == 0
    assert runner.requests == [_request(tmp_path)]
    assert runner.timeouts == [30]
    assert runner.log_paths == [log]


def test_prepare_response_overrides_and_no_outputs(tmp_path: Path) -> None:
    respond = prepare_model_response(
        data_overrides={"static_glb_triangles": 7}, warnings=("w",), write_outputs=False
    )
    runner = FakeBlenderRunner({"prepare_model": respond})
    result = runner.run(
        _request(tmp_path), timeout_s=1, log_path=tmp_path / "l", cancel=CancelToken()
    )
    assert result.data["static_glb_triangles"] == 7
    assert result.warnings == ["w"]
    assert not (tmp_path / "work").exists()


def test_exception_responses_are_raised(tmp_path: Path) -> None:
    runner = FakeBlenderRunner(
        {
            "prepare_model": job_error_response("E_BJ_SHAPEKEYS", "keys", meshes=["Face"]),
            "probe": lambda _req: BlenderTimeoutError("slow"),
        }
    )
    with pytest.raises(BlenderJobError) as info:
        runner.run(_request(tmp_path), timeout_s=1, log_path=tmp_path / "l", cancel=CancelToken())
    assert info.value.job_code == "E_BJ_SHAPEKEYS"
    assert info.value.detail == {"meshes": ["Face"], "job_id": JOB_ID}
    probe = JobRequest(job="probe", job_id=JOB_ID, params={})
    with pytest.raises(BlenderTimeoutError):
        runner.run(probe, timeout_s=1, log_path=tmp_path / "l", cancel=CancelToken())
    assert runner.calls == {"prepare_model": 1, "probe": 1}


def test_cancelled_token_and_unknown_job(tmp_path: Path) -> None:
    runner = FakeBlenderRunner()
    cancel = CancelToken()
    cancel.set()
    with pytest.raises(CancelledError):
        runner.run(_request(tmp_path), timeout_s=1, log_path=tmp_path / "l", cancel=cancel)
    with pytest.raises(BlenderJobError) as info:
        runner.run(_request(tmp_path), timeout_s=1, log_path=tmp_path / "l", cancel=CancelToken())
    assert info.value.job_code == "E_BJ_UNKNOWN_JOB"


def test_sample_data_is_a_valid_contract_value() -> None:
    data = sample_prepare_data(static_glb_sha256="0" * 64)
    assert data.rig.bones[-1].transform_path == "Armature/Root/Spine/Head"
    assert data.mesh.faces_after == 24
