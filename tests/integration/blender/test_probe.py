"""WP-03 acceptance: the ``probe`` job runs under a real Blender through the real runner."""

from __future__ import annotations

import os
import threading
from pathlib import Path

import pytest

from grounded_pipeline.blender.locate import blender_version, check_version
from grounded_pipeline.blender.protocol import JobRequest, ProbeData
from grounded_pipeline.blender.runner import SubprocessBlenderRunner, default_entry_script
from grounded_pipeline.domain.ports import CancelToken

pytestmark = pytest.mark.blender


def test_probe_job_under_real_blender(blender_exe: Path, tmp_path: Path) -> None:
    version = blender_version(blender_exe)
    assert check_version(version, allow_unsupported=False) == []

    runner = SubprocessBlenderRunner(
        blender_exe,
        default_entry_script(),
        threading.BoundedSemaphore(1),
        os.environ,
        poll_interval_s=0.2,
    )
    req = JobRequest(job="probe", job_id="01PROBEINTEGRATION00000000", params={})
    log_path = tmp_path / "probe.log"

    result = runner.run(req, timeout_s=300, log_path=log_path, cancel=CancelToken())

    assert result.ok is True
    assert result.job == "probe"
    assert result.job_id == req.job_id
    assert result.duration_s > 0
    assert result.blender_version.startswith("4.5")
    probe = ProbeData.model_validate(result.data)
    assert probe.blender_version == result.blender_version
    assert probe.fbx_io_available is True
    assert probe.gltf_io_available is True
    assert probe.has_cycles is True
    assert probe.python_version.startswith("3.")
    text = log_path.read_text()
    assert "[entry] job=probe" in text
    assert "[entry] ok=True exit=0" in text
