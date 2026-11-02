"""``blender_jobs/entry.py`` run in-process under plain CPython with fake jobs in ``JOBS``."""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from tinyworld_pipeline.blender.protocol import PROTOCOL_VERSION, JobResult
from tinyworld_pipeline.blender_jobs import entry, protocol
from tinyworld_pipeline.blender_jobs.errors import JobFailure
from tinyworld_pipeline.blender_jobs.jobs import JOBS

JOB_ID = "01ENTRYTEST000000000000000"


def _job_file(tmp_path: Path, payload: dict[str, Any] | str) -> Path:
    path = tmp_path / "job.json"
    path.write_text(payload if isinstance(payload, str) else json.dumps(payload))
    return path


def _request(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "job": "probe",
        "job_id": JOB_ID,
        "model_id": None,
        "params": {},
        "log_level": "INFO",
    }
    base.update(overrides)
    return base


def _run(tmp_path: Path, payload: dict[str, Any] | str) -> tuple[int, Path]:
    job = _job_file(tmp_path, payload)
    result = tmp_path / "out" / "result.json"
    code = entry.main(["entry.py", "-b", "--", "--job", str(job), "--result", str(result)])
    return code, result


def _load(result: Path) -> JobResult:
    """Validate with the orchestrator-side pydantic contract (the real consumer)."""
    return JobResult.model_validate_json(result.read_text())


@pytest.fixture
def fake_probe(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Replace the real ``probe`` job with a scripted one; returns a mutable spec dict."""
    spec: dict[str, Any] = {"mode": "ok", "seen": None}

    def fake(params: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
        spec["seen"] = params
        if spec["mode"] == "failure":
            raise JobFailure("E_BJ_NO_ARMATURE", "no armature", {"objects": ["Cube"]})
        if spec["mode"] == "boom":
            msg = "unexpected"
            raise RuntimeError(msg)
        return {"answer": 42, "echo": params}, ["minor warning"]

    monkeypatch.setitem(JOBS, "probe", fake)
    return spec


# --------------------------------------------------------------------------- happy path
def test_ok_result(tmp_path: Path, fake_probe: dict[str, Any]) -> None:
    code, result_path = _run(tmp_path, _request(params={"x": 1}))
    assert code == entry.EXIT_OK
    result = _load(result_path)
    assert result.ok is True
    assert result.job == "probe"
    assert result.job_id == JOB_ID
    assert result.data == {"answer": 42, "echo": {"x": 1}}
    assert result.warnings == ["minor warning"]
    assert result.error is None
    assert result.blender_version == "unknown"  # no bpy under plain CPython
    assert result.duration_s >= 0
    assert fake_probe["seen"] == {"x": 1}
    assert not result_path.with_name("result.json.tmp").exists()


def test_argv_without_double_dash_is_accepted(tmp_path: Path, fake_probe: dict[str, Any]) -> None:
    job = _job_file(tmp_path, _request())
    result = tmp_path / "result.json"
    assert entry.main(["entry.py", "--job", str(job), "--result", str(result)]) == entry.EXIT_OK
    assert _load(result).ok


# --------------------------------------------------------------------------- failures
def test_job_failure_is_ok_false_exit_zero(tmp_path: Path, fake_probe: dict[str, Any]) -> None:
    fake_probe["mode"] = "failure"
    code, result_path = _run(tmp_path, _request())
    assert code == entry.EXIT_OK
    result = _load(result_path)
    assert result.ok is False
    assert result.error is not None
    assert result.error.code == "E_BJ_NO_ARMATURE"
    assert result.error.message == "no armature"
    assert result.error.detail == {"objects": ["Cube"]}
    assert result.error.traceback is None


def test_unhandled_exception_exit_2_with_traceback(
    tmp_path: Path, fake_probe: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    fake_probe["mode"] = "boom"
    code, result_path = _run(tmp_path, _request())
    assert code == entry.EXIT_UNHANDLED == protocol.EXIT_UNHANDLED
    result = _load(result_path)
    assert result.ok is False
    assert result.error is not None
    assert result.error.code == protocol.E_BJ_UNHANDLED
    assert "RuntimeError: unexpected" in result.error.message
    assert result.error.traceback is not None
    assert "RuntimeError" in result.error.traceback
    assert "unhandled exception" in capsys.readouterr().err


# --------------------------------------------------------------------------- dispatch
def test_unregistered_known_job_name(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(JOBS, "probe")
    code, result_path = _run(tmp_path, _request())
    assert code == entry.EXIT_OK
    result = _load(result_path)
    assert result.ok is False
    assert result.error is not None
    assert result.error.code == protocol.E_BJ_UNKNOWN_JOB
    assert result.error.detail["known"] == sorted(JOBS)


def test_unknown_job_name(tmp_path: Path) -> None:
    code, result_path = _run(tmp_path, _request(job="make_coffee"))
    assert code == entry.EXIT_OK
    raw = json.loads(result_path.read_text())
    # Not a JobName on the orchestrator side, so validate with the dataclass mirror here.
    result = protocol.from_dict(protocol.JobResult, raw)
    assert result.ok is False
    assert result.error.code == protocol.E_BJ_UNKNOWN_JOB
    assert "make_coffee" in result.error.message


def test_protocol_mismatch(tmp_path: Path, fake_probe: dict[str, Any]) -> None:
    code, result_path = _run(tmp_path, _request(protocol_version=99))
    assert code == entry.EXIT_OK
    raw = json.loads(result_path.read_text())
    assert raw["protocol_version"] == PROTOCOL_VERSION  # entry answers in its own version
    result = _load(result_path)
    assert result.ok is False
    assert result.error is not None
    assert result.error.code == "E_BJ_PROTOCOL"
    assert result.error.detail == {"got": 99, "expected": PROTOCOL_VERSION}
    assert fake_probe["seen"] is None  # job never ran


# --------------------------------------------------------------------------- bad job file / args
def test_bad_job_file_garbage(tmp_path: Path) -> None:
    code, result_path = _run(tmp_path, "{ nope")
    assert code == entry.EXIT_BAD_JOB_FILE == protocol.EXIT_BAD_JOB_FILE
    assert not result_path.exists()  # nothing to identify the job by


def test_bad_job_file_missing(tmp_path: Path) -> None:
    result = tmp_path / "result.json"
    code = entry.main(
        ["entry.py", "--", "--job", str(tmp_path / "missing.json"), "--result", str(result)]
    )
    assert code == entry.EXIT_BAD_JOB_FILE
    assert not result.exists()


def test_bad_job_file_not_object(tmp_path: Path) -> None:
    code, result_path = _run(tmp_path, "[1, 2, 3]")
    assert code == entry.EXIT_BAD_JOB_FILE
    assert not result_path.exists()


def test_bad_job_file_unknown_key_writes_result_when_identifiable(tmp_path: Path) -> None:
    code, result_path = _run(tmp_path, _request(bogus=True))
    assert code == entry.EXIT_BAD_JOB_FILE
    result = _load(result_path)
    assert result.ok is False
    assert result.error is not None
    assert result.error.code == protocol.E_BJ_BAD_PARAMS
    assert "bogus" in result.error.message


def test_bad_job_file_params_not_object(tmp_path: Path) -> None:
    code, result_path = _run(tmp_path, _request(params="nope"))
    assert code == entry.EXIT_BAD_JOB_FILE
    assert _load(result_path).error is not None


def test_missing_arguments(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert (
        entry.main(["entry.py", "--", "--job", str(tmp_path / "j.json")]) == entry.EXIT_BAD_JOB_FILE
    )
    assert entry.main(["entry.py", "--", "--result"]) == entry.EXIT_BAD_JOB_FILE
    assert "usage" in capsys.readouterr().err


# --------------------------------------------------------------------------- bootstrap
def test_sys_path_bootstrap_from_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("GP_SRC_DIR", str(tmp_path))
    monkeypatch.setattr(sys, "path", list(sys.path))
    entry._bootstrap_sys_path()
    assert sys.path[0] == str(tmp_path)
    entry._bootstrap_sys_path()
    assert sys.path.count(str(tmp_path)) == 1


def test_sys_path_bootstrap_defaults_to_package_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GP_SRC_DIR", raising=False)
    monkeypatch.setattr(sys, "path", [p for p in sys.path if "src" not in p])
    entry._bootstrap_sys_path()
    assert (Path(sys.path[0]) / "tinyworld_pipeline" / "blender_jobs" / "entry.py").is_file()


def test_module_level_imports_are_stdlib_only() -> None:
    """entry.py must load inside Blender before GP_SRC_DIR is on sys.path (appendix B §1)."""
    tree = ast.parse(Path(entry.__file__).read_text())
    for node in tree.body:
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            continue
        for name in names:
            assert not name.startswith("tinyworld_pipeline"), name
            assert name.split(".")[0] not in {"bpy", "pydantic", "structlog"}, name
