"""Plan 13 §10: ``SubprocessBlenderRunner`` against the fake Blender executable (no Blender)."""

from __future__ import annotations

import json
import os
import re
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from fakes.fake_blender_exe import write_launcher
from tinyworld_pipeline.blender.protocol import PROTOCOL_VERSION, JobRequest, JobResult
from tinyworld_pipeline.blender.runner import (
    BlenderRunner,
    SubprocessBlenderRunner,
    default_entry_script,
    default_src_dir,
)
from tinyworld_pipeline.domain.errors import (
    BlenderCrashError,
    BlenderJobError,
    BlenderNoResultError,
    BlenderNotFoundError,
    BlenderTimeoutError,
    CancelledError,
    ProtocolDriftError,
)
from tinyworld_pipeline.domain.ports import CancelToken

FAKE_ENTRY = Path("/fake/entry.py")


@pytest.fixture
def fake_blender(tmp_path: Path) -> Path:
    return write_launcher(tmp_path / "bin")


@pytest.fixture
def log_path(tmp_path: Path) -> Path:
    return tmp_path / "logs" / "job.log"


RunnerFactory = Callable[..., SubprocessBlenderRunner]


@pytest.fixture
def make_runner(fake_blender: Path) -> RunnerFactory:
    def factory(**overrides: Any) -> SubprocessBlenderRunner:
        kwargs: dict[str, Any] = {
            "executable": fake_blender,
            "entry_script": FAKE_ENTRY,
            "semaphore": threading.BoundedSemaphore(1),
            "env": {"PATH": os.environ.get("PATH", "")},
            "poll_interval_s": 0.05,
            "kill_grace_s": 2.0,
        }
        kwargs.update(overrides)
        return SubprocessBlenderRunner(**kwargs)

    return factory


def _req(fake: str, job_id: str = "01JOBID000000000000000000X", **params: Any) -> JobRequest:
    return JobRequest(job="probe", job_id=job_id, params={"__fake": fake, **params})


def _pid_from_log(log_path: Path) -> int:
    match = re.search(r"^FAKE_PID (\d+)$", log_path.read_text(), flags=re.MULTILINE)
    assert match, log_path.read_text()
    return int(match.group(1))


def _pid_when_logged(log_path: Path, *, deadline_s: float = 20.0) -> int:
    """Block until the fake executable has written its ``FAKE_PID`` line (process start-up under a
    loaded machine can take well over a second), then return that pid."""
    end = time.monotonic() + deadline_s
    while time.monotonic() < end:
        if log_path.exists():
            match = re.search(r"^FAKE_PID (\d+)$", log_path.read_text(), flags=re.MULTILINE)
            if match:
                return int(match.group(1))
        time.sleep(0.02)
    msg = f"fake Blender never logged its pid within {deadline_s}s"
    raise AssertionError(msg)


def _process_gone(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    return False


# --------------------------------------------------------------------------- protocol conformance
def test_subprocess_runner_satisfies_port(make_runner: RunnerFactory) -> None:
    runner: BlenderRunner = make_runner()
    assert isinstance(runner, SubprocessBlenderRunner)


def test_command_shape_and_defaults(make_runner: RunnerFactory, fake_blender: Path) -> None:
    runner = make_runner()
    cmd = runner.command(Path("/t/job.json"), Path("/t/result.json"))
    assert cmd == [
        str(fake_blender),
        "-b",
        "--factory-startup",
        "--python-exit-code",
        "2",
        "--python",
        str(FAKE_ENTRY),
        "--",
        "--job",
        "/t/job.json",
        "--result",
        "/t/result.json",
    ]
    assert runner.child_env()["GP_SRC_DIR"] == str(default_src_dir())
    assert runner.child_env()["PYTHONDONTWRITEBYTECODE"] == "1"
    assert (default_src_dir() / "tinyworld_pipeline" / "__init__.py").is_file()
    assert default_entry_script().is_file()
    assert default_entry_script().name == "entry.py"


# --------------------------------------------------------------------------- outcomes
def test_ok_result_parsed(make_runner: RunnerFactory, log_path: Path) -> None:
    result = make_runner().run(
        _req("ok", extra=1), timeout_s=60, log_path=log_path, cancel=CancelToken()
    )
    assert isinstance(result, JobResult)
    assert result.ok is True
    assert result.job == "probe"
    assert result.protocol_version == PROTOCOL_VERSION
    assert result.data["echo"] == {"__fake": "ok", "extra": 1}
    assert result.blender_version == "4.5.12"


def test_job_error_passthrough(make_runner: RunnerFactory, log_path: Path) -> None:
    with pytest.raises(BlenderJobError) as info:
        make_runner().run(_req("job_error"), timeout_s=60, log_path=log_path, cancel=CancelToken())
    err = info.value
    assert err.job_code == "E_BJ_SHAPEKEYS"
    assert err.message == "mesh has shape keys"
    assert err.detail["meshes"] == ["Body"]
    assert err.detail["exit_code"] == 0
    assert err.to_dict()["code"] == "E_BJ_SHAPEKEYS"


def test_crash_raises_with_log_tail(make_runner: RunnerFactory, log_path: Path) -> None:
    with pytest.raises(BlenderCrashError) as info:
        make_runner().run(_req("crash"), timeout_s=60, log_path=log_path, cancel=CancelToken())
    err = info.value
    assert err.detail["exit_code"] == 1
    assert any("Segmentation fault (fake)" in line for line in err.detail["log_tail"])
    assert err.detail["log_path"] == str(log_path)
    assert "Segmentation fault (fake)" in log_path.read_text()


def test_crash_log_tail_is_bounded(make_runner: RunnerFactory, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True)
    log_path.write_text("".join(f"old line {i}\n" for i in range(100)))
    with pytest.raises(BlenderCrashError) as info:
        make_runner().run(_req("crash"), timeout_s=60, log_path=log_path, cancel=CancelToken())
    assert len(info.value.detail["log_tail"]) == 40


def test_no_result_with_exit_zero(make_runner: RunnerFactory, log_path: Path) -> None:
    with pytest.raises(BlenderNoResultError) as info:
        make_runner().run(_req("no_result"), timeout_s=60, log_path=log_path, cancel=CancelToken())
    assert info.value.code == "E_BLENDER_NO_RESULT"


def test_bad_json_is_protocol_drift(make_runner: RunnerFactory, log_path: Path) -> None:
    with pytest.raises(ProtocolDriftError) as info:
        make_runner().run(_req("bad_json"), timeout_s=60, log_path=log_path, cancel=CancelToken())
    assert "not valid JSON" in info.value.message


def test_protocol_version_mismatch_rejected(make_runner: RunnerFactory, log_path: Path) -> None:
    with pytest.raises(ProtocolDriftError) as info:
        make_runner().run(
            _req("wrong_protocol"), timeout_s=60, log_path=log_path, cancel=CancelToken()
        )
    assert info.value.detail == {
        "job_id": _req("x").job_id,
        "got": 99,
        "expected": PROTOCOL_VERSION,
    }


# --------------------------------------------------------------------------- timeout / cancel
def test_timeout_kills_process(make_runner: RunnerFactory, log_path: Path) -> None:
    pid_box: list[int] = []
    watcher = threading.Thread(target=lambda: pid_box.append(_pid_when_logged(log_path)))
    watcher.start()
    try:
        with pytest.raises(BlenderTimeoutError) as info:
            make_runner().run(
                _req("sleep:30"), timeout_s=2, log_path=log_path, cancel=CancelToken()
            )
    finally:
        watcher.join(timeout=25)
    assert info.value.detail["timeout_s"] == 2
    assert pid_box, "fake process never started"
    assert _process_gone(pid_box[0])


def test_cancel_mid_run_kills_process(make_runner: RunnerFactory, log_path: Path) -> None:
    cancel = CancelToken()
    pid_box: list[int] = []

    def cancel_once_running() -> None:
        pid_box.append(_pid_when_logged(log_path))
        cancel.set()

    canceller = threading.Thread(target=cancel_once_running)
    canceller.start()
    try:
        with pytest.raises(CancelledError) as info:
            make_runner().run(_req("sleep:30"), timeout_s=60, log_path=log_path, cancel=cancel)
    finally:
        canceller.join(timeout=25)
    assert info.value.exit_code == 130
    assert pid_box, "fake process never started"
    assert _process_gone(pid_box[0])


def test_cancel_before_start_does_not_spawn(make_runner: RunnerFactory, log_path: Path) -> None:
    cancel = CancelToken()
    cancel.set()
    with pytest.raises(CancelledError):
        make_runner().run(_req("ok"), timeout_s=60, log_path=log_path, cancel=cancel)
    assert not log_path.exists()


# --------------------------------------------------------------------------- concurrency
def test_semaphore_serialises_runs(make_runner: RunnerFactory, tmp_path: Path) -> None:
    stamp_dir = tmp_path / "stamps"
    stamp_dir.mkdir()
    runner = make_runner(semaphore=threading.BoundedSemaphore(1))
    errors: list[BaseException] = []

    def work(job_id: str) -> None:
        try:
            runner.run(
                _req("sleep:0.3", job_id=job_id, __stamp_dir=str(stamp_dir)),
                timeout_s=60,
                log_path=tmp_path / f"{job_id}.log",
                cancel=CancelToken(),
            )
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(f"JOB{i}",)) for i in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors
    stamps = [json.loads(p.read_text()) for p in sorted(stamp_dir.glob("*.json"))]
    assert len(stamps) == 2
    first, second = sorted(stamps, key=lambda s: s["start"])
    assert second["start"] >= first["end"], stamps


# --------------------------------------------------------------------------- environment / files
def test_child_env_contains_src_dir_and_no_bytecode(
    make_runner: RunnerFactory, log_path: Path, tmp_path: Path
) -> None:
    src = tmp_path / "src"
    make_runner(src_dir=src).run(_req("ok"), timeout_s=60, log_path=log_path, cancel=CancelToken())
    text = log_path.read_text()
    assert f"FAKE_ENV GP_SRC_DIR={src}" in text
    assert "FAKE_ENV PYTHONDONTWRITEBYTECODE=1" in text


def test_log_file_contains_fake_stdout_and_runner_header(
    make_runner: RunnerFactory, log_path: Path
) -> None:
    make_runner().run(_req("ok"), timeout_s=60, log_path=log_path, cancel=CancelToken())
    text = log_path.read_text()
    assert "[runner] job=probe" in text
    assert "FAKE_JOB probe" in text
    assert "--factory-startup" in text  # argv echoed by the fake


def test_log_is_appended_not_truncated(make_runner: RunnerFactory, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True)
    log_path.write_text("previous run\n")
    make_runner().run(_req("ok"), timeout_s=60, log_path=log_path, cancel=CancelToken())
    assert log_path.read_text().startswith("previous run\n")


def _temp_dirs_from_log(log_path: Path) -> set[Path]:
    """The fake echoes argv, which carries the temp dir of job.json."""
    found = set()
    for match in re.finditer(r"'--job', '([^']+)/job\.json'", log_path.read_text()):
        found.add(Path(match.group(1)))
    return found


def test_temp_dir_removed_by_default(make_runner: RunnerFactory, log_path: Path) -> None:
    make_runner().run(_req("ok"), timeout_s=60, log_path=log_path, cancel=CancelToken())
    (temp_dir,) = _temp_dirs_from_log(log_path)
    assert not temp_dir.exists()


@pytest.mark.parametrize("how", ["flag", "env"])
def test_keep_temp_keeps_job_and_result(
    make_runner: RunnerFactory, log_path: Path, how: str
) -> None:
    runner = (
        make_runner(keep_temp=True) if how == "flag" else make_runner(env={"GP_KEEP_TEMP": "1"})
    )
    runner.run(_req("ok"), timeout_s=60, log_path=log_path, cancel=CancelToken())
    (temp_dir,) = _temp_dirs_from_log(log_path)
    try:
        assert (temp_dir / "job.json").is_file()
        assert (temp_dir / "result.json").is_file()
        job = json.loads((temp_dir / "job.json").read_text())
        assert job["job"] == "probe"
        assert job["protocol_version"] == PROTOCOL_VERSION
    finally:
        for child in temp_dir.iterdir():
            child.unlink()
        temp_dir.rmdir()


def test_missing_executable_is_not_found_error(
    make_runner: RunnerFactory, log_path: Path, tmp_path: Path
) -> None:
    with pytest.raises(BlenderNotFoundError):
        make_runner(executable=tmp_path / "nope").run(
            _req("ok"), timeout_s=60, log_path=log_path, cancel=CancelToken()
        )


# --------------------------------------------------------------------------- result parsing edges
def _parse(req: JobRequest, payload: Any, tmp_path: Path) -> JobResult:
    result_json = tmp_path / "result.json"
    text = payload if isinstance(payload, str) else json.dumps(payload)
    result_json.write_text(text)
    return SubprocessBlenderRunner._parse_result(req, result_json, 0)


def _valid_payload(req: JobRequest, **overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "protocol_version": 1,
        "job": req.job,
        "job_id": req.job_id,
        "ok": True,
        "blender_version": "4.5.12",
        "duration_s": 0.5,
    }
    payload.update(overrides)
    return payload


def test_parse_rejects_non_object(tmp_path: Path) -> None:
    with pytest.raises(ProtocolDriftError, match="not a JSON object"):
        _parse(_req("ok"), [1, 2], tmp_path)


def test_parse_rejects_schema_violation(tmp_path: Path) -> None:
    req = _req("ok")
    with pytest.raises(ProtocolDriftError, match="JobResult contract") as info:
        _parse(req, _valid_payload(req, unexpected="field"), tmp_path)
    assert info.value.detail["errors"]


def test_parse_rejects_foreign_job_id(tmp_path: Path) -> None:
    req = _req("ok")
    with pytest.raises(ProtocolDriftError, match="different job"):
        _parse(req, _valid_payload(req, job_id="OTHER"), tmp_path)


def test_parse_rejects_not_ok_without_error(tmp_path: Path) -> None:
    req = _req("ok")
    with pytest.raises(ProtocolDriftError, match="no error"):
        _parse(req, _valid_payload(req, ok=False), tmp_path)


def test_parse_passes_traceback_through(tmp_path: Path) -> None:
    req = _req("ok")
    error = {"code": "E_BJ_UNHANDLED", "message": "boom", "traceback": "Traceback ..."}
    with pytest.raises(BlenderJobError) as info:
        _parse(req, _valid_payload(req, ok=False, error=error), tmp_path)
    assert info.value.traceback == "Traceback ..."
    assert info.value.job_code == "E_BJ_UNHANDLED"
