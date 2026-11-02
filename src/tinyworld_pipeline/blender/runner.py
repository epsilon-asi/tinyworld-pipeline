"""Blender subprocess runner (plan 13 §6, §8; 19 §5).

Every Blender job is a fresh ``blender -b --factory-startup --python entry.py`` process driven
through a temp directory holding ``job.json`` and ``result.json``. The runner owns the process
lifetime: it polls for completion while watching the :class:`CancelToken` and the timeout, kills
the whole process group when either fires, and translates the outcome into the typed errors of
plan 14 §2.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import IO, Any, Protocol

import structlog
from pydantic import ValidationError

import tinyworld_pipeline
from tinyworld_pipeline.blender.protocol import PROTOCOL_VERSION, JobRequest, JobResult
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

log = structlog.get_logger(__name__)

LOG_TAIL_LINES = 40
_IS_WINDOWS = sys.platform.startswith("win")


class BlenderRunner(Protocol):
    """Port through which stages run Blender jobs (fake: ``tests/fakes/fake_blender_runner.py``)."""

    def run(
        self, req: JobRequest, *, timeout_s: int, log_path: Path, cancel: CancelToken
    ) -> JobResult:
        """Run ``req`` to completion and return its result.

        Raises:
            BlenderTimeoutError: The job exceeded ``timeout_s`` and was killed.
            CancelledError: ``cancel`` was set while the job was running; the job was killed.
            BlenderCrashError: Blender exited non-zero without a result.
            BlenderNoResultError: Blender exited zero without a result.
            BlenderJobError: The job reported ``ok=False``.
            ProtocolDriftError: The result could not be parsed or has a foreign protocol version.
        """
        ...


def default_src_dir() -> Path:
    """Directory containing the ``tinyworld_pipeline`` package (what ``GP_SRC_DIR`` points to)."""
    return Path(tinyworld_pipeline.__file__).resolve().parent.parent


def default_entry_script() -> Path:
    """Path of the Blender-side entry point shipped with this package."""
    return Path(tinyworld_pipeline.__file__).resolve().parent / "blender_jobs" / "entry.py"


def _log_tail(path: Path, lines: int = LOG_TAIL_LINES) -> list[str]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    return text.splitlines()[-lines:]


class SubprocessBlenderRunner:
    """Run Blender jobs as headless subprocesses (plan 13 §6).

    Args:
        executable: Blender executable (see ``locate.locate_blender``).
        entry_script: Path of ``blender_jobs/entry.py``.
        semaphore: Caps concurrent Blender processes; held for the whole subprocess lifetime.
        env: Base environment for the child. ``GP_SRC_DIR`` and ``PYTHONDONTWRITEBYTECODE`` are
            added. ``GP_KEEP_TEMP=1`` keeps the per-job temp directory.
        src_dir: Directory containing the ``tinyworld_pipeline`` package; defaults to this install.
        keep_temp: Keep the per-job temp directory (``job.json``/``result.json``) for debugging.
        poll_interval_s: How often the child is polled for exit/cancel/timeout.
        kill_grace_s: Seconds between SIGTERM and SIGKILL on timeout.
    """

    def __init__(
        self,
        executable: Path,
        entry_script: Path,
        semaphore: threading.BoundedSemaphore,
        env: Mapping[str, str],
        *,
        src_dir: Path | None = None,
        keep_temp: bool = False,
        poll_interval_s: float = 0.5,
        kill_grace_s: float = 10.0,
    ) -> None:
        self._executable = Path(executable)
        self._entry_script = Path(entry_script)
        self._semaphore = semaphore
        self._env = dict(env)
        self._src_dir = Path(src_dir) if src_dir is not None else default_src_dir()
        self._keep_temp = keep_temp or self._env.get("GP_KEEP_TEMP") == "1"
        self._poll_interval_s = poll_interval_s
        self._kill_grace_s = kill_grace_s

    # ------------------------------------------------------------------ public
    def run(
        self, req: JobRequest, *, timeout_s: int, log_path: Path, cancel: CancelToken
    ) -> JobResult:
        """Run one job; see :class:`BlenderRunner` for the raised errors."""
        with self._semaphore:
            return self._run_exclusive(req, timeout_s=timeout_s, log_path=log_path, cancel=cancel)

    # ------------------------------------------------------------------ internals
    def command(self, job_json: Path, result_json: Path) -> list[str]:
        """Argv for one job (plan 13 §6)."""
        return [
            str(self._executable),
            "-b",
            "--factory-startup",
            "--python-exit-code",
            "2",
            "--python",
            str(self._entry_script),
            "--",
            "--job",
            str(job_json),
            "--result",
            str(result_json),
        ]

    def child_env(self) -> dict[str, str]:
        """Environment passed to the child."""
        return {
            **self._env,
            "GP_SRC_DIR": str(self._src_dir),
            "PYTHONDONTWRITEBYTECODE": "1",
        }

    def _run_exclusive(
        self, req: JobRequest, *, timeout_s: int, log_path: Path, cancel: CancelToken
    ) -> JobResult:
        if cancel.is_set():
            raise CancelledError(f"Cancelled before Blender job {req.job} started.")
        tmp_dir = Path(tempfile.mkdtemp(prefix=f"gp-blender-{req.job}-"))
        try:
            job_json = tmp_dir / "job.json"
            result_json = tmp_dir / "result.json"
            job_json.write_text(req.model_dump_json(indent=2), encoding="utf-8")
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as log_file:
                exit_code = self._spawn_and_wait(
                    req, job_json, result_json, log_file, timeout_s=timeout_s, cancel=cancel
                )
            return self._interpret(req, exit_code, result_json, log_path)
        finally:
            if self._keep_temp:
                log.debug("blender.temp_kept", job_id=req.job_id, temp_dir=str(tmp_dir))
            else:
                shutil.rmtree(tmp_dir, ignore_errors=True)

    def _spawn_and_wait(
        self,
        req: JobRequest,
        job_json: Path,
        result_json: Path,
        log_file: IO[str],
        *,
        timeout_s: int,
        cancel: CancelToken,
    ) -> int:
        cmd = self.command(job_json, result_json)
        log_file.write(f"[runner] job={req.job} job_id={req.job_id}\n[runner] argv={cmd}\n")
        log_file.flush()
        popen_kwargs: dict[str, Any] = {
            "stdout": log_file,
            "stderr": subprocess.STDOUT,
            "stdin": subprocess.DEVNULL,
            "env": self.child_env(),
            "cwd": str(job_json.parent),
        }
        if _IS_WINDOWS:  # pragma: no cover - Windows host deferred (ADR-0002)
            popen_kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        else:
            popen_kwargs["start_new_session"] = True
        started = time.monotonic()
        try:
            proc = subprocess.Popen(cmd, **popen_kwargs)  # noqa: S603 - argv is a fixed list
        except OSError as exc:
            raise BlenderNotFoundError(
                f"Cannot start Blender at {self._executable}: {exc}",
                detail={"path": str(self._executable)},
                hint="Check [blender].executable / --blender.",
            ) from exc
        log.info(
            "blender.spawn",
            job=req.job,
            job_id=req.job_id,
            pid=proc.pid,
            timeout_s=timeout_s,
            log_path=str(log_file.name),
        )
        outcome = "exit"
        try:
            exit_code = self._wait(proc, started, timeout_s=timeout_s, cancel=cancel)
        except BlenderTimeoutError:
            outcome = "timeout"
            raise
        except CancelledError:
            outcome = "cancelled"
            raise
        finally:
            log.info(
                "blender.exit",
                job=req.job,
                job_id=req.job_id,
                pid=proc.pid,
                exit_code=proc.returncode,
                outcome=outcome,
                duration_s=round(time.monotonic() - started, 3),
            )
        return exit_code

    def _wait(
        self, proc: subprocess.Popen[bytes], started: float, *, timeout_s: int, cancel: CancelToken
    ) -> int:
        deadline = started + timeout_s
        while True:
            try:
                return proc.wait(timeout=self._poll_interval_s)
            except subprocess.TimeoutExpired:
                pass
            if cancel.is_set():
                self._kill_tree(proc)
                raise CancelledError("Blender job cancelled.", detail={"pid": proc.pid})
            if time.monotonic() >= deadline:
                self._terminate_tree(proc)
                raise BlenderTimeoutError(
                    f"Blender job exceeded {timeout_s}s and was killed.",
                    detail={"timeout_s": timeout_s, "pid": proc.pid},
                    hint="Raise [blender].job_timeout_s or simplify the model.",
                )

    def _terminate_tree(self, proc: subprocess.Popen[bytes]) -> None:
        """SIGTERM the process group, then SIGKILL after the grace period."""
        if _IS_WINDOWS:  # pragma: no cover - Windows host deferred (ADR-0002)
            self._kill_tree(proc)
            return
        self._signal_group(proc, signal.SIGTERM)
        try:
            proc.wait(timeout=self._kill_grace_s)
        except subprocess.TimeoutExpired:
            self._kill_tree(proc)

    def _kill_tree(self, proc: subprocess.Popen[bytes]) -> None:
        """Kill the whole process tree and reap it."""
        if _IS_WINDOWS:  # pragma: no cover - Windows host deferred (ADR-0002)
            taskkill = ["taskkill", "/PID", str(proc.pid), "/T", "/F"]
            subprocess.run(taskkill, check=False, capture_output=True)  # noqa: S603 - fixed argv
        else:
            self._signal_group(proc, signal.SIGKILL)
        try:
            proc.wait(timeout=self._kill_grace_s)
        except subprocess.TimeoutExpired:  # pragma: no cover - unkillable process
            log.warning("blender.kill_failed", pid=proc.pid)

    @staticmethod
    def _signal_group(proc: subprocess.Popen[bytes], sig: signal.Signals) -> None:
        if proc.poll() is not None:
            return
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:  # pragma: no cover - raced with natural exit
            pass
        except PermissionError:  # pragma: no cover - defensive
            proc.send_signal(sig)

    def _interpret(
        self, req: JobRequest, exit_code: int, result_json: Path, log_path: Path
    ) -> JobResult:
        if not result_json.exists():
            if exit_code == 0:
                raise BlenderNoResultError(
                    "Blender exited 0 but wrote no result.json.",
                    detail={"log_path": str(log_path), "log_tail": _log_tail(log_path)},
                )
            raise BlenderCrashError(
                f"Blender exited with code {exit_code} without writing result.json.",
                detail={
                    "exit_code": exit_code,
                    "log_path": str(log_path),
                    "log_tail": _log_tail(log_path),
                },
                hint=f"See {log_path}",
            )
        return self._parse_result(req, result_json, exit_code)

    @staticmethod
    def _parse_result(req: JobRequest, result_json: Path, exit_code: int) -> JobResult:
        raw_text = result_json.read_text(encoding="utf-8")
        try:
            raw: Any = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise ProtocolDriftError(
                "result.json is not valid JSON.",
                detail={"job_id": req.job_id, "excerpt": raw_text[:200]},
                cause=exc,
            ) from exc
        if not isinstance(raw, dict):
            raise ProtocolDriftError(
                "result.json is not a JSON object.", detail={"job_id": req.job_id}
            )
        version = raw.get("protocol_version")
        if version != PROTOCOL_VERSION:
            raise ProtocolDriftError(
                f"Blender job protocol version mismatch: got {version!r}, "
                f"expected {PROTOCOL_VERSION}.",
                detail={"job_id": req.job_id, "got": version, "expected": PROTOCOL_VERSION},
                hint="Orchestrator and blender_jobs must come from the same install.",
            )
        try:
            result = JobResult.model_validate(raw)
        except ValidationError as exc:
            raise ProtocolDriftError(
                "result.json does not match the JobResult contract.",
                detail={"job_id": req.job_id, "errors": exc.errors(include_url=False)},
                cause=exc,
            ) from exc
        if result.job_id != req.job_id or result.job != req.job:
            raise ProtocolDriftError(
                "result.json belongs to a different job.",
                detail={"expected": [req.job, req.job_id], "got": [result.job, result.job_id]},
            )
        if not result.ok:
            if result.error is None:
                raise ProtocolDriftError(
                    "result.json has ok=false but no error.", detail={"job_id": req.job_id}
                )
            raise BlenderJobError(
                result.error.code,
                result.error.message,
                detail={**result.error.detail, "job_id": req.job_id, "exit_code": exit_code},
                traceback=result.error.traceback,
            )
        return result
