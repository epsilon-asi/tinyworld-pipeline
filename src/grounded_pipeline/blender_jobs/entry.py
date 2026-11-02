"""Blender-side entry point (plan 06 §5, 13 §6, appendix B §1).

Invoked as ``blender -b --factory-startup --python-exit-code 2 --python entry.py -- --job job.json
--result result.json``. Only the standard library is imported at module level; the
``grounded_pipeline.blender_jobs`` package is imported after ``GP_SRC_DIR`` has been put on
``sys.path``, and ``bpy`` is used only when importable, so this file also runs under plain CPython
(unit tests use a fake job registered in ``jobs.JOBS``).

Exit codes: ``0`` result written (``ok`` may be false for expected failures), ``2`` unhandled
exception (result written with traceback when possible), ``3`` invalid job file / arguments.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import traceback
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from grounded_pipeline.blender_jobs.protocol import JobError, JobRequest

EXIT_OK = 0
EXIT_UNHANDLED = 2
EXIT_BAD_JOB_FILE = 3

_LOG = "[entry] "


def _bootstrap_sys_path() -> None:
    """Make ``grounded_pipeline.blender_jobs`` importable inside Blender's interpreter."""
    src_dir = os.environ.get("GP_SRC_DIR") or str(Path(__file__).resolve().parents[2])
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)


def _job_args(argv: Sequence[str]) -> list[str]:
    """Arguments after ``--`` (Blender passes everything after it untouched)."""
    if "--" in argv:
        return list(argv[argv.index("--") + 1 :])
    return list(argv[1:])


def _parse_args(args: Sequence[str]) -> tuple[Path | None, Path | None]:
    job: Path | None = None
    result: Path | None = None
    index = 0
    while index < len(args):
        flag = args[index]
        value = args[index + 1] if index + 1 < len(args) else None
        if flag == "--job" and value is not None:
            job = Path(value)
            index += 2
        elif flag == "--result" and value is not None:
            result = Path(value)
            index += 2
        else:
            index += 1
    return job, result


def _blender() -> Any:  # noqa: ANN401 - bpy module or None
    try:
        import bpy  # noqa: PLC0415 - after GP_SRC_DIR bootstrap
    except ImportError:
        return None
    return bpy


def _blender_version(bpy: Any) -> str:  # noqa: ANN401 - bpy module or None
    if bpy is None:
        return "unknown"
    return ".".join(str(part) for part in bpy.app.version)  # "4.5.12" (version_string adds "LTS")


def _write_result(path: Path, payload: dict[str, Any]) -> None:
    """Atomic write: temp file in the same directory, then ``os.replace``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)


def _load_request(job_path: Path) -> tuple[JobRequest | None, dict[str, Any] | None, str | None]:
    """Return ``(request, raw_dict, problem)``; ``request`` is None when the file is invalid."""
    from grounded_pipeline.blender_jobs import protocol  # noqa: PLC0415 - after GP_SRC_DIR bootstrap

    try:
        raw: Any = json.loads(job_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return None, None, f"cannot read job file {job_path}: {exc}"
    if not isinstance(raw, dict):
        return None, None, "job file is not a JSON object"
    try:
        request: JobRequest = protocol.from_dict(protocol.JobRequest, raw)
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        return None, raw, f"job file does not match JobRequest: {exc}"
    params: object = request.params  # from_dict does not type-check leaf values
    if not isinstance(params, dict):
        return None, raw, "job file: params must be an object"
    return request, raw, None


def _result_payload(
    *,
    job: str,
    job_id: str,
    ok: bool,
    blender_version: str,
    duration_s: float,
    data: dict[str, Any] | None = None,
    warnings: list[str] | None = None,
    error: JobError | None = None,
) -> dict[str, Any]:
    from grounded_pipeline.blender_jobs import protocol  # noqa: PLC0415 - after GP_SRC_DIR bootstrap

    result = protocol.JobResult(
        job=job,
        job_id=job_id,
        ok=ok,
        blender_version=blender_version,
        duration_s=duration_s,
        data=data or {},
        warnings=warnings or [],
        error=error,
    )
    payload: dict[str, Any] = protocol.to_dict(result)
    return payload


def _execute(request: JobRequest, bpy: Any) -> tuple[dict[str, Any], int]:  # noqa: ANN401
    """Run the job; return ``(result payload, exit code)``."""
    from grounded_pipeline.blender_jobs import protocol  # noqa: PLC0415 - after GP_SRC_DIR bootstrap
    from grounded_pipeline.blender_jobs.errors import JobFailure  # noqa: PLC0415 - after GP_SRC_DIR bootstrap
    from grounded_pipeline.blender_jobs.jobs import JOBS  # noqa: PLC0415 - after GP_SRC_DIR bootstrap

    started = time.perf_counter()
    version = _blender_version(bpy)

    def finish(
        *,
        ok: bool,
        data: dict[str, Any] | None = None,
        warnings: list[str] | None = None,
        error: JobError | None = None,
    ) -> dict[str, Any]:
        return _result_payload(
            job=request.job,
            job_id=request.job_id,
            ok=ok,
            blender_version=version,
            duration_s=round(time.perf_counter() - started, 4),
            data=data,
            warnings=warnings,
            error=error,
        )

    if request.protocol_version != protocol.PROTOCOL_VERSION:
        error = protocol.JobError(
            code="E_BJ_PROTOCOL",
            message=(
                f"protocol_version {request.protocol_version} != {protocol.PROTOCOL_VERSION}"
                " (orchestrator and blender_jobs must come from the same install)"
            ),
            detail={"got": request.protocol_version, "expected": protocol.PROTOCOL_VERSION},
        )
        return finish(ok=False, error=error), EXIT_OK

    job_fn = JOBS.get(request.job)
    if job_fn is None:
        error = protocol.JobError(
            code=protocol.E_BJ_UNKNOWN_JOB,
            message=f"unknown job {request.job!r}",
            detail={"job": request.job, "known": sorted(JOBS)},
        )
        return finish(ok=False, error=error), EXIT_OK

    try:
        if bpy is not None:
            bpy.ops.wm.read_factory_settings(use_empty=True)
        data, warnings = job_fn(dict(request.params))
    except JobFailure as failure:
        sys.stderr.write(f"{_LOG}job {request.job} failed: {failure.code}: {failure.message}\n")
        error = protocol.JobError(
            code=failure.code, message=failure.message, detail=dict(failure.detail)
        )
        return finish(ok=False, error=error), EXIT_OK
    except Exception as exc:
        tb = traceback.format_exc()
        sys.stderr.write(f"{_LOG}unhandled exception in job {request.job}:\n{tb}")
        error = protocol.JobError(
            code=protocol.E_BJ_UNHANDLED,
            message=f"{type(exc).__name__}: {exc}",
            detail={"exception_type": type(exc).__name__},
            traceback=tb,
        )
        return finish(ok=False, error=error), EXIT_UNHANDLED
    return finish(ok=True, data=dict(data), warnings=list(warnings)), EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    """Run one job described by ``--job`` and write ``--result``; return the process exit code."""
    _bootstrap_sys_path()
    args = _job_args(sys.argv if argv is None else argv)
    job_path, result_path = _parse_args(args)
    if job_path is None or result_path is None:
        sys.stderr.write(f"{_LOG}usage: entry.py -- --job <job.json> --result <result.json>\n")
        return EXIT_BAD_JOB_FILE

    from grounded_pipeline.blender_jobs import protocol  # noqa: PLC0415 - after GP_SRC_DIR bootstrap

    bpy = _blender()
    request, raw, problem = _load_request(job_path)
    if request is None:
        sys.stderr.write(f"{_LOG}{problem}\n")
        raw = raw or {}
        job = raw.get("job")
        job_id = raw.get("job_id")
        if isinstance(job, str) and isinstance(job_id, str):
            payload = _result_payload(
                job=job,
                job_id=job_id,
                ok=False,
                blender_version=_blender_version(bpy),
                duration_s=0.0,
                error=protocol.JobError(
                    code=protocol.E_BJ_BAD_PARAMS, message=problem or "invalid job file"
                ),
            )
            _write_result(result_path, payload)
        return EXIT_BAD_JOB_FILE

    logging.basicConfig(
        level=getattr(logging, request.log_level, logging.INFO),
        stream=sys.stderr,
        format="[%(name)s] %(levelname)s %(message)s",
    )
    sys.stderr.write(f"{_LOG}job={request.job} job_id={request.job_id}\n")
    payload, exit_code = _execute(request, bpy)
    _write_result(result_path, payload)
    sys.stderr.write(f"{_LOG}ok={payload['ok']} exit={exit_code} result={result_path}\n")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
