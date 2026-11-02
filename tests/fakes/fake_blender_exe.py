"""Fake Blender executable for ``SubprocessBlenderRunner`` tests (plan 15 §3.3).

Run as ``python fake_blender_exe.py ...`` through the launcher written by :func:`write_launcher`.
Standard library only (it runs under ``sys.executable`` in a bare environment).

Behaviours:

* ``--version`` → prints ``Blender <FAKE_BLENDER_VERSION or 4.5.12>`` and exits 0.
* ``-- --job <job.json> --result <result.json>`` → reads ``job.json`` and acts on
  ``params["__fake"]``:

  - ``"ok"`` (default): valid ``JobResult`` echoing ``params`` in ``data["echo"]``.
  - ``"job_error"``: ``ok=False`` with code ``E_BJ_SHAPEKEYS``.
  - ``"crash"``: writes to stderr and exits 1 without a result.
  - ``"no_result"``: exits 0 without a result.
  - ``"sleep:N"``: sleeps N seconds, then behaves like ``"ok"``.
  - ``"bad_json"``: writes garbage into ``result.json``.
  - ``"wrong_protocol"``: valid result with ``protocol_version`` 99.

Diagnostics written to stdout (and therefore the runner's log file): ``FAKE_PID <pid>``,
``FAKE_ARGV <argv>``, ``FAKE_ENV <name>=<value>`` for the variables the runner must set. If
``params["__stamp_dir"]`` is set, ``<stamp_dir>/<job_id>.json`` records start/end monotonic times
(for the concurrency test).
"""

from __future__ import annotations

import json
import os
import stat
import sys
import time
from pathlib import Path
from typing import Any

DEFAULT_VERSION = "4.5.12"
PROTOCOL_VERSION = 1
_ENV_OF_INTEREST = ("GP_SRC_DIR", "PYTHONDONTWRITEBYTECODE")


def write_launcher(directory: Path) -> Path:
    """Write an executable wrapper that runs this file under ``sys.executable``; return its path.

    The only platform-specific piece of the fake (ADR-0002): a ``.cmd`` file on Windows, a ``sh``
    script elsewhere.
    """
    script = Path(__file__).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    if sys.platform.startswith("win"):  # pragma: no cover - Windows host deferred
        launcher = directory / "fake_blender.cmd"
        launcher.write_text(f'@echo off\r\n"{sys.executable}" "{script}" %*\r\n', encoding="utf-8")
        return launcher
    launcher = directory / "fake_blender"
    launcher.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n', encoding="utf-8")
    launcher.chmod(launcher.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return launcher


def _result(job: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "job": job["job"],
        "job_id": job["job_id"],
        "ok": True,
        "blender_version": os.environ.get("FAKE_BLENDER_VERSION", DEFAULT_VERSION),
        "duration_s": 0.01,
        "data": {"echo": job.get("params", {})},
        "warnings": [],
        "error": None,
    }
    base.update(overrides)
    return base


def _write(path: Path, payload: Any) -> None:
    text = payload if isinstance(payload, str) else json.dumps(payload, indent=2)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _stamp(job: dict[str, Any], key: str, value: float) -> None:
    stamp_dir = job.get("params", {}).get("__stamp_dir")
    if not stamp_dir:
        return
    path = Path(stamp_dir) / f"{job['job_id']}.json"
    data: dict[str, float] = json.loads(path.read_text()) if path.exists() else {}
    data[key] = value
    path.write_text(json.dumps(data))


def _run_job(args: list[str]) -> int:  # noqa: PLR0911 - one return per scripted behaviour
    job_path = Path(args[args.index("--job") + 1])
    result_path = Path(args[args.index("--result") + 1])
    job: dict[str, Any] = json.loads(job_path.read_text(encoding="utf-8"))
    fake = str(job.get("params", {}).get("__fake", "ok"))
    _stamp(job, "start", time.monotonic())
    sys.stdout.write(f"FAKE_JOB {job['job']} {job['job_id']} mode={fake}\n")
    sys.stdout.flush()

    if fake.startswith("sleep:"):
        time.sleep(float(fake.split(":", 1)[1]))
        fake = "ok"

    try:
        if fake == "ok":
            _write(result_path, _result(job))
            return 0
        if fake == "job_error":
            error = {
                "code": "E_BJ_SHAPEKEYS",
                "message": "mesh has shape keys",
                "detail": {"meshes": ["Body"]},
                "traceback": None,
            }
            _write(result_path, _result(job, ok=False, data={}, error=error))
            return 0
        if fake == "crash":
            sys.stderr.write("Segmentation fault (fake)\n")
            sys.stderr.flush()
            return 1
        if fake == "no_result":
            return 0
        if fake == "bad_json":
            _write(result_path, "{ this is not json")
            return 0
        if fake == "wrong_protocol":
            _write(result_path, _result(job, protocol_version=99))
            return 0
        sys.stderr.write(f"fake: unknown __fake mode {fake!r}\n")
        return 4
    finally:
        _stamp(job, "end", time.monotonic())


def main(argv: list[str]) -> int:
    """Entry point; ``argv`` excludes the program name."""
    if "--version" in argv:
        version = os.environ.get("FAKE_BLENDER_VERSION", DEFAULT_VERSION)
        sys.stdout.write(f"Blender {version}\n\tbuild date: fake\n")
        return 0
    sys.stdout.write(f"FAKE_PID {os.getpid()}\nFAKE_ARGV {argv}\n")
    for name in _ENV_OF_INTEREST:
        sys.stdout.write(f"FAKE_ENV {name}={os.environ.get(name, '<unset>')}\n")
    sys.stdout.flush()
    if "--" not in argv:
        sys.stderr.write("fake: expected -- --job <path> --result <path>\n")
        return 4
    return _run_job(argv[argv.index("--") + 1 :])


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
