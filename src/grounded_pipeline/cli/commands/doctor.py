"""``gp doctor`` (plan 08 §2): environment checks, exit 0 when all pass else 3."""

from __future__ import annotations

import platform
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import typer
from ulid import ULID

from grounded_pipeline.blender.locate import BlenderInfo, check_version
from grounded_pipeline.blender.protocol import JobRequest, ProbeData
from grounded_pipeline.blender.runner import SubprocessBlenderRunner, default_entry_script
from grounded_pipeline.cli import render
from grounded_pipeline.cli.boundary import command
from grounded_pipeline.cli.commands.common import LibraryOption, state_for
from grounded_pipeline.cli.render import Check
from grounded_pipeline.cli.state import CliOutput, CliState, ErrorDocument
from grounded_pipeline.cli.wiring import blender_problems, cli_overrides_from, load_library
from grounded_pipeline.domain.errors import PipelineError, UsageError
from grounded_pipeline.domain.library_config import BlenderSection, LibraryConfig
from grounded_pipeline.domain.ports import CancelToken
from grounded_pipeline.library.discovery import Library
from grounded_pipeline.library.validation import validate_library
from grounded_pipeline.pipeline.locks import blender_semaphore

COMMAND = "doctor"
MIN_PYTHON = (3, 12)
MIN_FREE_GB = 5
PROBE_TIMEOUT_S = 300
E_DOCTOR = "E_DOCTOR"


# --------------------------------------------------------------------------- checks
def check_python() -> Check:
    """Interpreter version ≥ 3.12."""
    ok = sys.version_info[:2] >= MIN_PYTHON
    return Check("python", ok, f"{platform.python_version()} ({sys.executable})")


def _probe(info: BlenderInfo, env: dict[str, str]) -> tuple[bool, str]:
    runner = SubprocessBlenderRunner(info.path, default_entry_script(), blender_semaphore(1), env)
    request = JobRequest(job="probe", job_id=str(ULID()), params={})
    with tempfile.TemporaryDirectory(prefix="gp-doctor-") as tmp:
        try:
            result = runner.run(
                request,
                timeout_s=PROBE_TIMEOUT_S,
                log_path=Path(tmp) / "probe.log",
                cancel=CancelToken(),
            )
        except PipelineError as exc:
            return False, f"probe failed: [{exc.code}] {exc.message}"
    try:
        data = ProbeData.model_validate(result.data)
    except ValueError:
        return True, f"probe ok (Blender {result.blender_version}; no capability data)"
    flags = ", ".join(
        f"{name}={'yes' if value else 'no'}"
        for name, value in (
            ("cycles", data.has_cycles),
            ("fbx", data.fbx_io_available),
            ("gltf", data.gltf_io_available),
        )
    )
    all_present = data.has_cycles and data.fbx_io_available and data.gltf_io_available
    return (
        all_present,
        f"probe ok (Blender {data.blender_version}, python {data.python_version}): {flags}",
    )


def check_blender(state: CliState, section: BlenderSection) -> list[Check]:
    """Executable found, version policy, and a real ``probe`` job round-trip."""
    info, problems = blender_problems(state, section)
    if info is None:
        return [Check("blender", False, problems[0].message if problems else "not found")]
    warnings = check_version(info.version, allow_unsupported=True)
    found = Check("blender", True, f"{info.version_string} at {info.path}")
    policy_ok = not problems
    policy = Check(
        "blender version",
        policy_ok,
        "; ".join([*(p.message for p in problems), *warnings]) or "supported (4.5 LTS policy)",
    )
    ok, detail = _probe(info, dict(state.env))
    return [found, policy, Check("blender probe", ok, detail)]


def check_tripo(state: CliState, config: LibraryConfig | None) -> list[Check]:
    """Key present (redacted) and a balance call."""
    secrets = state.secrets()
    if not secrets.has_tripo_key or secrets.tripo_api_key is None:
        return [Check("tripo key", False, "TRIPO_API_KEY not set")]
    section = config.tripo if config is not None else LibraryConfig().tripo
    key_check = Check("tripo key", True, secrets.redacted_tripo_key())
    client = state.hooks.tripo_client_factory(secrets.tripo_api_key, section)
    try:
        data = client.balance()
    except PipelineError as exc:
        return [key_check, Check("tripo balance", False, f"[{exc.code}] {exc.message}")]
    finally:
        client.close()
    return [
        key_check,
        Check("tripo balance", True, f"balance {data.balance:g}, frozen {data.frozen:g}"),
    ]


def check_library(state: CliState) -> tuple[Check, Library | None]:
    """Library valid when one is found; skipped (not failed) when none is."""
    try:
        library = load_library(state)
    except UsageError as exc:
        return Check("library", True, exc.message, skipped=True), None
    problems = validate_library(library, env=state.env, cli=cli_overrides_from(state))
    errors = [p for p in problems if p.severity == "error"]
    warnings = len(problems) - len(errors)
    if errors:
        codes = ", ".join(dict.fromkeys(p.code for p in errors))
        return Check(
            "library", False, f"{library.paths.root}: {len(errors)} error(s): {codes}"
        ), library
    return Check("library", True, f"{library.paths.root} ({warnings} warning(s))"), library


def check_disk(path: Path) -> Check:
    """Free space ≥ 5 GB at ``path``."""
    free_gb = shutil.disk_usage(path).free / 1024**3
    return Check("free disk", free_gb >= MIN_FREE_GB, f"{free_gb:.1f} GB free at {path}")


# --------------------------------------------------------------------------- command
def run_checks(state: CliState) -> list[Check]:
    """Every doctor check in display order."""
    library_check, library = check_library(state)
    config = library.config if library is not None else None
    checks = [check_python()]
    checks.extend(check_blender(state, config.blender if config else BlenderSection()))
    checks.extend(check_tripo(state, config))
    checks.append(library_check)
    checks.append(check_disk(library.paths.root if library is not None else state.cwd))
    return checks


def checks_json(checks: list[Check]) -> list[dict[str, Any]]:
    """``doctor.json`` rows."""
    return [
        {"name": c.name, "ok": c.ok, "skipped": c.skipped, "status": c.status, "detail": c.detail}
        for c in checks
    ]


@command(COMMAND)
def doctor_command(ctx: typer.Context, library_root: LibraryOption = None) -> CliOutput:
    """Check Python, Blender (+ probe job), the Tripo key (+ balance), the library and free disk."""
    state = state_for(ctx, library_root)
    checks = run_checks(state)
    render.doctor_checklist(state.console, checks)
    failed = [c for c in checks if not c.ok and not c.skipped]
    fields = {"checks": checks_json(checks)}
    if failed:
        names = ", ".join(c.name for c in failed)
        error = ErrorDocument(
            code=E_DOCTOR,
            message=f"{len(failed)} check(s) failed: {names}",
            hint="see the table above",
        )
        return CliOutput.failure(COMMAND, error, exit_code=3).model_copy(update=fields)
    return CliOutput.success(COMMAND, **fields)


def register(app: typer.Typer) -> None:
    """Attach to the root app."""
    app.command(COMMAND)(doctor_command)


__all__ = [
    "COMMAND",
    "E_DOCTOR",
    "check_blender",
    "check_disk",
    "check_library",
    "check_python",
    "check_tripo",
    "checks_json",
    "doctor_command",
    "register",
    "run_checks",
]
