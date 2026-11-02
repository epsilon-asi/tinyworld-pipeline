"""Global options, the exception boundary, ``--json`` contract and library-root precedence."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tinyworld_pipeline import __version__
from tinyworld_pipeline.cli.app import build_app
from tinyworld_pipeline.cli.boundary import run_body
from tinyworld_pipeline.cli.state import (
    CliHooks,
    CliOutput,
    CliState,
    ErrorDocument,
    UnexpectedError,
)
from tinyworld_pipeline.cli.wiring import make_tripo_client, resolve_blender, secrets_from_mapping
from tinyworld_pipeline.domain.errors import CancelledError, UsageError
from tinyworld_pipeline.observability.console import Console
from tinyworld_pipeline.observability.logging import get_logging_handle

from .conftest import FIXTURE_LIBRARY, Harness, MakeHarness

SNAPSHOT_DIR = Path(__file__).parent / "snapshots"
HELP_COMMANDS: dict[str, list[str]] = {
    "gp": [],
    "init": ["init"],
    "new-model": ["new-model"],
    "validate": ["validate"],
    "run": ["run"],
    "status": ["status"],
    "clean": ["clean"],
    "tripo": ["tripo"],
    "tripo_balance": ["tripo", "balance"],
    "tripo_task": ["tripo", "task"],
    "tripo_usage": ["tripo", "usage"],
    "unity": ["unity"],
    "unity_sync": ["unity", "sync"],
    "doctor": ["doctor"],
}
SNAPSHOT_ENV = {"COLUMNS": "100", "TERM": "dumb", "TERMINAL_WIDTH": "100", "NO_COLOR": "1"}


# --------------------------------------------------------------------------- version / help
def test_version_flag_prints_version_and_exits_zero(harness: Harness) -> None:
    result = harness.invoke("--version")
    assert result.exit_code == 0
    assert result.stdout.strip() == f"gp {__version__}"


def test_no_args_shows_help(harness: Harness) -> None:
    result = harness.invoke()
    assert "Usage: gp" in result.output
    assert result.exit_code in (0, 2)


@pytest.mark.parametrize(("name", "args"), sorted(HELP_COMMANDS.items()))
def test_help_snapshot(name: str, args: list[str], harness: Harness) -> None:
    """``gp … --help`` rendered at 100 columns without colour matches the committed snapshot.

    Regenerate with ``GP_UPDATE_SNAPSHOTS=1 uv run pytest tests/unit/cli -k snapshot``.
    """
    result = harness.invoke("--no-color", *args, "--help", env=SNAPSHOT_ENV)
    assert result.exit_code == 0
    path = SNAPSHOT_DIR / f"{name}.txt"
    if os.environ.get("GP_UPDATE_SNAPSHOTS") == "1":
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(result.output, encoding="utf-8")
    assert path.exists(), f"missing snapshot {path}; run with GP_UPDATE_SNAPSHOTS=1"
    assert result.output == path.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- json contract
def test_json_mode_prints_exactly_one_document_on_stdout(lib: Path, tmp_path: Path) -> None:
    """With the default console, ``--json`` stdout is one parseable document (logs go to stderr)."""
    app = build_app(env={"PATH": os.environ.get("PATH", "")}, cwd=tmp_path)
    from typer.testing import CliRunner  # noqa: PLC0415 - local to keep conftest the only importer

    result = CliRunner().invoke(app, ["--json", "status", "--library", str(lib)])
    assert result.exit_code == 0, result.output
    document = json.loads(result.stdout)
    output = CliOutput.model_validate(document)
    assert output.command == "status"
    assert output.ok is True
    assert output.exit_code == 0
    assert [m["model_id"] for m in document["models"]] == ["cube_palette", "cube_rigged"]


def test_json_error_document_has_code_and_message(harness: Harness, tmp_path: Path) -> None:
    result, document = harness.json("status", "--library", str(tmp_path / "nowhere"))
    assert result.exit_code == 2
    assert document["error"]["code"] == "E_USAGE"
    assert "library.toml" in document["error"]["message"]
    assert document["error"]["hint"]


def test_cli_output_to_json_is_sorted_and_terminated() -> None:
    doc = CliOutput.success("x", b=1, a=2).to_json()
    assert doc.endswith("\n")
    assert doc.index('"a"') < doc.index('"b"') < doc.index('"command"')
    assert CliOutput.failure("x", UsageError("bad")).exit_code == 2
    assert CliOutput.failure("x", ErrorDocument(code="E_X", message="m")).exit_code == 1


# --------------------------------------------------------------------------- boundary
def _state(tmp_path: Path) -> CliState:
    return CliState(
        library=None,
        json=False,
        log_level=None,
        blender=None,
        no_color=True,
        env={},
        cwd=tmp_path,
        console=Console.recording(),
        hooks=CliHooks(None, make_tripo_client, resolve_blender, lambda: secrets_from_mapping({})),
    )


def test_run_body_maps_pipeline_error(tmp_path: Path) -> None:
    state = _state(tmp_path)

    def body() -> CliOutput:
        raise UsageError("nope", hint="do this")

    output = run_body("x", state, body)
    assert (output.exit_code, output.error and output.error.code) == (2, "E_USAGE")
    assert "E_USAGE: nope" in state.console.text  # type: ignore[attr-defined]
    assert "hint: do this" in state.console.text  # type: ignore[attr-defined]


def test_run_body_maps_keyboard_interrupt_to_130(tmp_path: Path) -> None:
    def body() -> CliOutput:
        raise KeyboardInterrupt

    output = run_body("x", _state(tmp_path), body)
    assert output.exit_code == CancelledError.exit_code == 130
    assert output.error is not None
    assert output.error.code == CancelledError.code


def test_run_body_maps_unexpected_to_exit_1(tmp_path: Path) -> None:
    def body() -> CliOutput:
        msg = "boom"
        raise RuntimeError(msg)

    output = run_body("x", _state(tmp_path), body)
    assert output.exit_code == 1
    assert output.error is not None
    assert output.error.code == UnexpectedError.code == "E_UNEXPECTED"
    assert "RuntimeError: boom" in output.error.message


def test_run_body_prints_returned_error_document(tmp_path: Path) -> None:
    state = _state(tmp_path)
    returned = CliOutput.failure(
        "x", ErrorDocument(code="E_DOCTOR", message="2 failed"), exit_code=3
    )
    output = run_body("x", state, lambda: returned)
    assert output is returned
    assert "E_DOCTOR: 2 failed" in state.console.text  # type: ignore[attr-defined]


def test_unexpected_error_through_the_cli_exits_1(make_harness: MakeHarness) -> None:
    def exploding_factory(*_: object) -> object:
        msg = "kaboom"
        raise RuntimeError(msg)

    harness = make_harness(env={"TRIPO_API_KEY": "sk-x"}, tripo_client_factory=exploding_factory)
    result, document = harness.json("tripo", "balance")
    assert result.exit_code == 1
    assert document["error"]["code"] == "E_UNEXPECTED"
    assert "kaboom" in document["error"]["message"]


def test_logging_handle_is_closed_after_each_invocation(harness: Harness, lib: Path) -> None:
    harness.invoke("status", "--library", str(lib))
    assert get_logging_handle() is None


def test_log_level_flag_and_env_are_accepted(make_harness: MakeHarness, lib: Path) -> None:
    harness = make_harness(env={"GP_LOGGING__LEVEL": "DEBUG"})
    assert harness.invoke("--log-level", "warning", "status", "--library", str(lib)).exit_code == 0
    assert harness.invoke("status", "--library", str(lib)).exit_code == 0


# --------------------------------------------------------------------------- library root
def test_library_root_flag_beats_env(make_harness: MakeHarness, lib: Path, tmp_path: Path) -> None:
    harness = make_harness(env={"GP_LIBRARY": str(tmp_path / "missing")})
    assert harness.invoke("--library", str(lib), "status").exit_code == 0
    assert harness.invoke("status").exit_code == 2  # env points nowhere


def test_library_root_env_beats_ancestor(
    make_harness: MakeHarness, lib: Path, tmp_path: Path
) -> None:
    other = tmp_path / "other"
    other.mkdir()
    harness = make_harness(env={"GP_LIBRARY": str(lib)}, cwd=other)
    assert harness.invoke("status").exit_code == 0


def test_library_root_ancestor_search(make_harness: MakeHarness, lib: Path) -> None:
    harness = make_harness(cwd=lib / "models" / "cube_rigged")
    result = harness.invoke("status")
    assert result.exit_code == 0
    assert "cube_rigged" in harness.text


def test_library_root_default_subdirectory(make_harness: MakeHarness, tmp_path: Path) -> None:
    import shutil  # noqa: PLC0415 - one-off copy

    shutil.copytree(FIXTURE_LIBRARY, tmp_path / "library")
    harness = make_harness(cwd=tmp_path)
    assert harness.invoke("status").exit_code == 0


def test_library_root_none_found_is_usage_error(make_harness: MakeHarness, tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    harness = make_harness(cwd=empty)
    result = harness.invoke("status")
    assert result.exit_code == 2
    assert "no asset library found" in harness.text
    assert "--library" in harness.text


def test_per_command_library_option_overrides_global(
    harness: Harness, lib: Path, tmp_path: Path
) -> None:
    result = harness.invoke("--library", str(tmp_path / "nope"), "status", "--library", str(lib))
    assert result.exit_code == 0
