from __future__ import annotations

import io
from pathlib import Path

from rich.console import Console as RichConsole

from grounded_pipeline.domain.errors import (
    BlenderJobError,
    PipelineError,
    Problem,
    TripoAuthError,
)
from grounded_pipeline.observability.console import Console, RecordingConsole

PROBLEMS = [
    Problem(
        code="E_FRONTMATTER",
        severity="error",
        model_id="hero",
        path="model.md:texture.quality",
        message="unknown quality 'ultra'",
        hint="use one of: standard, detailed",
    ),
    Problem(code="W_NO_STYLE", severity="warning", model_id="hero", message="no style.md"),
    Problem(
        code="E_MISSING_FBX",
        severity="error",
        model_id="crate",
        message="no source FBX",
        hint="drop crate.fbx into the model folder",
    ),
    Problem(code="E_LIBRARY_TOML", severity="error", message="library.toml missing"),
]


def test_recording_factory_returns_plain_text_console() -> None:
    console = Console.recording()
    assert isinstance(console, RecordingConsole)
    assert isinstance(console.rich, RichConsole)
    console.print("hello [bold]world[/bold]", style="red")
    assert console.text == "hello world\n"
    assert "\x1b[" not in console.text
    assert console.clear().text == ""


def test_print_problems_grouped_by_model_with_hints() -> None:
    console = Console.recording()
    console.print_problems(PROBLEMS)
    text = console.text
    lines = text.splitlines()

    assert lines[0] == "hero"
    assert "[E_FRONTMATTER] model.md:texture.quality — unknown quality 'ultra'" in lines[1]
    assert lines[1].lstrip().startswith("✗")
    assert lines[2].strip() == "hint: use one of: standard, detailed"
    assert "[W_NO_STYLE] no style.md" in lines[3]
    assert lines[3].lstrip().startswith("!")
    assert lines[4] == "crate"
    assert "[E_MISSING_FBX] no source FBX" in lines[5]
    assert "hint: drop crate.fbx" in lines[6]
    assert lines[7] == "(library)"
    assert "[E_LIBRARY_TOML] library.toml missing" in lines[8]
    assert text.index("hero") < text.index("crate") < text.index("(library)")


def test_print_problems_empty_prints_nothing() -> None:
    console = Console.recording()
    console.print_problems([])
    assert console.text == ""


def test_print_warnings_grouped_and_skips_empty_groups() -> None:
    console = Console.recording()
    console.print_warnings({"hero": ["rfc: bone count differs"], "crate": [], "tree": ["a", "b"]})
    lines = console.text.splitlines()
    assert lines[0] == "Warnings"
    assert lines[1].strip() == "hero"
    assert lines[2].strip() == "! rfc: bone count differs"
    assert "crate" not in console.text
    assert lines[3].strip() == "tree"
    assert lines[4].strip() == "! a"
    assert lines[5].strip() == "! b"


def test_print_warnings_nothing_when_all_empty() -> None:
    console = Console.recording()
    console.print_warnings({"hero": []})
    console.print_warnings({})
    assert console.text == ""


def test_print_error_shows_code_message_hint_and_log_path(tmp_path: Path) -> None:
    console = Console.recording()
    log_path = tmp_path / "hero" / ".pipeline" / "logs" / "prepare-01H.log"
    err = TripoAuthError("API key rejected", hint="check TRIPO_API_KEY")
    console.print_error(err, log_path=log_path)
    lines = console.text.splitlines()
    assert lines[0] == "✗ E_TRIPO_AUTH: API key rejected"
    assert lines[1].strip() == "hint: check TRIPO_API_KEY"
    assert lines[2].strip() == f"log:  {log_path}"


def test_print_error_without_hint_or_log_path_and_job_code_passthrough() -> None:
    console = Console.recording()
    console.print_error(BlenderJobError("E_BJ_NO_MESH", "no mesh objects"))
    assert console.text == "✗ E_BJ_NO_MESH: no mesh objects\n"


def test_table_renders_title_columns_and_rows() -> None:
    console = Console.recording()
    rows = [("hero", "prepare", "ok"), ("crate", 2, None)]
    console.table("Models", ["model", "stage", "status"], rows)
    text = console.text
    assert "Models" in text
    for cell in ("model", "stage", "status", "hero", "prepare", "ok", "crate", "2", "None"):
        assert cell in text


def test_rule_prints_title() -> None:
    console = Console.recording()
    console.rule("Summary")
    assert "Summary" in console.text


def test_quiet_suppresses_informational_but_keeps_problems_and_errors() -> None:
    console = Console.recording(quiet=True)
    console.print("info line")
    console.rule("rule")
    console.table("T", ["a"], [("1",)])
    console.print_warnings({"hero": ["w"]})
    assert console.text == ""
    assert not console.verbose_allowed
    assert not console.silent

    console.print("important line", important=True)
    console.print_problems(PROBLEMS[:1])
    console.print_error(PipelineError("failed"), log_path=None)
    text = console.text
    assert "important line" in text
    assert "E_FRONTMATTER" in text
    assert "✗ E_PIPELINE: failed" in text


def test_json_mode_silences_everything() -> None:
    console = Console.recording(json_mode=True)
    console.print("info", important=True)
    console.rule("r")
    console.table("T", ["a"], [("1",)])
    console.print_warnings({"hero": ["w"]})
    console.print_problems(PROBLEMS)
    console.print_error(PipelineError("failed"), log_path=Path("/x.log"))
    assert console.text == ""
    assert console.silent


def test_console_writes_to_given_file_and_honours_no_color() -> None:
    buffer = io.StringIO()
    console = Console(file=buffer, no_color=True, force_terminal=True, width=80)
    console.print("[red]x[/red]")
    assert buffer.getvalue() == "x\n"
    assert console.no_color
