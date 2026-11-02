"""``gp unity sync [--model ID]... [--dest PATH]`` (plan 08 §2, 12 §3)."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Any

import typer
from ulid import ULID

from tinyworld_pipeline.cli import render
from tinyworld_pipeline.cli.boundary import command
from tinyworld_pipeline.cli.commands.common import LibraryOption, ModelOption, state_for
from tinyworld_pipeline.cli.state import CliOutput
from tinyworld_pipeline.cli.wiring import load_library
from tinyworld_pipeline.domain.errors import UsageError
from tinyworld_pipeline.library.validation import raise_if_errors
from tinyworld_pipeline.unity.sync import copy_model, validate_dest

COMMAND = "unity sync"

unity_app = typer.Typer(name="unity", help="Unity hand-off helpers.", no_args_is_help=True)


@command(COMMAND)
def sync_command(
    ctx: typer.Context,
    model: ModelOption = None,
    dest: Annotated[
        Path | None,
        typer.Option(
            "--dest",
            help="Unity import folder (default: library.unity_import_dir).",
            metavar="PATH",
        ),
    ] = None,
    library_root: LibraryOption = None,
) -> CliOutput:
    """Copy output/<id>/ into <dest>/<id>/ atomically, preserving Unity .meta files."""
    state = state_for(ctx, library_root)
    library = load_library(state)
    destination = dest or library.config.library.unity_import_dir
    if destination is None:
        raise UsageError(
            "no Unity destination",
            hint="pass --dest PATH or set [library].unity_import_dir in library.toml",
        )
    destination = destination.expanduser().resolve()
    raise_if_errors(validate_dest(destination))
    run_token = str(ULID())
    rows: list[dict[str, Any]] = []
    for model_id in library.select(ids=model or None):
        output_dir = library.paths.model(model_id).output
        if not output_dir.is_dir():
            rows.append({"model_id": model_id, "synced": False, "reason": "no output/ folder"})
            render.message(state.console, f"  {model_id}: skipped (no output/)", style="yellow")
            continue
        report = copy_model(output_dir, destination, model_id, run_token=run_token)
        rows.append({"model_id": model_id, "synced": True, **asdict(report)})
        render.message(
            state.console,
            f"  {model_id}: {report.copied} file(s) copied, {report.removed} stale removed, "
            f"{report.meta_preserved} .meta kept",
        )
    render.notice(state.console, f"synced to {destination}")
    return CliOutput.success(COMMAND, dest=str(destination), run_token=run_token, models=rows)


def register(app: typer.Typer) -> None:
    """Attach ``gp unity …`` to the root app."""
    unity_app.command("sync")(sync_command)
    app.add_typer(unity_app, name="unity")


__all__ = ["COMMAND", "register", "sync_command", "unity_app"]
