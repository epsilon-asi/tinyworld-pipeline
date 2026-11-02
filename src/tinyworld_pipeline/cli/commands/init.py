"""``gp init [PATH] [--force]`` (plan 08 §2)."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from tinyworld_pipeline.cli import render
from tinyworld_pipeline.cli.boundary import command
from tinyworld_pipeline.cli.commands.common import LibraryOption, state_for
from tinyworld_pipeline.cli.state import CliOutput
from tinyworld_pipeline.library.paths import DEFAULT_LIBRARY_DIRNAME, LIBRARY_ENV_VAR
from tinyworld_pipeline.library.scaffold import init_library

COMMAND = "init"


@command(COMMAND)
def init_command(
    ctx: typer.Context,
    path: Annotated[
        Path | None,
        typer.Argument(help="Library root to create (default: --library, $GP_LIBRARY, ./library)."),
    ] = None,
    force: Annotated[
        bool, typer.Option("--force", help="Overwrite an existing library.toml (other files kept).")
    ] = False,
    library_root: LibraryOption = None,
) -> CliOutput:
    """Scaffold a new asset library (library.toml, style/, models/, .gitignore)."""
    state = state_for(ctx, library_root)
    target = path or state.library
    if target is None:
        from_env = state.env.get(LIBRARY_ENV_VAR)
        target = Path(from_env) if from_env else state.cwd / DEFAULT_LIBRARY_DIRNAME
    paths = init_library(target.expanduser().resolve(), force=force)
    render.notice(state.console, f"initialised library at {paths.root}")
    render.message(state.console, "next: gp new-model <id> --source <file.fbx>", style="dim")
    return CliOutput.success(COMMAND, path=str(paths.root), forced=force)


def register(app: typer.Typer) -> None:
    """Attach to the root app."""
    app.command(COMMAND)(init_command)


__all__ = ["COMMAND", "init_command", "register"]
