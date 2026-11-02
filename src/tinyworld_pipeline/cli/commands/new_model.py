"""``gp new-model <model_id> --source FILE …`` (plan 08 §2)."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from tinyworld_pipeline.cli import render
from tinyworld_pipeline.cli.boundary import command
from tinyworld_pipeline.cli.commands.common import LibraryOption, state_for
from tinyworld_pipeline.cli.state import CliOutput
from tinyworld_pipeline.cli.wiring import resolve_library_root
from tinyworld_pipeline.domain.errors import UsageError
from tinyworld_pipeline.library.scaffold import CopyMode, new_model

COMMAND = "new-model"


def _mode(*, copy: bool, move: bool, link: bool) -> CopyMode:
    chosen = [name for name, flag in (("copy", copy), ("move", move), ("link", link)) if flag]
    if len(chosen) > 1:
        raise UsageError(
            "--copy, --move and --link are mutually exclusive", hint="pass at most one of them"
        )
    if chosen == ["move"]:
        return "move"
    if chosen == ["link"]:
        return "link"
    return "copy"


@command(COMMAND)
def new_model_command(  # noqa: PLR0917 - typer options are positional by convention
    ctx: typer.Context,
    model_id: Annotated[
        str,
        typer.Argument(
            help="New model id: lowercase letters, digits and underscores (2-64 chars)."
        ),
    ],
    source: Annotated[
        Path,
        typer.Option("--source", help="Rigged source FBX to bring in.", metavar="FILE"),
    ],
    display_name: Annotated[
        str | None, typer.Option("--display-name", help="Human-readable name.", metavar="TEXT")
    ] = None,
    anim: Annotated[
        list[str] | None,
        typer.Option("--anim", help="Animation FBX glob (repeatable).", metavar="GLOB"),
    ] = None,
    copy: Annotated[bool, typer.Option("--copy", help="Copy the FBX (default).")] = False,
    move: Annotated[bool, typer.Option("--move", help="Move the FBX instead of copying.")] = False,
    link: Annotated[
        bool, typer.Option("--link", help="Symlink the FBX instead of copying.")
    ] = False,
    library_root: LibraryOption = None,
) -> CliOutput:
    """Create models/<model_id>/ with a model.md template and the source FBX."""
    state = state_for(ctx, library_root)
    mode = _mode(copy=copy, move=move, link=link)
    root = resolve_library_root(state)
    paths = new_model(
        root,
        model_id,
        source.expanduser().resolve(),
        display_name=display_name,
        anim_globs=tuple(anim or ()),
        mode=mode,
    )
    render.notice(state.console, f"created {paths.folder}")
    render.message(
        state.console, f"edit {paths.model_md} (## prompt), then: gp validate", style="dim"
    )
    return CliOutput.success(
        COMMAND,
        model_id=model_id,
        folder=str(paths.folder),
        model_md=str(paths.model_md),
        mode=mode,
    )


def register(app: typer.Typer) -> None:
    """Attach to the root app."""
    app.command(COMMAND)(new_model_command)


__all__ = ["COMMAND", "new_model_command", "register"]
