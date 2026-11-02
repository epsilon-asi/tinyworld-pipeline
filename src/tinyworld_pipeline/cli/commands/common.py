"""Option types and helpers shared by several commands."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from tinyworld_pipeline.cli.state import CliState, get_state

ModelOption = Annotated[
    list[str] | None,
    typer.Option("--model", "-m", help="Model id (repeatable).", metavar="ID", show_default=False),
]
YesOption = Annotated[bool, typer.Option("--yes", "-y", help="Do not ask for confirmation.")]
LibraryOption = Annotated[
    Path | None,
    typer.Option(
        "--library",
        help="Asset library root (same as the global option).",
        metavar="PATH",
        show_default=False,
    ),
]


def state_for(ctx: typer.Context, library: Path | None = None) -> CliState:
    """The invocation state, with a per-command ``--library`` applied when given."""
    state = get_state(ctx)
    if library is not None:
        state.library = library
    return state


__all__ = ["LibraryOption", "ModelOption", "YesOption", "state_for"]
