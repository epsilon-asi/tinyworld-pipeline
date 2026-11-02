"""``gp clean [--model ID]... (--work | --output | --state | --all) [--yes]`` (plan 08 §2).

Only ``work/``, ``output/`` and ``.pipeline/`` of the selected models are ever removed; ``source/``,
``references/`` and ``model.md`` are never touched.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Annotated

import typer

from tinyworld_pipeline.cli import render
from tinyworld_pipeline.cli.boundary import command
from tinyworld_pipeline.cli.commands.common import LibraryOption, ModelOption, YesOption, state_for
from tinyworld_pipeline.cli.state import CliOutput
from tinyworld_pipeline.cli.wiring import load_library
from tinyworld_pipeline.domain.errors import UsageError
from tinyworld_pipeline.library.paths import ModelPaths

COMMAND = "clean"
Target = tuple[str, str, Path]
"""``(model_id, kind, path)``."""


def targets_for(paths: ModelPaths, *, work: bool, output: bool, state: bool) -> list[Target]:
    """Existing trees of one model matching the requested kinds."""
    wanted = (
        ("work", work, paths.work),
        ("output", output, paths.output),
        ("state", state, paths.pipeline),
    )
    return [(paths.model_id, kind, path) for kind, flag, path in wanted if flag and path.exists()]


@command(COMMAND)
def clean_command(  # noqa: PLR0917 - typer options are positional by convention
    ctx: typer.Context,
    model: ModelOption = None,
    work: Annotated[bool, typer.Option("--work", help="Remove work/ (intermediates).")] = False,
    output: Annotated[
        bool, typer.Option("--output", help="Remove output/ (deliverables).")
    ] = False,
    state: Annotated[
        bool, typer.Option("--state", help="Remove .pipeline/ (state + logs).")
    ] = False,
    all_: Annotated[
        bool, typer.Option("--all", help="Remove work/, output/ and .pipeline/.")
    ] = False,
    yes: YesOption = False,
    library_root: LibraryOption = None,
) -> CliOutput:
    """Delete intermediates, deliverables and/or state of the selected models."""
    cli = state_for(ctx, library_root)
    if all_:
        work = output = state = True
    if not (work or output or state):
        raise UsageError("choose what to clean", hint="pass --work, --output, --state or --all")
    library = load_library(cli)
    targets: list[Target] = []
    for model_id in library.select(ids=model or None):
        targets.extend(
            targets_for(library.paths.model(model_id), work=work, output=output, state=state)
        )
    if not targets:
        render.message(cli.console, "nothing to clean")
        return CliOutput.success(COMMAND, removed=[], confirmed=True)
    for model_id, kind, path in targets:
        render.message(cli.console, f"  {model_id}: {kind} -> {path}")
    if (
        not yes
        and not cli.json
        and not typer.confirm(f"remove {len(targets)} tree(s)?", default=False)
    ):
        render.notice(cli.console, "aborted; nothing removed")
        return CliOutput.success(COMMAND, removed=[], confirmed=False)
    removed = []
    for model_id, kind, path in targets:
        shutil.rmtree(path)
        removed.append({"model_id": model_id, "kind": kind, "path": str(path)})
    render.notice(cli.console, f"removed {len(removed)} tree(s)")
    return CliOutput.success(COMMAND, removed=removed, confirmed=True)


def register(app: typer.Typer) -> None:
    """Attach to the root app."""
    app.command(COMMAND)(clean_command)


__all__ = ["COMMAND", "clean_command", "register", "targets_for"]
