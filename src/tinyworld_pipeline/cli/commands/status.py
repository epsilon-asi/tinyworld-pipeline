"""``gp status [--model ID]...`` (plan 08 §2): reads ``state.json`` only."""

from __future__ import annotations

from typing import Any

import typer

from tinyworld_pipeline import __version__
from tinyworld_pipeline.cli import render
from tinyworld_pipeline.cli.boundary import command
from tinyworld_pipeline.cli.commands.common import LibraryOption, ModelOption, state_for
from tinyworld_pipeline.cli.state import CliOutput
from tinyworld_pipeline.cli.wiring import load_library
from tinyworld_pipeline.domain.state import ModelState
from tinyworld_pipeline.pipeline.state_store import JsonStateStore

COMMAND = "status"


def state_json(state: ModelState) -> dict[str, Any]:
    """Compact per-model status for the JSON document."""
    return {
        "model_id": state.model_id,
        "stages": {name: rec.status.value for name, rec in state.stages.items()},
        "credits_total": state.credits_total,
        "last_run_id": state.last_run_id,
        "last_error": render.last_error(state) if render.last_error(state) != "-" else None,
    }


@command(COMMAND)
def status_command(
    ctx: typer.Context, model: ModelOption = None, library_root: LibraryOption = None
) -> CliOutput:
    """Show every stage's status per model, lifetime credits, last run and last error."""
    state = state_for(ctx, library_root)
    library = load_library(state)
    store = JsonStateStore(library.paths, pipeline_version=__version__)
    states = [store.load(model_id) for model_id in library.select(ids=model or None)]
    render.status_table(state.console, states)
    return CliOutput.success(COMMAND, models=[state_json(s) for s in states])


def register(app: typer.Typer) -> None:
    """Attach to the root app."""
    app.command(COMMAND)(status_command)


__all__ = ["COMMAND", "register", "state_json", "status_command"]
