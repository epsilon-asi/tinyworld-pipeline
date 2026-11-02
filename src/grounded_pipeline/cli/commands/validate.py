"""``gp validate [--model ID]...`` (plan 08 §2): exit 0 clean/warnings, 3 on any error."""

from __future__ import annotations

from typing import Any

import typer

from grounded_pipeline.cli import render
from grounded_pipeline.cli.boundary import command
from grounded_pipeline.cli.commands.common import LibraryOption, ModelOption, state_for
from grounded_pipeline.cli.state import CliOutput, ErrorDocument
from grounded_pipeline.cli.wiring import blender_check, cli_overrides_from, load_library
from grounded_pipeline.domain.errors import Problem, ValidationError
from grounded_pipeline.library.validation import validate_library

COMMAND = "validate"


def problems_json(problems: list[Problem]) -> list[dict[str, Any]]:
    """JSON-friendly problem list."""
    return [p.model_dump(mode="json") for p in problems]


def models_json(problems: list[Problem], model_ids: list[str]) -> list[dict[str, Any]]:
    """Per-model status rows for the JSON document."""
    groups = render.group_problems(problems, model_ids)
    return [
        {
            "model_id": model_id or None,
            "status": render.validation_status(items),
            "problems": problems_json(items),
        }
        for model_id, items in groups.items()
        if model_id or items
    ]


@command(COMMAND)
def validate_command(
    ctx: typer.Context, model: ModelOption = None, library_root: LibraryOption = None
) -> CliOutput:
    """Check library.toml, style.md and every (or the selected) model.md; verify Blender."""
    state = state_for(ctx, library_root)
    library = load_library(state)
    model_ids = library.select(ids=model or None)
    problems = validate_library(
        library,
        model or None,
        env=state.env,
        cli=cli_overrides_from(state),
        blender_check=blender_check(state),
    )
    render.validate_table(state.console, problems, model_ids)
    errors = [p for p in problems if p.severity == "error"]
    fields = {"models": models_json(problems, model_ids), "problems": problems_json(problems)}
    if errors:
        exc = ValidationError(problems)
        return CliOutput.failure(
            COMMAND, ErrorDocument.from_error(exc), exit_code=exc.exit_code
        ).model_copy(update=fields)
    render.message(state.console, "library is valid", style="bold green")
    return CliOutput.success(COMMAND, **fields)


def register(app: typer.Typer) -> None:
    """Attach to the root app."""
    app.command(COMMAND)(validate_command)


__all__ = ["COMMAND", "models_json", "problems_json", "register", "validate_command"]
