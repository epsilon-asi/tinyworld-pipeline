"""``gp tripo balance | task <id> | usage`` (plan 08 §2). Read-only account calls."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Annotated, Any

import typer

from grounded_pipeline.cli import render
from grounded_pipeline.cli.boundary import command
from grounded_pipeline.cli.commands.common import LibraryOption, state_for
from grounded_pipeline.cli.state import CliOutput, CliState
from grounded_pipeline.cli.wiring import load_library
from grounded_pipeline.domain.errors import UsageError
from grounded_pipeline.domain.library_config import TripoSection
from grounded_pipeline.tripo.client import TripoClient
from grounded_pipeline.tripo.models import TaskDetail

tripo_app = typer.Typer(
    name="tripo", help="Tripo account queries (no credits spent).", no_args_is_help=True
)

NO_KEY_HINT = "export TRIPO_API_KEY or add it to .env / ~/.config/grounded-pipeline/.env"


def tripo_section(state: CliState) -> TripoSection:
    """``[tripo]`` of the current library when one is found, else defaults."""
    try:
        return load_library(state).config.tripo
    except UsageError:
        return TripoSection()


@contextmanager
def tripo_client(state: CliState) -> Iterator[TripoClient]:
    """A client for the configured key; closed on exit.

    Raises:
        UsageError: ``TRIPO_API_KEY`` is not set (exit 2).
    """
    secrets = state.secrets()
    if not secrets.has_tripo_key or secrets.tripo_api_key is None:
        raise UsageError("TRIPO_API_KEY is not set", hint=NO_KEY_HINT)
    client = state.hooks.tripo_client_factory(secrets.tripo_api_key, tripo_section(state))
    try:
        yield client
    finally:
        client.close()


def task_json(detail: TaskDetail) -> dict[str, Any]:
    """JSON-friendly task detail (URLs as strings)."""
    return detail.model_dump(mode="json")


@command("tripo balance")
def balance_command(ctx: typer.Context, library_root: LibraryOption = None) -> CliOutput:
    """GET /account/balance: available and frozen credits."""
    state = state_for(ctx, library_root)
    with tripo_client(state) as client:
        data = client.balance()
    render.message(state.console, f"balance: {data.balance:g}   frozen: {data.frozen:g}")
    return CliOutput.success("tripo balance", balance=data.balance, frozen=data.frozen)


@command("tripo task")
def task_command(
    ctx: typer.Context,
    task_id: Annotated[str, typer.Argument(help="Tripo task id.")],
    library_root: LibraryOption = None,
) -> CliOutput:
    """GET /tasks/{id}: status, progress, output URLs and credits."""
    state = state_for(ctx, library_root)
    with tripo_client(state) as client:
        detail = client.get_task(task_id)
    output = detail.output
    rows = [
        ("task_id", detail.task_id),
        ("type", detail.type),
        ("status", str(detail.status)),
        ("progress", f"{detail.progress}%"),
        ("credits", f"{detail.credits_consumed:g}" if detail.credits_consumed is not None else "-"),
        ("model_url", str(output.model_url) if output and output.model_url else "-"),
        (
            "preview_url",
            str(output.rendered_image_url) if output and output.rendered_image_url else "-",
        ),
        ("error", f"{detail.error_code}: {detail.error_message}" if detail.error_code else "-"),
    ]
    state.console.table(f"task {task_id}", ("field", "value"), rows)
    return CliOutput.success("tripo task", task=task_json(detail))


@command("tripo usage")
def usage_command(ctx: typer.Context, library_root: LibraryOption = None) -> CliOutput:
    """GET /account/usage: recent tasks and the credits they consumed."""
    state = state_for(ctx, library_root)
    with tripo_client(state) as client:
        entries = client.usage()
    state.console.table(
        "usage",
        ("task_id", "type", "credits", "created_at"),
        [(e.task_id, e.type, f"{e.credits_consumed:g}", e.created_at or "-") for e in entries],
    )
    total = round(sum(e.credits_consumed for e in entries), 6)
    render.message(state.console, f"{len(entries)} task(s), {total:g} credits")
    return CliOutput.success(
        "tripo usage", entries=[e.model_dump(mode="json") for e in entries], credits_total=total
    )


def register(app: typer.Typer) -> None:
    """Attach ``gp tripo …`` to the root app."""
    tripo_app.command("balance")(balance_command)
    tripo_app.command("task")(task_command)
    tripo_app.command("usage")(usage_command)
    app.add_typer(tripo_app, name="tripo")


__all__ = [
    "NO_KEY_HINT",
    "balance_command",
    "register",
    "task_command",
    "task_json",
    "tripo_app",
    "tripo_client",
    "tripo_section",
    "usage_command",
]
