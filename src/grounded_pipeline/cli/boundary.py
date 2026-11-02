"""The one exception boundary of the CLI (plan 08 §2 exit codes, 08 §4 JSON, 14 §2 errors).

Every command is wrapped by :func:`command`; the body returns a :class:`CliOutput` or raises a
:class:`PipelineError`. The wrapper prints the error (unless ``--json``), emits the single JSON
document when requested, closes the per-invocation resources and exits with the mapped code:
``PipelineError.exit_code``, ``130`` for Ctrl-C, ``1`` (``E_UNEXPECTED``) for anything else.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import ParamSpec

import structlog
import typer

from grounded_pipeline.cli import render
from grounded_pipeline.cli.state import CliOutput, CliState, UnexpectedError
from grounded_pipeline.domain.errors import CancelledError, PipelineError

log = structlog.get_logger(__name__)

P = ParamSpec("P")
Body = Callable[P, CliOutput]


def _find_state(args: tuple[object, ...], kwargs: dict[str, object]) -> CliState:
    for value in (*args, *kwargs.values()):
        finder = getattr(value, "find_object", None)  # typer vendors click: no shared class
        if finder is None:
            continue
        state = finder(CliState)
        if isinstance(state, CliState):
            return state
    msg = "command functions must take `ctx: typer.Context` and run under build_app()"
    raise RuntimeError(msg)


def run_body(name: str, state: CliState, body: Callable[[], CliOutput]) -> CliOutput:
    """Execute ``body`` under the boundary and return the document to emit (never raises)."""
    state.command = name
    try:
        output = body()
    except PipelineError as exc:
        output = CliOutput.failure(name, exc)
        state.console.print_error(exc)
    except KeyboardInterrupt:
        cancelled = CancelledError("interrupted by user", hint="rerun to resume where it stopped")
        output = CliOutput.failure(name, cancelled)
        state.console.print_error(cancelled)
    except Exception as unexpected:  # the CLI top level is the one allowed catch-all (plan 19 §3)
        log.debug("cli.unexpected", exc_info=True)
        wrapped = UnexpectedError(
            f"{type(unexpected).__name__}: {unexpected}",
            hint="rerun with --log-level DEBUG for the traceback",
            cause=unexpected,
        )
        output = CliOutput.failure(name, wrapped)
        state.console.print_error(wrapped)
    else:
        if output.error is not None:
            render.print_error_document(state.console, output.error)
    return output


def finish(state: CliState, output: CliOutput) -> None:
    """Emit the JSON document (``--json``), release resources and exit with the code."""
    try:
        if state.json:
            render.emit_json(output)
    finally:
        state.close()
    raise typer.Exit(output.exit_code)


def command(name: str) -> Callable[[Body[P]], Callable[P, None]]:
    """Decorate a typer command body (``ctx: typer.Context, …) -> CliOutput``).

    Args:
        name: Command name used in the JSON document (``"run"``, ``"tripo balance"``).
    """

    def decorate(body: Body[P]) -> Callable[P, None]:
        @functools.wraps(body)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> None:
            state = _find_state(args, kwargs)
            output = run_body(name, state, lambda: body(*args, **kwargs))
            finish(state, output)

        return wrapper

    return decorate


__all__ = ["command", "finish", "run_body"]
