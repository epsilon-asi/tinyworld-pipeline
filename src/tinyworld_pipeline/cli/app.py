"""``gp`` entry point: global options, logging setup and the command registry.

:func:`build_app` is the test seam (plan 08 §5): it accepts a :class:`ContextFactory` so the run
command uses fakes, a recording console, an explicit environment and client/Blender hooks. The
production entry point is :func:`main`. Exit codes and ``--json`` documents are handled by
:mod:`tinyworld_pipeline.cli.boundary`, which wraps every command.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer

from tinyworld_pipeline import __version__
from tinyworld_pipeline.cli import render
from tinyworld_pipeline.cli.commands import (
    clean,
    doctor,
    init,
    new_model,
    run,
    status,
    tripo,
    unity,
    validate,
)
from tinyworld_pipeline.cli.state import BlenderResolver, CliHooks, CliState, TripoClientFactory
from tinyworld_pipeline.cli.wiring import (
    default_secrets_loader,
    make_tripo_client,
    process_environment,
    resolve_blender,
    secrets_from_mapping,
)
from tinyworld_pipeline.observability.console import Console
from tinyworld_pipeline.observability.logging import configure_logging
from tinyworld_pipeline.pipeline.runner import ContextFactory

APP_NAME = "gp"
APP_HELP = "Upres + retexture rigged game assets through Blender and Tripo; Unity-ready output."
ENV_LOG_LEVEL = "GP_LOGGING__LEVEL"
DEFAULT_LOG_LEVEL = "INFO"


class LogLevel(StrEnum):
    """``--log-level`` choices (plan 08 §1)."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


def _version_callback(value: bool) -> None:
    if value:
        render.emit_version(__version__)
        raise typer.Exit(0)


def build_app(
    *,
    ctx_factory: ContextFactory | None = None,
    console: Console | None = None,
    env: Mapping[str, str] | None = None,
    tripo_client_factory: TripoClientFactory | None = None,
    blender_resolver: BlenderResolver | None = None,
    cwd: Path | None = None,
) -> typer.Typer:
    """Create the ``gp`` application.

    Args:
        ctx_factory: Replaces the production :class:`PipelineContext` assembly in ``gp run``
            (tests inject fakes). ``None`` builds Blender/Tripo adapters for real.
        console: Console to render through; defaults to a stdout console honouring the global
            flags. Tests pass ``Console.recording()``.
        env: Environment mapping for ``GP_*`` lookups and secrets. When given, secrets are read
            from this mapping only (never from ``.env`` files); ``None`` uses the process
            environment and the plan 07 §2 ``.env`` locations.
        tripo_client_factory: Builds the Tripo client (tests pass one bound to a mocked
            transport).
        blender_resolver: Locates Blender and reads its version (tests pass a stub).
        cwd: Start directory for library discovery; defaults to the process cwd.

    Returns:
        A configured :class:`typer.Typer`.
    """
    hooks = CliHooks(
        ctx_factory=ctx_factory,
        tripo_client_factory=tripo_client_factory or make_tripo_client,
        blender_resolver=blender_resolver or resolve_blender,
        secrets_loader=(lambda: secrets_from_mapping(env))
        if env is not None
        else default_secrets_loader,
    )
    environment: Mapping[str, str] = env if env is not None else process_environment()
    app = typer.Typer(
        name=APP_NAME,
        help=APP_HELP,
        add_completion=False,
        no_args_is_help=True,
        pretty_exceptions_enable=False,
        context_settings={"help_option_names": ["-h", "--help"]},
    )

    @app.callback()
    def _root(  # noqa: PLR0917 - typer global options are positional by convention
        ctx: typer.Context,
        library: Annotated[
            Path | None,
            typer.Option(
                "--library", help="Asset library root (default: auto-detect).", metavar="PATH"
            ),
        ] = None,
        json_output: Annotated[
            bool, typer.Option("--json", help="Print one JSON document on stdout.")
        ] = False,
        log_level: Annotated[
            LogLevel | None,
            typer.Option(
                "--log-level", help="Console log level (default: config).", case_sensitive=False
            ),
        ] = None,
        blender: Annotated[
            Path | None,
            typer.Option("--blender", help="Blender executable override.", metavar="PATH"),
        ] = None,
        no_color: Annotated[bool, typer.Option("--no-color", help="Disable colours.")] = False,
        version: Annotated[
            bool,
            typer.Option(
                "--version",
                help="Print the version and exit.",
                callback=_version_callback,
                is_eager=True,
            ),
        ] = False,
    ) -> None:
        """Upres + retexture rigged game assets through Blender and Tripo; Unity-ready output."""
        del version
        level = (log_level.value if log_level else None) or environment.get(ENV_LOG_LEVEL)
        handle = configure_logging(
            level=level or DEFAULT_LOG_LEVEL,
            json_console=json_output,
            no_color=no_color,
            run_log_path=None,
        )
        ctx.obj = CliState(
            library=library,
            json=json_output,
            log_level=log_level.value if log_level else None,
            blender=blender,
            no_color=no_color,
            env=environment,
            cwd=(cwd or Path.cwd()).resolve(),
            console=console or Console(no_color=no_color, json_mode=json_output),
            hooks=hooks,
            logging_handle=handle,
        )

    for module in (init, new_model, validate, run, status, clean, tripo, unity, doctor):
        module.register(app)
    return app


def main() -> None:
    """Console-script entry point (``gp = tinyworld_pipeline.cli.app:main``)."""
    build_app()()


__all__ = ["APP_NAME", "LogLevel", "build_app", "main"]
