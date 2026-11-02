"""Per-invocation CLI state, test seams and the ``--json`` output contract (plan 08 §1, §4).

:class:`CliState` is built once by the root callback and stored on ``ctx.obj``; every command reads
the global options, the console and the injected hooks from it. :class:`CliOutput` is the single
JSON document a command returns; the exception boundary in :mod:`tinyworld_pipeline.cli.app`
serialises it (``--json``) and turns ``exit_code`` into the process exit status.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

import typer
from pydantic import BaseModel, ConfigDict, SecretStr

from tinyworld_pipeline.domain.errors import PipelineError

if TYPE_CHECKING:
    from tinyworld_pipeline.blender.locate import BlenderInfo
    from tinyworld_pipeline.config.settings import Secrets
    from tinyworld_pipeline.domain.library_config import BlenderSection, TripoSection
    from tinyworld_pipeline.domain.ports import CancelToken
    from tinyworld_pipeline.observability.console import Console
    from tinyworld_pipeline.observability.logging import LoggingHandle
    from tinyworld_pipeline.pipeline.runner import ContextFactory
    from tinyworld_pipeline.tripo.client import TripoClient

TripoClientFactory = Callable[[SecretStr, "TripoSection"], "TripoClient"]
"""Builds the Tripo client for ``gp tripo …`` / ``gp doctor`` (tests inject a mocked one)."""

BlenderResolver = Callable[[Path | None, Mapping[str, str], "BlenderSection"], "BlenderInfo"]
"""``(--blender, env, [blender] section) -> BlenderInfo``; raises ``BlenderNotFoundError`` /
``BlenderVersionError``. Tests inject a stub so no subprocess is spawned."""

SecretsLoader = Callable[[], "Secrets"]


class UnexpectedError(PipelineError):
    """Wrapper for anything that is not a :class:`PipelineError` (exit 1, ``E_UNEXPECTED``)."""

    code = "E_UNEXPECTED"
    exit_code = 1


class ErrorDocument(BaseModel):
    """``error`` member of a failed :class:`CliOutput` (plan 08 §4)."""

    model_config = ConfigDict(frozen=True)

    code: str
    message: str
    hint: str | None = None
    detail: dict[str, Any] | None = None

    @classmethod
    def from_error(cls, exc: PipelineError) -> ErrorDocument:
        """Build from a :class:`PipelineError` (uses ``to_dict`` so job codes are honoured)."""
        data = exc.to_dict()
        return cls(
            code=str(data["code"]),
            message=str(data["message"]),
            hint=data.get("hint"),
            detail=data.get("detail"),
        )


class CliOutput(BaseModel):
    """The one JSON document a command prints with ``--json``.

    Command-specific fields are allowed as extras (``models``, ``run_id``, ``checks`` …); they
    must already be JSON-friendly values.
    """

    model_config = ConfigDict(extra="allow", frozen=True)

    command: str
    ok: bool
    exit_code: int
    error: ErrorDocument | None = None

    @classmethod
    def success(cls, command: str, **fields: Any) -> Self:  # noqa: ANN401 - JSON extras
        """A successful (exit 0) document with extra fields."""
        return cls(command=command, ok=True, exit_code=0, **fields)

    @classmethod
    def failure(
        cls, command: str, error: PipelineError | ErrorDocument, *, exit_code: int | None = None
    ) -> Self:
        """A failed document; the exit code defaults to the error's."""
        doc = error if isinstance(error, ErrorDocument) else ErrorDocument.from_error(error)
        code = exit_code if exit_code is not None else _exit_code_of(error)
        return cls(command=command, ok=False, exit_code=code, error=doc)

    def to_json(self) -> str:
        """Canonical rendering (2-space indent, sorted keys, UTF-8, trailing newline)."""
        data = self.model_dump(mode="json", exclude_none=False)
        return json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _exit_code_of(error: PipelineError | ErrorDocument) -> int:
    return error.exit_code if isinstance(error, PipelineError) else 1


@dataclass(frozen=True)
class CliHooks:
    """Injection points for tests; production values come from :mod:`cli.wiring`."""

    ctx_factory: ContextFactory | None
    tripo_client_factory: TripoClientFactory
    blender_resolver: BlenderResolver
    secrets_loader: SecretsLoader


@dataclass
class CliState:
    """Global options plus the shared console/logging/hook objects (``ctx.obj``)."""

    library: Path | None
    json: bool
    log_level: str | None
    blender: Path | None
    no_color: bool
    env: Mapping[str, str]
    cwd: Path
    console: Console
    hooks: CliHooks
    logging_handle: LoggingHandle | None = None
    exit_stack: ExitStack = field(default_factory=ExitStack)
    active_cancel: CancelToken | None = None
    command: str | None = None
    _secrets: Secrets | None = field(default=None, repr=False)

    def secrets(self) -> Secrets:
        """Load secrets once; register the Tripo key with the log redactor."""
        if self._secrets is None:
            self._secrets = self.hooks.secrets_loader()
            key = self._secrets.tripo_api_key
            if self.logging_handle is not None and key is not None:
                self.logging_handle.add_secret(key.get_secret_value())
        return self._secrets

    def close(self) -> None:
        """Release everything the invocation opened (log tees, then the logging handle)."""
        self.exit_stack.close()
        if self.logging_handle is not None:
            self.logging_handle.close()
            self.logging_handle = None


def get_state(ctx: typer.Context) -> CliState:
    """The :class:`CliState` the root callback stored on the click context.

    Raises:
        RuntimeError: The callback did not run (programming error).
    """
    state = ctx.find_object(CliState)
    if state is None:
        msg = "CliState missing from the click context; build the app with build_app()"
        raise RuntimeError(msg)
    return state


__all__ = [
    "BlenderResolver",
    "CliHooks",
    "CliOutput",
    "CliState",
    "ErrorDocument",
    "SecretsLoader",
    "TripoClientFactory",
    "UnexpectedError",
    "get_state",
]
