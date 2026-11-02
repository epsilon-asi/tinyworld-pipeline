"""structlog configuration: processor chain, secret redaction, dual sinks, per-stage tee (plan 14).

One shared processor chain (``merge_contextvars`` → ``add_log_level`` → ISO UTC timestamp →
stack/exception rendering → secret redaction) feeds two stdlib handlers through
:class:`structlog.stdlib.ProcessorFormatter`:

* **console** — ``ConsoleRenderer`` (colours when allowed) or ``JSONRenderer`` with
  ``json_console``; filtered by ``level``; written to stderr so ``--json`` stdout stays clean.
* **run file** — JSON lines at ``run_log_path``; always DEBUG.

:meth:`LoggingHandle.tee` adds a temporary third JSON-lines handler for one stage
(``<model>/.pipeline/logs/<stage>-<run_id>.log``). Context keys (``run_id``, ``model_id``,
``stage``, ``job_id``, ``task_id``, ``attempt``) are bound through :mod:`structlog.contextvars`, so
every line logged inside :func:`bind_run` / :func:`bind_model` / :func:`bind_stage` carries them
without repeating them by hand.

Redaction runs *after* ``format_exc_info`` (plan 14 §4 lists it before) so the rendered traceback
text is scrubbed as well; the processor order is otherwise the plan's.

Example::

    handle = configure_logging(level="INFO", json_console=False, no_color=False,
                               run_log_path=runs_dir / run_id / "run.log",
                               secrets=[secrets.tripo_api_key.get_secret_value()])
    with bind_run(run_id), bind_model("hero"), bind_stage("prepare"):
        with handle.tee(model_dir / ".pipeline" / "logs" / f"prepare-{run_id}.log"):
            log.info(events.STAGE_START)
    handle.close()

Worker threads do not inherit context variables automatically. Run thread bodies via
``contextvars.copy_context().run(...)`` or re-bind inside the thread.
"""

from __future__ import annotations

import logging
import re
import sys
from collections.abc import Iterable, Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, TextIO

import structlog
from structlog.stdlib import ProcessorFormatter

if TYPE_CHECKING:
    from structlog.typing import EventDict, Processor, WrappedLogger

__all__ = [
    "MASK",
    "THIRD_PARTY_LOGGERS",
    "LoggingHandle",
    "SecretRedactor",
    "bind_context",
    "bind_model",
    "bind_run",
    "bind_stage",
    "configure_logging",
    "get_logging_handle",
    "mask_secret",
]

MASK: Final[str] = "****"
# Shorter values are masked entirely: a four-char suffix would leak most of them.
_MIN_SECRET_LEN: Final[int] = 8
_SUFFIX_VISIBLE: Final[int] = 4
THIRD_PARTY_LOGGERS: Final[tuple[str, ...]] = ("httpx", "httpcore", "PIL", "filelock")
_SENSITIVE_KEYS: Final[frozenset[str]] = frozenset({"authorization", "x-api-key", "api_key"})
_BEARER_RE: Final[re.Pattern[str]] = re.compile(r"(?i)(bearer\s+)([^\s,;'\"<>]+)")


def mask_secret(secret: str) -> str:
    """Return ``****`` + the last four characters (or just ``****`` for short values)."""
    if len(secret) < _MIN_SECRET_LEN:
        return MASK
    return MASK + secret[-_SUFFIX_VISIBLE:]


class SecretRedactor:
    """structlog processor that scrubs registered secrets and bearer tokens from event dicts.

    Every string value is scanned — recursively through dicts, lists, tuples and sets — so secrets
    hiding in nested ``detail`` payloads, headers and formatted exception text are masked too.
    Values under sensitive keys (``authorization``, ``x-api-key``, ``api_key``) are masked
    wholesale.
    Non-string scalars are left untouched.
    """

    def __init__(self, secrets: Iterable[str] = ()) -> None:
        self._secrets: list[str] = []
        for value in secrets:
            self.add(value)

    @property
    def secrets(self) -> tuple[str, ...]:
        """Registered secrets, longest first (so overlapping keys mask correctly)."""
        return tuple(self._secrets)

    def add(self, value: str) -> None:
        """Register ``value`` for redaction. Empty strings and duplicates are ignored."""
        if not value or value in self._secrets:
            return
        self._secrets.append(value)
        self._secrets.sort(key=len, reverse=True)

    def scrub_text(self, text: str) -> str:
        """Mask every registered secret and any ``Bearer <token>`` sequence in ``text``."""
        for secret in self._secrets:
            if secret in text:
                text = text.replace(secret, mask_secret(secret))
        return _BEARER_RE.sub(lambda m: m.group(1) + _mask_token(m.group(2)), text)

    def scrub(self, value: Any) -> Any:  # noqa: ANN401 - event values are arbitrary
        """Return ``value`` with secrets masked, recursing into containers."""
        if isinstance(value, str):
            return self.scrub_text(value)
        if isinstance(value, Mapping):
            return {k: self._scrub_item(k, v) for k, v in value.items()}
        if isinstance(value, list):
            return [self.scrub(v) for v in value]
        if isinstance(value, tuple):
            return tuple(self.scrub(v) for v in value)
        if isinstance(value, set | frozenset):
            return {self.scrub(v) for v in value}
        return value

    def _scrub_item(self, key: object, value: Any) -> Any:  # noqa: ANN401
        if isinstance(key, str) and key.lower() in _SENSITIVE_KEYS and isinstance(value, str):
            return _mask_token(value)
        return self.scrub(value)

    def __call__(
        self, _logger: WrappedLogger, _method_name: str, event_dict: EventDict
    ) -> EventDict:
        """Processor entry point."""
        for key, value in list(event_dict.items()):
            event_dict[key] = self._scrub_item(key, value)
        return event_dict


def _mask_token(token: str) -> str:
    if token.startswith(MASK):  # already masked (e.g. registered secret inside a Bearer header)
        return token
    return mask_secret(token)


# --------------------------------------------------------------------------- context binding
def bind_context(**values: object) -> AbstractContextManager[None]:
    """Bind arbitrary keys (``job_id``, ``task_id``, ``attempt`` …) for the duration of the block.

    Returns:
        A context manager; the keys are removed (or restored) again on exit.
    """
    return structlog.contextvars.bound_contextvars(**values)


def bind_run(run_id: str) -> AbstractContextManager[None]:
    """Bind ``run_id`` for the duration of the block."""
    return bind_context(run_id=run_id)


def bind_model(model_id: str) -> AbstractContextManager[None]:
    """Bind ``model_id`` for the duration of the block."""
    return bind_context(model_id=model_id)


def bind_stage(stage: str) -> AbstractContextManager[None]:
    """Bind ``stage`` for the duration of the block."""
    return bind_context(stage=stage)


# --------------------------------------------------------------------------- handle
class _MatchFilter(logging.Filter):
    """Pass only structlog records whose event dict contains every ``key=value`` in ``match``."""

    def __init__(self, match: Mapping[str, object]) -> None:
        super().__init__()
        self._match = dict(match)

    def filter(self, record: logging.LogRecord) -> bool:
        event = record.msg
        if not isinstance(event, dict):
            return False
        return all(event.get(k) == v for k, v in self._match.items())


class LoggingHandle:
    """Owner of the handlers installed by :func:`configure_logging`.

    Attributes:
        level: Console level name as configured (upper-case).
        run_log_path: Path of the JSON-lines run log, or ``None`` when no file sink was requested.
    """

    def __init__(
        self,
        *,
        level: str,
        run_log_path: Path | None,
        redactor: SecretRedactor,
        console_handler: logging.Handler,
        file_handler: logging.Handler | None,
        json_formatter: ProcessorFormatter,
        previous_root_level: int,
    ) -> None:
        self.level = level
        self.run_log_path = run_log_path
        self._redactor = redactor
        self._console_handler = console_handler
        self._file_handler = file_handler
        self._json_formatter = json_formatter
        self._previous_root_level = previous_root_level
        self._tees: list[logging.Handler] = []
        self._closed = False

    @property
    def handlers(self) -> tuple[logging.Handler, ...]:
        """Handlers currently installed on the root logger by this handle (console, file, tees)."""
        root = logging.getLogger()
        mine = [self._console_handler, self._file_handler, *self._tees]
        return tuple(h for h in mine if h is not None and h in root.handlers)

    @property
    def secrets(self) -> tuple[str, ...]:
        """Secrets currently redacted (for diagnostics/tests; never log this)."""
        return self._redactor.secrets

    def add_secret(self, value: str) -> None:
        """Register another value to redact from every subsequent log line."""
        self._redactor.add(value)

    @contextmanager
    def tee(self, path: Path, *, match: Mapping[str, object] | None = None) -> Iterator[Path]:
        """Mirror every DEBUG+ line to ``path`` (JSON lines) while the block runs.

        Args:
            path: Target file, created (with parents) on entry and appended to.
            match: Optional ``key=value`` pairs a line's event dict must carry to be written, e.g.
                ``{"model_id": "hero"}`` so parallel models do not cross-contaminate stage logs.

        Yields:
            ``path``.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(path, encoding="utf-8")
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(self._json_formatter)
        if match:
            handler.addFilter(_MatchFilter(match))
        root = logging.getLogger()
        root.addHandler(handler)
        self._tees.append(handler)
        try:
            yield path
        finally:
            root.removeHandler(handler)
            handler.close()
            if handler in self._tees:  # close() may already have dropped it
                self._tees.remove(handler)

    def close(self) -> None:
        """Remove and close every handler this handle installed. Safe to call twice."""
        global _ACTIVE  # noqa: PLW0603 - single process-wide logging configuration
        if self._closed:
            return
        self._closed = True
        root = logging.getLogger()
        for handler in [self._console_handler, self._file_handler, *self._tees]:
            if handler is None:
                continue
            root.removeHandler(handler)
            handler.close()
        self._tees.clear()
        root.setLevel(self._previous_root_level)
        if _ACTIVE is self:
            _ACTIVE = None


_ACTIVE: LoggingHandle | None = None


def get_logging_handle() -> LoggingHandle | None:
    """Return the handle from the most recent :func:`configure_logging`, if still open."""
    return _ACTIVE


# --------------------------------------------------------------------------- configuration
def _shared_processors() -> list[Processor]:
    return [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]


def _parse_level(level: str) -> int:
    mapping = logging.getLevelNamesMapping()
    try:
        return mapping[level.upper()]
    except KeyError:
        valid = ", ".join(sorted(k for k in mapping if k != "NOTSET"))
        msg = f"unknown log level {level!r}; expected one of {valid}"
        raise ValueError(msg) from None


def _stream_is_tty(stream: TextIO) -> bool:
    isatty = getattr(stream, "isatty", None)
    try:
        return bool(isatty()) if callable(isatty) else False
    except (ValueError, OSError):
        return False


def configure_logging(
    *,
    level: str,
    json_console: bool,
    no_color: bool,
    run_log_path: Path | None,
    secrets: Iterable[str] = (),
    console_stream: TextIO | None = None,
) -> LoggingHandle:
    """Install the structlog pipeline and both sinks; replaces any previous configuration.

    Args:
        level: Console threshold (``"DEBUG"``, ``"INFO"`` …, case-insensitive). The run file and
            any tee always receive DEBUG.
        json_console: Render console lines as JSON instead of the human ``ConsoleRenderer``.
        no_color: Disable ANSI colours. Colours are also disabled when ``json_console`` is set or
            the console stream is not a TTY.
        run_log_path: JSON-lines run log (parents created). ``None`` installs no file sink
            (``gp doctor``, ``gp validate``).
        secrets: Clear-text values to redact from every line (the Tripo API key).
        console_stream: Where console lines go; defaults to ``sys.stderr`` so ``--json`` stdout
            stays a single document.

    Returns:
        The :class:`LoggingHandle` owning the installed handlers.

    Raises:
        ValueError: ``level`` is not a known logging level name.
    """
    console_level = _parse_level(level)
    previous = get_logging_handle()
    if previous is not None:
        previous.close()

    redactor = SecretRedactor(secrets)
    shared: list[Processor] = [*_shared_processors(), redactor]

    structlog.configure(
        processors=[*shared, ProcessorFormatter.wrap_for_formatter],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=False,
    )

    json_formatter = ProcessorFormatter(
        processors=[ProcessorFormatter.remove_processors_meta, structlog.processors.JSONRenderer()],
        foreign_pre_chain=shared,
    )

    stream = console_stream if console_stream is not None else sys.stderr
    console_renderer: Processor
    if json_console:
        console_renderer = structlog.processors.JSONRenderer()
    else:
        colors = not no_color and _stream_is_tty(stream)
        # The traceback is already rendered (and redacted) text by the time it reaches the
        # renderer; plain_traceback prints it as-is instead of warning about pretty exceptions.
        console_renderer = structlog.dev.ConsoleRenderer(
            colors=colors, exception_formatter=structlog.dev.plain_traceback
        )
    console_formatter = ProcessorFormatter(
        processors=[ProcessorFormatter.remove_processors_meta, console_renderer],
        foreign_pre_chain=shared,
    )

    console_handler = logging.StreamHandler(stream)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(console_formatter)

    file_handler: logging.FileHandler | None = None
    if run_log_path is not None:
        run_log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(run_log_path, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(json_formatter)

    root = logging.getLogger()
    previous_root_level = root.level
    root.setLevel(logging.DEBUG)
    root.addHandler(console_handler)
    if file_handler is not None:
        root.addHandler(file_handler)

    for name in THIRD_PARTY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

    handle = LoggingHandle(
        level=level.upper(),
        run_log_path=run_log_path,
        redactor=redactor,
        console_handler=console_handler,
        file_handler=file_handler,
        json_formatter=json_formatter,
        previous_root_level=previous_root_level,
    )
    global _ACTIVE  # noqa: PLW0603 - single process-wide logging configuration
    _ACTIVE = handle
    return handle
