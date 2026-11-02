"""Human-facing terminal output (plan 14 §5).

This module and ``cli/render.py`` are the only places allowed to write to the terminal; everything
else logs through structlog. :class:`Console` wraps :class:`rich.console.Console` and knows the
three output modes of the CLI:

* normal — rich output, colours when the terminal supports them;
* ``quiet`` — progress, tables and warnings are suppressed, problems and errors still print;
* ``json_mode`` — nothing human-readable is written so stdout stays a single JSON document.

Use :meth:`Console.recording` in tests to capture plain text.
"""

from __future__ import annotations

import io
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import IO, Self

from rich.console import Console as RichConsole
from rich.table import Table
from rich.text import Text

from grounded_pipeline.domain.errors import PipelineError, Problem

__all__ = ["Console", "RecordingConsole"]

_LIBRARY_GROUP = "(library)"
_RECORDING_WIDTH = 120


class Console:
    """Thin wrapper around rich that honours ``--quiet`` / ``--no-color`` / ``--json``."""

    def __init__(
        self,
        *,
        no_color: bool = False,
        quiet: bool = False,
        json_mode: bool = False,
        file: IO[str] | None = None,
        stderr: bool = False,
        force_terminal: bool | None = None,
        width: int | None = None,
    ) -> None:
        """Create a console.

        Args:
            no_color: Never emit ANSI colour codes.
            quiet: Suppress informational output (``print``, ``rule``, ``table``, warnings).
            json_mode: Suppress all human-readable output.
            file: Target stream; defaults to stdout (or stderr with ``stderr=True``).
            stderr: Write to stderr instead of stdout when ``file`` is not given.
            force_terminal: Override rich's TTY detection (tests pass ``False``).
            width: Fixed width in columns (tests pass one for deterministic wrapping).
        """
        self.no_color = no_color
        self.quiet = quiet
        self.json_mode = json_mode
        self._rich = RichConsole(
            file=file,
            stderr=stderr,
            no_color=no_color,
            force_terminal=force_terminal,
            width=width,
            highlight=False,
            soft_wrap=True,
            emoji=False,
        )

    @classmethod
    def recording(cls, *, quiet: bool = False, json_mode: bool = False) -> RecordingConsole:
        """A console writing plain text into memory; read it back through ``.text``."""
        return RecordingConsole(quiet=quiet, json_mode=json_mode)

    # ------------------------------------------------------------------ mode predicates
    @property
    def silent(self) -> bool:
        """``True`` when nothing at all may be written (``json_mode``)."""
        return self.json_mode

    @property
    def verbose_allowed(self) -> bool:
        """``True`` when informational output may be written (not quiet, not JSON)."""
        return not (self.json_mode or self.quiet)

    @property
    def rich(self) -> RichConsole:
        """The underlying rich console (for ``cli/render.py`` progress bars)."""
        return self._rich

    # ------------------------------------------------------------------ primitives
    def print(self, *objects: object, style: str | None = None, important: bool = False) -> None:
        """Write ``objects`` on one line. Suppressed by ``quiet`` unless ``important``."""
        if self.silent or (self.quiet and not important):
            return
        self._rich.print(*objects, style=style)

    def rule(self, title: str = "") -> None:
        """Horizontal rule with an optional title (informational)."""
        if not self.verbose_allowed:
            return
        self._rich.rule(title)

    def table(
        self,
        title: str | None,
        columns: Sequence[str],
        rows: Iterable[Sequence[object]],
    ) -> None:
        """Render a simple table (informational).

        Args:
            title: Optional caption above the table.
            columns: Header labels.
            rows: Row cells; every cell is rendered with ``str()``.
        """
        if not self.verbose_allowed:
            return
        table = Table(title=title, show_lines=False)
        for column in columns:
            table.add_column(column)
        for row in rows:
            table.add_row(*(str(cell) for cell in row))
        self._rich.print(table)

    # ------------------------------------------------------------------ plan 14 §5 helpers
    def print_problems(self, problems: Sequence[Problem]) -> None:
        """Render validation problems grouped by model, each with its hint on the next line.

        Errors and warnings are both shown (prefixed ``✗`` / ``!``); library-wide problems (no
        ``model_id``) are grouped under ``(library)``. Always printed unless ``json_mode``.
        """
        if self.silent or not problems:
            return
        groups: dict[str, list[Problem]] = {}
        for problem in problems:
            groups.setdefault(problem.model_id or _LIBRARY_GROUP, []).append(problem)
        for model_id, items in groups.items():
            self._rich.print(Text(model_id, style="bold"))
            for problem in items:
                self._rich.print(_problem_line(problem))
                if problem.hint:
                    self._rich.print(Text(f"      hint: {problem.hint}", style="dim"))

    def print_warnings(self, warnings: dict[str, list[str]]) -> None:
        """Print the end-of-run “Warnings” block grouped by model (informational)."""
        if not self.verbose_allowed:
            return
        if not any(warnings.values()):
            return
        self._rich.print(Text("Warnings", style="bold yellow"))
        for model_id, messages in warnings.items():
            if not messages:
                continue
            self._rich.print(Text(f"  {model_id}", style="bold"))
            for message in messages:
                self._rich.print(Text(f"    ! {message}", style="yellow"))

    def print_error(self, err: PipelineError, *, log_path: Path | None = None) -> None:
        """Print a failure: code, message, hint (if any) and the stage log path (if any).

        Always printed unless ``json_mode`` — a quiet run still tells you why it failed.
        """
        if self.silent:
            return
        line = Text("✗ ", style="bold red")
        line.append(err.to_dict()["code"], style="bold red")
        line.append(": ")
        line.append(err.message)
        self._rich.print(line)
        if err.hint:
            self._rich.print(Text(f"  hint: {err.hint}", style="dim"))
        if log_path is not None:
            self._rich.print(Text(f"  log:  {log_path}", style="dim"))


class RecordingConsole(Console):
    """Console writing plain (colour-free) text into memory for assertions."""

    def __init__(self, *, quiet: bool = False, json_mode: bool = False) -> None:
        self._buffer = io.StringIO()
        super().__init__(
            no_color=True,
            quiet=quiet,
            json_mode=json_mode,
            file=self._buffer,
            force_terminal=False,
            width=_RECORDING_WIDTH,
        )

    @property
    def text(self) -> str:
        """Everything written so far."""
        return self._buffer.getvalue()

    def clear(self) -> Self:
        """Discard captured text."""
        self._buffer.seek(0)
        self._buffer.truncate()
        return self


def _problem_line(problem: Problem) -> Text:
    marker, style = ("✗", "red") if problem.severity == "error" else ("!", "yellow")
    text = Text(f"  {marker} ", style=style)
    text.append(f"[{problem.code}] ", style=f"bold {style}")
    if problem.path:
        text.append(f"{problem.path} — ", style="dim")
    text.append(problem.message)
    return text
