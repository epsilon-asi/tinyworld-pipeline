"""Blender-side job failure (plan 06 §5, 14 §2).

Runs inside Blender's Python: standard library only.
"""

from __future__ import annotations

from typing import Any


class JobFailure(Exception):  # noqa: N818 - protocol name fixed by plan 16 WP-03
    """An *expected* job failure with a stable ``E_BJ_*`` code.

    ``entry.py`` turns it into ``JobResult(ok=False, error=JobError(code, message, detail))`` and
    exits 0 (the failure is part of the protocol, not a crash).
    """

    def __init__(self, code: str, message: str, detail: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.detail: dict[str, Any] = dict(detail or {})
