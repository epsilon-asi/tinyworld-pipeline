"""Typed error hierarchy and validation problems (plan 14 §2-§3).

Every anticipated failure is a :class:`PipelineError` subclass with a stable ``code`` that appears
in state files, run reports, ``--json`` output and tests. Exit codes follow plan 08 §2.
"""

from __future__ import annotations

from typing import Any, ClassVar, Literal

from grounded_pipeline.domain._base import FrozenModel
from grounded_pipeline.domain.ids import ModelId


class Problem(FrozenModel):
    """One validation finding (error or warning) with an optional one-line fix."""

    code: str
    severity: Literal["error", "warning"]
    model_id: ModelId | None = None
    path: str | None = None
    message: str
    hint: str | None = None

    def render(self) -> str:
        """Single-line human rendering: ``[E_CODE] model.md:key — message (hint)``."""
        where = f"{self.path} — " if self.path else ""
        hint = f" ({self.hint})" if self.hint else ""
        return f"[{self.code}] {where}{self.message}{hint}"


class PipelineError(Exception):
    """Base class for every anticipated failure."""

    code: ClassVar[str] = "E_PIPELINE"
    exit_code: ClassVar[int] = 1

    def __init__(
        self,
        message: str,
        *,
        detail: dict[str, Any] | None = None,
        hint: str | None = None,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.detail: dict[str, Any] = dict(detail or {})
        self.hint = hint
        if cause is not None:
            self.__cause__ = cause

    def __str__(self) -> str:
        return self.message

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly rendering used by ``--json`` output and reports."""
        out: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.hint:
            out["hint"] = self.hint
        if self.detail:
            out["detail"] = self.detail
        return out


# --------------------------------------------------------------------------- usage / configuration
class UsageError(PipelineError):
    code = "E_USAGE"
    exit_code = 2


class ConfigError(PipelineError):
    code = "E_CONFIG"
    exit_code = 2


class ValidationError(PipelineError):
    """Library/model validation failed. Carries the full list of problems."""

    code = "E_VALIDATION"
    exit_code = 3

    def __init__(self, problems: list[Problem], *, message: str | None = None) -> None:
        errors = [p for p in problems if p.severity == "error"]
        msg = message or f"{len(errors)} validation error(s)"
        super().__init__(msg, detail={"problems": [p.model_dump(mode="json") for p in problems]})
        self.problems = list(problems)


class StagePrereqError(PipelineError):
    code = "E_STAGE_PREREQ"
    exit_code = 3


class LibraryBusyError(PipelineError):
    code = "E_LIBRARY_BUSY"
    exit_code = 2


# --------------------------------------------------------------------------- budget
class BudgetExceededError(PipelineError):
    code = "E_BUDGET"
    exit_code = 5


# --------------------------------------------------------------------------- blender
class BlenderNotFoundError(ConfigError):
    code = "E_BLENDER_NOT_FOUND"


class BlenderVersionError(ConfigError):
    code = "E_BLENDER_VERSION"


class BlenderTimeoutError(PipelineError):
    code = "E_BLENDER_TIMEOUT"


class BlenderCrashError(PipelineError):
    code = "E_BLENDER_CRASH"


class BlenderNoResultError(PipelineError):
    code = "E_BLENDER_NO_RESULT"


class BlenderJobError(PipelineError):
    """A Blender job returned ``ok=False``. The code is the job's own ``E_BJ_*`` code."""

    code = "E_BJ"

    def __init__(
        self,
        job_code: str,
        message: str,
        *,
        detail: dict[str, Any] | None = None,
        hint: str | None = None,
        traceback: str | None = None,
    ) -> None:
        super().__init__(message, detail=detail, hint=hint)
        self.job_code = job_code
        self.traceback = traceback

    def to_dict(self) -> dict[str, Any]:
        out = super().to_dict()
        out["code"] = self.job_code
        return out


class ProtocolDriftError(PipelineError):
    code = "E_PROTOCOL"


# --------------------------------------------------------------------------- tripo
class TripoError(PipelineError):
    code = "E_TRIPO"


class TripoAuthError(TripoError):
    code = "E_TRIPO_AUTH"
    exit_code = 2


class TripoApiError(TripoError):
    code = "E_TRIPO_API"


class TripoRateLimitedError(TripoError):
    code = "E_TRIPO_RATE_LIMIT"

    def __init__(
        self,
        message: str,
        *,
        retry_after_s: float | None = None,
        detail: dict[str, Any] | None = None,
        hint: str | None = None,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(message, detail=detail, hint=hint, cause=cause)
        self.retry_after_s = retry_after_s


class TripoServerError(TripoError):
    code = "E_TRIPO_SERVER"


class TripoTransientError(TripoError):
    code = "E_TRIPO_TRANSIENT"


class TripoTaskFailedError(TripoError):
    code = "E_TRIPO_TASK_FAILED"


class TripoTaskTimeoutError(TripoError):
    code = "E_TRIPO_TASK_TIMEOUT"


class TripoAmbiguousCreateError(TripoError):
    code = "E_TRIPO_AMBIGUOUS_CREATE"


class TripoDownloadError(TripoError):
    code = "E_TRIPO_DOWNLOAD"


# --------------------------------------------------------------------------- verify/apply/export
class VerifySelfMismatchError(PipelineError):
    code = "E_VERIFY_SELF_MISMATCH"


class GeometryChangedError(PipelineError):
    code = "E_GEOMETRY_CHANGED"


class UnsupportedGlbError(PipelineError):
    code = "E_GLB_UNSUPPORTED"


class MultiMaterialResultError(PipelineError):
    code = "E_MULTI_MATERIAL"


class TexturePackError(PipelineError):
    code = "E_TEXTURE_PACK"


class RigFidelityError(PipelineError):
    code = "E_RIG_FIDELITY"


class AnimNameCollisionError(PipelineError):
    code = "E_ANIM_NAME_COLLISION"


class HandoffError(PipelineError):
    code = "E_HANDOFF"


class PromptTooLongError(ValidationError):
    """Composed prompt exceeds the provider limit (plan 10 §6)."""

    code = "E_PROMPT_TOO_LONG"


class CancelledError(PipelineError):
    code = "E_CANCELLED"
    exit_code = 130


# Plan 14 §2 names (kept as aliases; ruff N818 requires the *Error suffix on the canonical names).
TripoRateLimited = TripoRateLimitedError
TripoTaskFailed = TripoTaskFailedError
TripoTaskTimeout = TripoTaskTimeoutError


def all_error_classes() -> list[type[PipelineError]]:
    """Every concrete :class:`PipelineError` subclass defined here (for uniqueness tests)."""
    found: list[type[PipelineError]] = []
    stack: list[type[PipelineError]] = [PipelineError]
    while stack:
        cls = stack.pop()
        found.append(cls)
        stack.extend(cls.__subclasses__())
    return found
