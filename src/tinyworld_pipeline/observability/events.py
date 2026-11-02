"""Structured log event names (plan 14 §4, plan 19 §6).

Every event is a ``noun.verb`` string constant. Packages log with these constants instead of
string literals so the names stay greppable and stable across the run log, the per-stage tee and
tests. The values match the literals that were in use before this module existed; owners of the
other packages adopt the constants at their own pace.
"""

from __future__ import annotations

from typing import Final

# --------------------------------------------------------------------------- run lifecycle
RUN_START: Final[str] = "run.start"
RUN_DONE: Final[str] = "run.done"

# --------------------------------------------------------------------------- stage lifecycle
STAGE_START: Final[str] = "stage.start"
STAGE_SKIP: Final[str] = "stage.skip"
STAGE_DONE: Final[str] = "stage.done"
STAGE_FAIL: Final[str] = "stage.fail"

# --------------------------------------------------------------------------- budget
BUDGET_RESERVE: Final[str] = "budget.reserve"
BUDGET_COMMIT: Final[str] = "budget.commit"

# --------------------------------------------------------------------------- blender
BLENDER_SPAWN: Final[str] = "blender.spawn"
BLENDER_EXIT: Final[str] = "blender.exit"
BLENDER_TEMP_KEPT: Final[str] = "blender.temp_kept"
BLENDER_KILL_FAILED: Final[str] = "blender.kill_failed"

# --------------------------------------------------------------------------- tripo client
TRIPO_REQUEST: Final[str] = "tripo.request"
TRIPO_RESPONSE: Final[str] = "tripo.response"
TRIPO_RETRY: Final[str] = "tripo.retry"
TRIPO_POLL: Final[str] = "tripo.poll"
TRIPO_CREDITS: Final[str] = "tripo.credits"
TRIPO_DOWNLOAD: Final[str] = "tripo.download"
TRIPO_DOWNLOAD_RETRY: Final[str] = "tripo.download_retry"
TRIPO_UPLOAD: Final[str] = "tripo.upload"
TRIPO_TASK_CREATED: Final[str] = "tripo.task_created"
TRIPO_TASK_RESUMED: Final[str] = "tripo.task_resumed"
TRIPO_PREVIEW_SKIPPED: Final[str] = "tripo.preview_skipped"

# --------------------------------------------------------------------------- texture stage
TEXTURE_TASK_PERSISTED: Final[str] = "texture.task_persisted"
TEXTURE_TASK_ABANDONED: Final[str] = "texture.task_abandoned"
TEXTURE_TASK_RESUMING: Final[str] = "texture.task_resuming"

# --------------------------------------------------------------------------- verify/export/handoff
VERIFY_ROUTE: Final[str] = "verify.route"
RFC_RESULT: Final[str] = "rfc.result"
HANDOFF_SYNC: Final[str] = "handoff.sync"

ALL_EVENTS: Final[tuple[str, ...]] = (
    RUN_START,
    RUN_DONE,
    STAGE_START,
    STAGE_SKIP,
    STAGE_DONE,
    STAGE_FAIL,
    BUDGET_RESERVE,
    BUDGET_COMMIT,
    BLENDER_SPAWN,
    BLENDER_EXIT,
    BLENDER_TEMP_KEPT,
    BLENDER_KILL_FAILED,
    TRIPO_REQUEST,
    TRIPO_RESPONSE,
    TRIPO_RETRY,
    TRIPO_POLL,
    TRIPO_CREDITS,
    TRIPO_DOWNLOAD,
    TRIPO_DOWNLOAD_RETRY,
    TRIPO_UPLOAD,
    TRIPO_TASK_CREATED,
    TRIPO_TASK_RESUMED,
    TRIPO_PREVIEW_SKIPPED,
    TEXTURE_TASK_PERSISTED,
    TEXTURE_TASK_ABANDONED,
    TEXTURE_TASK_RESUMING,
    VERIFY_ROUTE,
    RFC_RESULT,
    HANDOFF_SYNC,
)
"""Every event constant, for the uniqueness/format test."""
