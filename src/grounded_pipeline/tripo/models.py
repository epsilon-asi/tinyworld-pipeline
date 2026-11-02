"""Tripo API v3 DTOs (plan 06 §8, appendix A; verified against developers.tripo3d.ai 2026-11-02)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, HttpUrl, model_validator

from grounded_pipeline.domain._base import FrozenModel

TaskStatus = Literal[
    "queued", "running", "success", "failed", "cancelled", "banned", "expired", "unknown"
]
TERMINAL_FAILURE_STATUSES: frozenset[str] = frozenset({"failed", "cancelled", "banned", "expired"})
ViewName = Literal["front", "left", "back", "right"]
VIEW_NAMES: tuple[ViewName, ...] = ("front", "left", "back", "right")

MIN_MULTIVIEW_IMAGES = 2


class ApiEnvelope[T](BaseModel):
    """``{"code": 0, "data": …}`` on success; ``message``/``suggestion``/``request_id`` on
    failure."""

    model_config = ConfigDict(extra="allow", frozen=True)

    code: int
    data: T | None = None
    status: str | None = None
    message: str | None = None
    suggestion: str | None = None
    request_id: str | None = None

    @property
    def ok(self) -> bool:
        return self.code == 0


class FileUploadData(FrozenModel):
    file_token: str


class PresignData(FrozenModel):
    presigned_url: HttpUrl
    file_token: str
    expires_in: int | None = None


class TripoFileRef(FrozenModel):
    """Reference to an uploaded file or a public URL. Exactly one of the two is set."""

    file_token: str | None = None
    url: str | None = None

    @model_validator(mode="after")
    def _exactly_one(self) -> TripoFileRef:
        if (self.file_token is None) == (self.url is None):
            msg = "TripoFileRef needs exactly one of file_token or url"
            raise ValueError(msg)
        return self


class TexturePrompt(FrozenModel):
    """``texture_prompt`` object.

    ``text`` / ``image`` / ``images`` are mutually exclusive; ``style_image`` requires ``text``;
    ``images`` is a view-key array with ``front`` and at least two views.
    """

    text: str | None = None
    style_image: TripoFileRef | None = None
    image: str | None = None  # file_token or URL
    images: list[dict[ViewName, str]] | None = None

    @model_validator(mode="after")
    def _rules(self) -> TexturePrompt:
        modes = [m for m in ("text", "image", "images") if getattr(self, m) is not None]
        if len(modes) != 1:
            msg = f"exactly one of text/image/images must be set, got {modes or 'none'}"
            raise ValueError(msg)
        if self.style_image is not None and self.text is None:
            msg = "style_image is only allowed together with text"
            raise ValueError(msg)
        if self.images is not None:
            views: list[str] = []
            for entry in self.images:
                if len(entry) != 1:
                    msg = "each images[] entry must have exactly one view key"
                    raise ValueError(msg)
                views.extend(entry.keys())
            if len(set(views)) != len(views):
                msg = "duplicate view in images[]"
                raise ValueError(msg)
            if "front" not in views:
                msg = "images[] requires a front view"
                raise ValueError(msg)
            if len(views) < MIN_MULTIVIEW_IMAGES:
                msg = "images[] requires at least two views"
                raise ValueError(msg)
        return self


class TextureTaskRequest(FrozenModel):
    """Body of ``POST /models/texture`` (dumped with ``exclude_none=True``)."""

    input: str  # file_token | task_id | URL
    model: str
    texture_prompt: TexturePrompt | None = None
    pbr: bool = True
    texture_seed: int | None = None
    texture_alignment: Literal["original_image", "geometry"] = "original_image"
    texture_quality: Literal["fast", "standard", "detailed", "extreme"] = "standard"
    bake: bool = True
    delight: bool = True
    part_names: list[str] | None = None
    compress: Literal["geometry"] | None = None  # never set by us: we must parse the GLB


class TaskCreateData(FrozenModel):
    task_id: str


class TaskOutput(BaseModel):
    """Task output. Forward compatible (``extra="allow"``)."""

    model_config = ConfigDict(extra="allow", frozen=True)

    model_url: HttpUrl | None = None  # signed CDN URL; expires ~5 min → download immediately
    rendered_image_url: HttpUrl | None = None
    normalize_scale: float | None = None  # uploaded meshes are rescaled by Tripo
    normalize_translation: tuple[float, float, float] | None = None


class TaskDetail(BaseModel):
    """``data`` of ``GET /tasks/{task_id}``."""

    model_config = ConfigDict(extra="allow", frozen=True)

    task_id: str
    type: str
    status: TaskStatus
    progress: int = 0
    error_code: int | None = None
    error_message: str | None = None
    output: TaskOutput | None = None
    credits_consumed: float | None = None
    created_at: str | None = None
    completed_at: str | None = None

    @property
    def is_terminal_failure(self) -> bool:
        return self.status in TERMINAL_FAILURE_STATUSES

    @property
    def is_success(self) -> bool:
        return self.status == "success"


class TaskBatchData(BaseModel):
    """``data`` of ``POST /tasks/list`` (verified 2026-11-02): details keyed by task id plus the
    ids the account does not own (``missed`` may be null or omitted when all were found)."""

    model_config = ConfigDict(extra="allow", frozen=True)

    tasks: dict[str, TaskDetail] = {}
    missed: list[str] | None = None


class BalanceData(FrozenModel):
    balance: float
    frozen: float


class UsageEntry(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)

    task_id: str
    type: str
    credits_consumed: float
    created_at: str | None = None


def envelope_of[T](data_type: type[T]) -> type[ApiEnvelope[T]]:
    """``ApiEnvelope[data_type]`` helper so call sites read ``envelope_of(TaskDetail)``."""
    return ApiEnvelope[data_type]  # type: ignore[valid-type]  # pydantic generic parametrisation


__all__ = [
    "MIN_MULTIVIEW_IMAGES",
    "TERMINAL_FAILURE_STATUSES",
    "VIEW_NAMES",
    "ApiEnvelope",
    "BalanceData",
    "FileUploadData",
    "PresignData",
    "TaskBatchData",
    "TaskCreateData",
    "TaskDetail",
    "TaskOutput",
    "TaskStatus",
    "TexturePrompt",
    "TextureTaskRequest",
    "TripoFileRef",
    "UsageEntry",
    "ViewName",
    "envelope_of",
]
