"""Texture provider port and its request/result types (plan 06 §7)."""

from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import Field

from grounded_pipeline.domain._base import FrozenModel
from grounded_pipeline.domain.ids import ModelId, Sha256
from grounded_pipeline.domain.ports import CancelToken


class GuidanceMode(StrEnum):
    TEXT = "text"
    TEXT_STYLE = "text+style"
    IMAGE = "image"
    IMAGES = "images"


class TextureRequest(FrozenModel):
    model_id: ModelId
    mesh_path: Path  # static.glb
    mode: GuidanceMode
    prompt: str | None
    style_image: Path | None
    guidance_images: dict[str, Path]  # keys front/left/back/right
    model_version: str
    quality: str
    pbr: bool
    bake: bool
    delight: bool
    alignment: str
    seed: int
    request_sha256: Sha256  # over all of the above + mesh sha → cache key


class TextureResult(FrozenModel):
    provider: Literal["tripo"]
    task_id: str
    result_glb: Path
    preview_png: Path | None
    credits_consumed: float
    started_at: str
    finished_at: str
    raw_output: dict[str, Any] = Field(default_factory=dict)  # provider output verbatim


OnTaskCreated = Callable[[str], None]


class TextureProvider(Protocol):
    """Port for texturing a static mesh. Production adapter: ``TripoTextureProvider``."""

    def estimate_credits(self, req: TextureRequest) -> float:
        """Best-effort credit estimate for ``--dry-run`` and budget reservation."""
        ...

    def texture(
        self,
        req: TextureRequest,
        *,
        resume_task_id: str | None,
        cancel: CancelToken,
        on_task_created: OnTaskCreated | None = None,
    ) -> TextureResult:
        """Texture ``req.mesh_path``; re-attach to ``resume_task_id`` when given.

        ``on_task_created`` fires with the provider task id as soon as a task exists, so the caller
        can persist it before polling (resume semantics, plan 10 §1).
        """
        ...
