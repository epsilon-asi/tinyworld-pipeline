"""``TripoTextureProvider`` — the production :class:`TextureProvider` (plan 10 §2).

The provider is a thin choreography over :class:`~grounded_pipeline.tripo.client.TripoClient`:
upload the mesh and reference images, build the ``texture_prompt`` for the guidance mode, create
the task (or re-attach to ``resume_task_id``), announce the task id, wait, download *immediately*
(signed output URLs expire ~5 min after the detail was fetched) and translate the detail into a
:class:`TextureResult`.

Thread-safety: one provider instance is shared by every model of a run, so it keeps no per-call
state. The exact ``POST /models/texture`` body is returned inside
``TextureResult.raw_output["request_body"]`` (``None`` when a task was resumed) rather than on an
attribute.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog

from grounded_pipeline.domain.errors import TripoDownloadError
from grounded_pipeline.domain.ports import CancelToken
from grounded_pipeline.texturing.provider import (
    GuidanceMode,
    OnTaskCreated,
    TextureRequest,
    TextureResult,
)
from grounded_pipeline.texturing.request_builder import VIEW_ORDER
from grounded_pipeline.tripo.budget import estimate_credits
from grounded_pipeline.tripo.client import TripoClient
from grounded_pipeline.tripo.models import (
    TaskDetail,
    TexturePrompt,
    TextureTaskRequest,
    TripoFileRef,
)

EVENT_TASK_CREATED = "tripo.task_created"
EVENT_TASK_RESUMED = "tripo.task_resumed"
EVENT_CREDITS = "tripo.credits"
EVENT_DOWNLOAD_RETRY = "tripo.download_retry"
EVENT_PREVIEW_SKIPPED = "tripo.preview_skipped"

RESULT_GLB_NAME = "result.glb"
PREVIEW_PNG_NAME = "preview.png"
OUTPUT_DIR_NAME = "tripo"
REQUEST_BODY_KEY = "request_body"
"""Key under which the sent ``TextureTaskRequest`` body is stored in ``raw_output``."""

DEFAULT_POLL_INTERVAL_S = 2.0
DEFAULT_POLL_TIMEOUT_S = 900.0
DEFAULT_PRESIGN_ABOVE_MB = 60

Uploader = Callable[[Path], str]
WorkDirFor = Callable[[TextureRequest], Path]
Now = Callable[[], datetime]

_log = structlog.get_logger(__name__)


def _utc_now() -> datetime:
    return datetime.now(UTC)


def build_texture_prompt(req: TextureRequest, upload: Uploader) -> TexturePrompt:
    """Map a :class:`TextureRequest` onto Tripo's ``texture_prompt`` object (plan 10 §2 table).

    Args:
        req: The request; ``req.mode`` selects the shape.
        upload: Uploads an image and returns its ``file_token`` (called once per image used).

    Returns:
        ``text`` → ``{text}``; ``text+style`` → ``{text, style_image: {file_token}}``;
        ``image`` → ``{image: <front token>}``; ``images`` → ``{images: [{front: tok}, …]}`` in
        view order with missing views omitted.

    Raises:
        ValueError: The request lacks what its mode needs (no prompt, no style image, no front
            view) — the request builder and library validation normally prevent this.
    """
    mode = req.mode
    if mode in (GuidanceMode.TEXT, GuidanceMode.TEXT_STYLE):
        if req.prompt is None:
            msg = f"mode {mode.value!r} requires a prompt"
            raise ValueError(msg)
        if mode is GuidanceMode.TEXT:
            return TexturePrompt(text=req.prompt)
        if req.style_image is None:
            msg = "mode 'text+style' requires a style image"
            raise ValueError(msg)
        token = upload(req.style_image)
        return TexturePrompt(text=req.prompt, style_image=TripoFileRef(file_token=token))
    if "front" not in req.guidance_images:
        msg = f"mode {mode.value!r} requires a 'front' guidance image"
        raise ValueError(msg)
    if mode is GuidanceMode.IMAGE:
        return TexturePrompt(image=upload(req.guidance_images["front"]))
    present = [view for view in VIEW_ORDER if view in req.guidance_images]
    images: list[dict[str, str]] = [{view: upload(req.guidance_images[view])} for view in present]
    return TexturePrompt.model_validate({"images": images})


def build_task_request(
    req: TextureRequest, *, mesh_token: str, prompt: TexturePrompt
) -> TextureTaskRequest:
    """The exact ``POST /models/texture`` body for ``req`` (validated against the DTO)."""
    return TextureTaskRequest.model_validate(
        {
            "input": mesh_token,
            "model": req.model_version,
            "texture_prompt": prompt,
            "pbr": req.pbr,
            "texture_seed": req.seed,
            "texture_alignment": req.alignment,
            "texture_quality": req.quality,
            "bake": req.bake,
            "delight": req.delight,
        }
    )


class TripoTextureProvider:
    """Production texture provider backed by :class:`TripoClient`.

    Args:
        client: Configured client (owns HTTP, retries, redaction).
        credit_table: ``[tripo.credit_table]`` used by :meth:`estimate_credits`.
        poll_interval_s: Initial poll interval (``TripoSection.poll_interval_s``).
        poll_timeout_s: Overall wait budget per task (``TripoSection.poll_timeout_s``).
        presign_above_mb: Upload threshold for the presigned flow.
        work_dir_for: Where to put ``result.glb``/``preview.png`` for a request; defaults to
            ``req.mesh_path.parent / "tripo"`` (i.e. ``work/tripo``).
        now: Wall-clock source for ``started_at``/``finished_at`` (tests inject a fixed one).
    """

    def __init__(
        self,
        client: TripoClient,
        *,
        credit_table: Mapping[str, float],
        poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
        poll_timeout_s: float = DEFAULT_POLL_TIMEOUT_S,
        presign_above_mb: int = DEFAULT_PRESIGN_ABOVE_MB,
        work_dir_for: WorkDirFor | None = None,
        now: Now | None = None,
    ) -> None:
        self._client = client
        self._credit_table = dict(credit_table)
        self._poll_interval_s = poll_interval_s
        self._poll_timeout_s = poll_timeout_s
        self._presign_above_mb = presign_above_mb
        self._work_dir_for = work_dir_for
        self._now = now or _utc_now

    # ------------------------------------------------------------------ TextureProvider
    def estimate_credits(self, req: TextureRequest) -> float:
        """Credit estimate from the table (quality + style-image surcharge when used)."""
        uses_style = req.mode is GuidanceMode.TEXT_STYLE and req.style_image is not None
        return estimate_credits(req.quality, style_image=uses_style, table=self._credit_table)

    def texture(
        self,
        req: TextureRequest,
        *,
        resume_task_id: str | None,
        cancel: CancelToken,
        on_task_created: OnTaskCreated | None = None,
    ) -> TextureResult:
        """Texture ``req.mesh_path`` on Tripo (see module docstring for the choreography).

        With ``resume_task_id`` no upload or create call is made: the task detail is fetched and
        either downloaded straight away (already ``success``) or polled to completion.
        ``on_task_created`` fires before any polling in both paths so the caller can persist the
        id.

        Raises:
            TripoError: Any API/transport/task failure from the client.
            TripoDownloadError: The result could not be downloaded even after re-fetching a fresh
                signed URL, or the task succeeded without a ``model_url``.
            CancelledError: ``cancel`` was set while waiting.
        """
        log = _log.bind(model_id=req.model_id, request_sha256=req.request_sha256)
        started_at = self._now().isoformat()
        out_dir = self._out_dir(req)
        body: TextureTaskRequest | None = None
        if resume_task_id is not None:
            task_id = resume_task_id
            log.info(EVENT_TASK_RESUMED, task_id=task_id)
            if on_task_created is not None:
                on_task_created(task_id)
            detail = self._client.get_task(task_id)
            if not detail.is_success:
                detail = self._wait(task_id, cancel)
        else:
            body = self._create_body(req)
            task_id = self._client.create_texture_task(body).task_id
            log.info(EVENT_TASK_CREATED, task_id=task_id, mode=req.mode.value)
            if on_task_created is not None:
                on_task_created(task_id)
            detail = self._wait(task_id, cancel)

        result_glb = self._download_result(task_id, detail, out_dir / RESULT_GLB_NAME)
        preview_png = self._download_preview(task_id, detail, out_dir / PREVIEW_PNG_NAME)
        spent = detail.credits_consumed
        if spent is None:
            spent = self.estimate_credits(req)
            log.warning(EVENT_CREDITS, task_id=task_id, credits=spent, source="estimate")
        else:
            log.info(EVENT_CREDITS, task_id=task_id, credits=spent, source="task_detail")
        raw: dict[str, Any] = detail.model_dump(mode="json")
        raw[REQUEST_BODY_KEY] = (
            None if body is None else body.model_dump(mode="json", exclude_none=True)
        )
        return TextureResult(
            provider="tripo",
            task_id=task_id,
            result_glb=result_glb,
            preview_png=preview_png,
            credits_consumed=spent,
            started_at=started_at,
            finished_at=self._now().isoformat(),
            raw_output=raw,
        )

    # ------------------------------------------------------------------ internals
    def _out_dir(self, req: TextureRequest) -> Path:
        if self._work_dir_for is not None:
            return self._work_dir_for(req)
        return req.mesh_path.parent / OUTPUT_DIR_NAME

    def _upload(self, path: Path) -> str:
        return self._client.upload(path, presign_above_mb=self._presign_above_mb)

    def _create_body(self, req: TextureRequest) -> TextureTaskRequest:
        mesh_token = self._upload(req.mesh_path)
        prompt = build_texture_prompt(req, self._upload)
        return build_task_request(req, mesh_token=mesh_token, prompt=prompt)

    def _wait(self, task_id: str, cancel: CancelToken) -> TaskDetail:
        return self._client.wait_for_task(
            task_id,
            interval_s=self._poll_interval_s,
            timeout_s=self._poll_timeout_s,
            cancel=cancel,
        )

    def _download_result(self, task_id: str, detail: TaskDetail, dest: Path) -> Path:
        url = _model_url(detail)
        if url is None:
            raise TripoDownloadError(
                f"task {task_id} succeeded but has no output.model_url",
                detail={"task_id": task_id, "output": _output_dump(detail)},
            )
        try:
            return self._client.download(url, dest)
        except TripoDownloadError as exc:
            # Signed URLs expire ~5 min after the detail was fetched: refresh once, then give up.
            _log.warning(EVENT_DOWNLOAD_RETRY, task_id=task_id, error=exc.code, file=dest.name)
            fresh_url = _model_url(self._client.get_task(task_id))
            if fresh_url is None:
                raise
            return self._client.download(fresh_url, dest)

    def _download_preview(self, task_id: str, detail: TaskDetail, dest: Path) -> Path | None:
        url = _preview_url(detail)
        if url is None:
            return None
        try:
            return self._client.download(url, dest)
        except TripoDownloadError as exc:
            _log.warning(EVENT_DOWNLOAD_RETRY, task_id=task_id, error=exc.code, file=dest.name)
            fresh_url = _preview_url(self._client.get_task(task_id))
            if fresh_url is None:
                return None
            try:
                return self._client.download(fresh_url, dest)
            except TripoDownloadError as retry_exc:
                # The preview is a convenience render, never a stage output: degrade, don't fail.
                _log.warning(EVENT_PREVIEW_SKIPPED, task_id=task_id, error=retry_exc.code)
                return None


def _model_url(detail: TaskDetail) -> str | None:
    if detail.output is None or detail.output.model_url is None:
        return None
    return str(detail.output.model_url)


def _preview_url(detail: TaskDetail) -> str | None:
    if detail.output is None or detail.output.rendered_image_url is None:
        return None
    return str(detail.output.rendered_image_url)


def _output_dump(detail: TaskDetail) -> dict[str, Any] | None:
    return None if detail.output is None else detail.output.model_dump(mode="json")


__all__ = [
    "DEFAULT_POLL_INTERVAL_S",
    "DEFAULT_POLL_TIMEOUT_S",
    "DEFAULT_PRESIGN_ABOVE_MB",
    "EVENT_CREDITS",
    "EVENT_DOWNLOAD_RETRY",
    "EVENT_PREVIEW_SKIPPED",
    "EVENT_TASK_CREATED",
    "EVENT_TASK_RESUMED",
    "OUTPUT_DIR_NAME",
    "PREVIEW_PNG_NAME",
    "REQUEST_BODY_KEY",
    "RESULT_GLB_NAME",
    "TripoTextureProvider",
    "build_task_request",
    "build_texture_prompt",
]
