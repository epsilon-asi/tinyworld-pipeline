"""Stage ``texture`` (plan 10 §1, normative): reserve credits, call the provider, persist results.

Resume contract (for the orchestrator): as soon as the provider reports a task id the stage
persists ``stages["texture"].extra = {"task_id", "request_sha256"}`` through
``ctx.state.update``. On the next run the stored id is reused only when ``options.resume_tasks``
is on **and** the stored ``request_sha256`` equals the freshly built one; otherwise the old task
is abandoned (logged, never cancelled — Tripo has no cancel endpoint) and a new one is created.

Outputs (all relative to the model folder, written atomically):

* ``work/tripo/request.json`` — ``{"request": TextureRequest (paths relative to the model folder),
  "task_request": exact POST /models/texture body or null when resumed, "resumed_task_id": …}``
* ``work/tripo/task.json`` — the ``TextureResult`` (paths relative; raw task detail included)
* ``work/tripo/result.glb`` (+ ``preview.png`` when Tripo provided a render)
"""

from __future__ import annotations

import json
import os
from pathlib import Path, PurePosixPath
from typing import Any

from tinyworld_pipeline.domain.errors import StagePrereqError
from tinyworld_pipeline.domain.state import StageName, StageOutcome
from tinyworld_pipeline.pipeline.context import ModelContext, PipelineContext
from tinyworld_pipeline.texturing.provider import TextureRequest, TextureResult
from tinyworld_pipeline.texturing.request_builder import (
    build_texture_request,
    texture_stage_fingerprint,
)

STATIC_GLB = PurePosixPath("work/static.glb")
TRIPO_DIR = PurePosixPath("work/tripo")
RESULT_GLB = TRIPO_DIR / "result.glb"
TASK_JSON = TRIPO_DIR / "task.json"
REQUEST_JSON = TRIPO_DIR / "request.json"
PREVIEW_PNG = TRIPO_DIR / "preview.png"

EXTRA_TASK_ID = "task_id"
EXTRA_REQUEST_SHA = "request_sha256"
RAW_REQUEST_BODY_KEY = "request_body"

EVENT_TASK_PERSISTED = "texture.task_persisted"
EVENT_TASK_ABANDONED = "texture.task_abandoned"
EVENT_TASK_RESUMING = "texture.task_resuming"


# --------------------------------------------------------------------------- documents
def _relative(path: Path, folder: Path) -> str:
    """Forward-slash path relative to ``folder`` (``../`` segments allowed for library files)."""
    return PurePosixPath(os.path.relpath(path.resolve(), folder.resolve())).as_posix()


def _relativise(data: dict[str, Any], keys: tuple[str, ...], folder: Path) -> dict[str, Any]:
    out = dict(data)
    for key in keys:
        value = out.get(key)
        if isinstance(value, str) and value:
            out[key] = _relative(Path(value), folder)
        elif isinstance(value, dict):
            out[key] = {k: _relative(Path(v), folder) for k, v in value.items()}
    return out


def request_document(req: TextureRequest, result: TextureResult, folder: Path) -> dict[str, Any]:
    """Contents of ``work/tripo/request.json`` (paths relative to the model folder)."""
    body = result.raw_output.get(RAW_REQUEST_BODY_KEY)
    return {
        "request": _relativise(
            req.model_dump(mode="json"), ("mesh_path", "style_image", "guidance_images"), folder
        ),
        "task_request": body,
        "resumed_task_id": result.task_id if body is None else None,
    }


def task_document(result: TextureResult, folder: Path) -> dict[str, Any]:
    """Contents of ``work/tripo/task.json`` (paths relative to the model folder)."""
    return _relativise(result.model_dump(mode="json"), ("result_glb", "preview_png"), folder)


def write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    """Write ``data`` as canonical JSON to ``path`` via a temp sibling and an atomic replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    text = json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


# --------------------------------------------------------------------------- the stage
class TextureStage:
    """Send ``work/static.glb`` to the texture provider and store the textured GLB."""

    name: StageName = "texture"

    def input_fingerprint(self, model: ModelContext, ctx: PipelineContext) -> str:
        """Plan 10 §1 composition; see :func:`texture_stage_fingerprint`."""
        return texture_stage_fingerprint(
            model.resolved,
            static_glb=model.paths.folder / Path(STATIC_GLB),
            library_root=ctx.library.paths.root,
        )

    def run(self, model: ModelContext, ctx: PipelineContext) -> StageOutcome:
        """Reserve → texture (resumable) → commit → write ``request.json``/``task.json``.

        Raises:
            StagePrereqError: ``work/static.glb`` is missing (``prepare`` did not run).
            BudgetExceededError: The estimate does not fit the remaining run budget (raised
                before any provider call).
            PipelineError: Whatever the provider raises; the reservation is released first.
        """
        static_glb = model.paths.folder / Path(STATIC_GLB)
        if not static_glb.is_file():
            raise StagePrereqError(
                f"{STATIC_GLB} missing for {model.model_id}; run the `prepare` stage first",
                detail={"model_id": model.model_id, "path": str(static_glb)},
            )
        req = build_texture_request(
            model.resolved, static_glb=static_glb, library_root=ctx.library.paths.root
        )
        reservation = ctx.budget.reserve(ctx.provider.estimate_credits(req))
        resume = self._resume_task_id(model, ctx, req)

        def on_task_created(task_id: str) -> None:
            ctx.state.update(
                model.model_id,
                self.name,
                extra={EXTRA_TASK_ID: task_id, EXTRA_REQUEST_SHA: req.request_sha256},
            )
            model.log.info(EVENT_TASK_PERSISTED, stage=self.name, task_id=task_id)

        try:
            result = ctx.provider.texture(
                req, resume_task_id=resume, cancel=ctx.cancel, on_task_created=on_task_created
            )
        except BaseException:
            ctx.budget.release(reservation)
            raise
        ctx.budget.commit(reservation, result.credits_consumed)

        folder = model.paths.folder
        write_json_atomic(folder / Path(REQUEST_JSON), request_document(req, result, folder))
        write_json_atomic(folder / Path(TASK_JSON), task_document(result, folder))
        outputs = [str(RESULT_GLB), str(TASK_JSON), str(REQUEST_JSON)]
        return StageOutcome(status="succeeded", outputs=outputs, extra=_outcome_extra(req, result))

    def _resume_task_id(
        self, model: ModelContext, ctx: PipelineContext, req: TextureRequest
    ) -> str | None:
        if not ctx.options.resume_tasks:
            return None
        record = ctx.state.load(model.model_id).stages.get(self.name)
        if record is None:
            return None
        stored = record.extra.get(EXTRA_TASK_ID)
        if not isinstance(stored, str) or not stored:
            return None
        if record.extra.get(EXTRA_REQUEST_SHA) != req.request_sha256:
            model.log.warning(
                EVENT_TASK_ABANDONED, stage=self.name, task_id=stored, reason="request changed"
            )
            return None
        model.log.info(EVENT_TASK_RESUMING, stage=self.name, task_id=stored)
        return stored


def _outcome_extra(req: TextureRequest, result: TextureResult) -> dict[str, Any]:
    output = result.raw_output.get("output")
    output = output if isinstance(output, dict) else {}
    return {
        EXTRA_TASK_ID: result.task_id,
        "credits": result.credits_consumed,
        EXTRA_REQUEST_SHA: req.request_sha256,
        "normalize_scale": output.get("normalize_scale"),
        "normalize_translation": output.get("normalize_translation"),
    }


__all__ = [
    "EVENT_TASK_ABANDONED",
    "EVENT_TASK_PERSISTED",
    "EVENT_TASK_RESUMING",
    "EXTRA_REQUEST_SHA",
    "EXTRA_TASK_ID",
    "PREVIEW_PNG",
    "REQUEST_JSON",
    "RESULT_GLB",
    "STATIC_GLB",
    "TASK_JSON",
    "TRIPO_DIR",
    "TextureStage",
    "request_document",
    "task_document",
    "write_json_atomic",
]
