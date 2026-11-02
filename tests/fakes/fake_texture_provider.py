"""``FakeTextureProvider`` — the test adapter for the ``TextureProvider`` port (plan 15 §3.4).

It never touches the network. ``texture()`` records the call (including ``resume_task_id``),
fires ``on_task_created`` with a deterministic id, optionally raises a scripted error, writes a
synthetic textured GLB (built with :mod:`fakes.glb_builder`) plus a tiny preview PNG under
``<mesh dir>/tripo/`` and returns a :class:`TextureResult` shaped like the Tripo adapter's,
including ``raw_output["output"]["normalize_scale"]`` and ``raw_output["request_body"]``.

The GLB flavour is a ``result_glb_factory`` so a test can ask for the same mesh as the input,
re-unwrapped UVs, a remeshed surface or a multi-material result.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from fakes.glb_builder import (
    MaterialSpec,
    cube,
    cube_mesh,
    encode_image,
    from_mesh,
    solid_image,
    textured_set,
    write_glb,
)
from grounded_pipeline.domain.errors import CancelledError
from grounded_pipeline.domain.ports import CancelToken
from grounded_pipeline.texturing.provider import OnTaskCreated, TextureRequest, TextureResult

GlbFactory = Callable[[TextureRequest], bytes]

OUTPUT_DIR_NAME = "tripo"
RESULT_GLB_NAME = "result.glb"
PREVIEW_PNG_NAME = "preview.png"
DEFAULT_TASK_PREFIX = "fake_task"
FAKE_STARTED_AT = "2026-11-02T12:00:00+00:00"
FAKE_FINISHED_AT = "2026-11-02T12:01:30+00:00"


# --------------------------------------------------------------------------- GLB factories
def glb_textured_cube(req: TextureRequest) -> bytes:
    """Default: the ``glb_builder`` cube with a full PBR texture set (same UVs as ``cube()``)."""
    images, material = textured_set()
    return cube(images=images, materials=[material])


def glb_copy_input(req: TextureRequest) -> bytes:
    """Return the input GLB bytes untouched (identical geometry/UVs, no textures)."""
    return req.mesh_path.read_bytes()


def glb_reunwrapped_cube(req: TextureRequest) -> bytes:
    """Cube with identical positions but different UVs (as a provider re-unwrapping would do)."""
    mesh = cube_mesh()
    assert mesh.uvs is not None
    uvs = (1.0 - mesh.uvs).astype(np.float32)
    images, material = textured_set()
    return from_mesh(mesh.positions, uvs, mesh.indices, images=images, materials=[material])


def glb_remeshed_cube(req: TextureRequest) -> bytes:
    """Cube with every triangle split into its own vertices and shuffled order (remeshed)."""
    images, material = textured_set()
    return cube(split_seams=True, shuffle=True, images=images, materials=[material])


def glb_multi_material_cube(req: TextureRequest) -> bytes:
    """Cube split into two primitives with two materials (the ``E_MULTI_MATERIAL`` case)."""
    images, material = textured_set()
    second = MaterialSpec(name="mat2", base_color_image=0)
    return cube(primitives=2, images=images, materials=[material, second])


def scaled_cube_factory(scale: float) -> GlbFactory:
    """Factory producing a uniformly scaled cube (mimics Tripo's ``normalize_scale``)."""

    def factory(req: TextureRequest) -> bytes:
        images, material = textured_set()
        return cube(scale=scale, images=images, materials=[material])

    return factory


# --------------------------------------------------------------------------- the fake
@dataclass(frozen=True)
class TextureCall:
    """One recorded ``texture()`` invocation."""

    request: TextureRequest
    resume_task_id: str | None
    task_id: str
    cancelled: bool


@dataclass
class FakeTextureProvider:
    """Scripted :class:`TextureProvider`.

    Args:
        result_glb_factory: Builds the result GLB bytes for a request (default textured cube).
        credits: ``credits_consumed`` reported in the result (also the estimate unless
            ``estimate`` is given).
        raise_: Exception raised by ``texture()`` *after* ``on_task_created`` fired (so resume
            semantics can be tested); ``None`` for success.
        preview: Whether to write and report ``preview.png``.
        estimate: Value returned by ``estimate_credits`` (defaults to ``credits``).
        normalize_scale: Reported in ``raw_output["output"]`` (``None`` omits both normalize
            fields).
        normalize_translation: Reported alongside ``normalize_scale``.
        task_id_prefix: Deterministic ids are ``f"{prefix}_{n:04d}"`` per created task.
        raise_before_task_created: When true, ``raise_`` fires before ``on_task_created``.
    """

    result_glb_factory: GlbFactory = glb_textured_cube
    credits: float = 10.0
    raise_: BaseException | None = None
    preview: bool = True
    estimate: float | None = None
    normalize_scale: float | None = None
    normalize_translation: tuple[float, float, float] | None = None
    task_id_prefix: str = DEFAULT_TASK_PREFIX
    raise_before_task_created: bool = False
    calls: list[TextureCall] = field(default_factory=list)
    estimate_calls: list[TextureRequest] = field(default_factory=list)
    created_task_ids: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------ TextureProvider
    def estimate_credits(self, req: TextureRequest) -> float:
        """Return the scripted estimate (recorded)."""
        self.estimate_calls.append(req)
        return self.credits if self.estimate is None else self.estimate

    def texture(
        self,
        req: TextureRequest,
        *,
        resume_task_id: str | None,
        cancel: CancelToken,
        on_task_created: OnTaskCreated | None = None,
    ) -> TextureResult:
        """Record the call, announce a task id, then either raise or write a synthetic result."""
        task_id = resume_task_id or self._next_task_id()
        self.calls.append(
            TextureCall(
                request=req,
                resume_task_id=resume_task_id,
                task_id=task_id,
                cancelled=cancel.is_set(),
            )
        )
        if self.raise_ is not None and self.raise_before_task_created:
            raise self.raise_
        if on_task_created is not None:
            on_task_created(task_id)
        if self.raise_ is not None:
            raise self.raise_
        if cancel.is_set():
            raise CancelledError(f"cancelled while waiting for task {task_id}")
        out_dir = req.mesh_path.parent / OUTPUT_DIR_NAME
        result_glb = write_glb(out_dir / RESULT_GLB_NAME, self.result_glb_factory(req))
        preview_png = self._write_preview(out_dir / PREVIEW_PNG_NAME) if self.preview else None
        return TextureResult(
            provider="tripo",
            task_id=task_id,
            result_glb=result_glb,
            preview_png=preview_png,
            credits_consumed=self.credits,
            started_at=FAKE_STARTED_AT,
            finished_at=FAKE_FINISHED_AT,
            raw_output=self._raw_output(req, task_id, resumed=resume_task_id is not None),
        )

    # ------------------------------------------------------------------ helpers
    @property
    def last_call(self) -> TextureCall:
        """The most recent recorded call (``IndexError`` when none)."""
        return self.calls[-1]

    def _next_task_id(self) -> str:
        task_id = f"{self.task_id_prefix}_{len(self.created_task_ids) + 1:04d}"
        self.created_task_ids.append(task_id)
        return task_id

    @staticmethod
    def _write_preview(path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(encode_image(solid_image((90, 90, 90, 255), size=4)))
        return path

    def _raw_output(self, req: TextureRequest, task_id: str, *, resumed: bool) -> dict[str, Any]:
        output: dict[str, Any] = {
            "model_url": f"fake://tripo/{task_id}/model_pbr.glb",
            "rendered_image_url": f"fake://tripo/{task_id}/preview.png" if self.preview else None,
        }
        if self.normalize_scale is not None:
            output["normalize_scale"] = self.normalize_scale
            output["normalize_translation"] = (
                None if self.normalize_translation is None else list(self.normalize_translation)
            )
        body: dict[str, Any] | None = None
        if not resumed:
            body = {
                "input": f"file_fake_{req.request_sha256[:12]}",
                "model": req.model_version,
                "texture_prompt": (
                    {"text": req.prompt} if req.prompt else {"image": "file_fake_front"}
                ),
                "pbr": req.pbr,
                "texture_seed": req.seed,
                "texture_alignment": req.alignment,
                "texture_quality": req.quality,
                "bake": req.bake,
                "delight": req.delight,
            }
        return {
            "task_id": task_id,
            "type": "texture_model",
            "status": "success",
            "progress": 100,
            "output": output,
            "credits_consumed": self.credits,
            "created_at": FAKE_STARTED_AT,
            "completed_at": FAKE_FINISHED_AT,
            "request_body": body,
        }


__all__ = [
    "DEFAULT_TASK_PREFIX",
    "FAKE_FINISHED_AT",
    "FAKE_STARTED_AT",
    "FakeTextureProvider",
    "GlbFactory",
    "TextureCall",
    "glb_copy_input",
    "glb_multi_material_cube",
    "glb_remeshed_cube",
    "glb_reunwrapped_cube",
    "glb_textured_cube",
    "scaled_cube_factory",
]
