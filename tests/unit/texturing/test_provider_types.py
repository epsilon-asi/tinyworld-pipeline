from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from tinyworld_pipeline.domain.ports import CancelToken
from tinyworld_pipeline.texturing.provider import (
    GuidanceMode,
    TextureProvider,
    TextureRequest,
    TextureResult,
)


def test_guidance_modes_match_config_prompt_modes() -> None:
    assert {m.value for m in GuidanceMode} == {"text", "text+style", "image", "images"}
    assert GuidanceMode("text+style") is GuidanceMode.TEXT_STYLE


def test_texture_request_requires_cache_key_sha() -> None:
    kwargs = {
        "model_id": "ant",
        "mesh_path": Path("work/static.glb"),
        "mode": GuidanceMode.TEXT,
        "prompt": "ant",
        "style_image": None,
        "guidance_images": {},
        "model_version": "v3.5-20260815",
        "quality": "standard",
        "pbr": True,
        "bake": True,
        "delight": True,
        "alignment": "original_image",
        "seed": 1,
    }
    req = TextureRequest(request_sha256="0" * 64, **kwargs)
    assert req.mode is GuidanceMode.TEXT
    with pytest.raises(ValidationError):
        TextureRequest(request_sha256="nope", **kwargs)


def test_texture_result_defaults() -> None:
    res = TextureResult(
        provider="tripo",
        task_id="t",
        result_glb=Path("work/tripo/result.glb"),
        preview_png=None,
        credits_consumed=10,
        started_at="a",
        finished_at="b",
    )
    assert res.raw_output == {}


class _Fake:
    def estimate_credits(self, req: TextureRequest) -> float:
        return 10.0

    def texture(
        self,
        req: TextureRequest,
        *,
        resume_task_id: str | None,
        cancel: CancelToken,
        on_task_created: object = None,
    ) -> TextureResult:
        raise NotImplementedError


def test_protocol_is_structural() -> None:
    provider: TextureProvider = _Fake()
    assert provider.estimate_credits is not None
