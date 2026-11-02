"""``TripoTextureProvider`` against the real ``TripoClient`` with ``respx`` (plan 10 §2, §8)."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

from fakes.glb_builder import cube, write_glb
from grounded_pipeline.domain.errors import (
    CancelledError,
    TripoDownloadError,
    TripoTaskFailedError,
)
from grounded_pipeline.domain.ports import CancelToken
from grounded_pipeline.texturing.provider import GuidanceMode, TextureRequest
from grounded_pipeline.texturing.tripo_provider import (
    REQUEST_BODY_KEY,
    TripoTextureProvider,
    build_texture_prompt,
)
from grounded_pipeline.tripo.client import TripoClient

from .conftest import (
    BASE_URL,
    GLB_BYTES,
    MODEL_URL,
    PNG_BYTES,
    PREVIEW_URL,
    TASK_ID,
    MakeClient,
    fixture_response,
    write_png,
)

CREDIT_TABLE = {
    "fast": 10.0,
    "standard": 10.0,
    "detailed": 20.0,
    "extreme": 30.0,
    "style_image": 2.0,
}
PROMPT = "a red ant"
FIXED_NOW = datetime(2026, 11, 2, 12, 0, tzinfo=UTC)
_FILENAME_RE = re.compile(rb'filename="([^"]+)"')


# --------------------------------------------------------------------------- helpers
def _upload_token(request: httpx.Request) -> httpx.Response:
    """Answer ``POST /files`` with a token derived from the multipart filename."""
    match = _FILENAME_RE.search(request.read())
    assert match is not None
    stem = Path(match.group(1).decode()).stem
    return httpx.Response(200, json={"code": 0, "data": {"file_token": f"file_{stem}"}})


def _glb_response() -> httpx.Response:
    return httpx.Response(200, content=GLB_BYTES, headers={"Content-Type": "model/gltf-binary"})


def _png_response() -> httpx.Response:
    return httpx.Response(200, content=PNG_BYTES, headers={"Content-Type": "image/png"})


def _forbidden() -> httpx.Response:
    return httpx.Response(403, text="expired")


class Routes:
    """The respx routes a texture run touches."""

    def __init__(self, router: respx.MockRouter) -> None:
        self.router = router
        self.upload = router.post(f"{BASE_URL}/files").mock(side_effect=_upload_token)
        self.create = router.post(f"{BASE_URL}/models/texture").mock(
            return_value=fixture_response("create_task")
        )
        self.task = router.get(f"{BASE_URL}/tasks/{TASK_ID}").mock(
            side_effect=[
                fixture_response("task_queued"),
                fixture_response("task_running"),
                fixture_response("task_success"),
            ]
        )
        self.download = router.get(url__eq=MODEL_URL).mock(return_value=_glb_response())
        self.preview = router.get(url__eq=PREVIEW_URL).mock(return_value=_png_response())


@pytest.fixture
def routes(router: respx.MockRouter) -> Routes:
    return Routes(router)


@pytest.fixture
def model_dir(tmp_path: Path) -> Path:
    folder = tmp_path / "models" / "ant"
    write_glb(folder / "work" / "static.glb", cube())
    write_png(folder / "references" / "style.png")
    for view in ("front", "left", "back", "right"):
        write_png(folder / "references" / f"{view}.png")
    return folder


def make_request(folder: Path, mode: GuidanceMode, **overrides: Any) -> TextureRequest:
    text_modes = (GuidanceMode.TEXT, GuidanceMode.TEXT_STYLE)
    guidance: dict[str, Path] = {}
    if mode is GuidanceMode.IMAGE:
        guidance = {"front": folder / "references" / "front.png"}
    elif mode is GuidanceMode.IMAGES:
        guidance = {
            "left": folder / "references" / "left.png",  # deliberately out of view order
            "front": folder / "references" / "front.png",
            "right": folder / "references" / "right.png",
        }
    fields: dict[str, Any] = {
        "model_id": "ant",
        "mesh_path": folder / "work" / "static.glb",
        "mode": mode,
        "prompt": PROMPT if mode in text_modes else None,
        "style_image": folder / "references" / "style.png"
        if mode is GuidanceMode.TEXT_STYLE
        else None,
        "guidance_images": guidance,
        "model_version": "v3.5-20260815",
        "quality": "standard",
        "pbr": True,
        "bake": True,
        "delight": False,
        "alignment": "original_image",
        "seed": 4242,
        "request_sha256": "ab" * 32,
    }
    fields.update(overrides)
    return TextureRequest.model_validate(fields)


def expected_body(texture_prompt: dict[str, Any], *, quality: str = "standard") -> dict[str, Any]:
    return {
        "input": "file_static",
        "model": "v3.5-20260815",
        "texture_prompt": texture_prompt,
        "pbr": True,
        "texture_seed": 4242,
        "texture_alignment": "original_image",
        "texture_quality": quality,
        "bake": True,
        "delight": False,
    }


MakeProvider = Callable[..., TripoTextureProvider]


@pytest.fixture
def make_provider(make_client: MakeClient) -> MakeProvider:
    def factory(client: TripoClient | None = None, **kwargs: Any) -> TripoTextureProvider:
        options: dict[str, Any] = {
            "credit_table": CREDIT_TABLE,
            "poll_interval_s": 2.0,
            "poll_timeout_s": 900.0,
            "now": lambda: FIXED_NOW,
        }
        options.update(kwargs)
        return TripoTextureProvider(client or make_client(), **options)

    return factory


def sent_body(route: respx.Route) -> dict[str, Any]:
    body: dict[str, Any] = json.loads(route.calls.last.request.read())
    return body


# --------------------------------------------------------------------------- guidance modes
def test_text_mode_sends_exact_body_and_downloads(
    routes: Routes, model_dir: Path, make_provider: MakeProvider
) -> None:
    provider = make_provider()
    req = make_request(model_dir, GuidanceMode.TEXT)
    res = provider.texture(req, resume_task_id=None, cancel=CancelToken())

    assert sent_body(routes.create) == expected_body({"text": PROMPT})
    assert routes.upload.call_count == 1
    assert res.provider == "tripo"
    assert res.task_id == TASK_ID
    assert res.result_glb == model_dir / "work" / "tripo" / "result.glb"
    assert res.result_glb.read_bytes() == GLB_BYTES
    assert res.preview_png == model_dir / "work" / "tripo" / "preview.png"
    assert res.preview_png.read_bytes() == PNG_BYTES
    assert res.credits_consumed == 10.0
    assert res.started_at == res.finished_at == FIXED_NOW.isoformat()
    assert res.raw_output["output"]["normalize_scale"] == 0.5
    assert res.raw_output["output"]["normalize_translation"] == [0.0, -0.25, 0.0]
    assert res.raw_output["status"] == "success"
    assert res.raw_output[REQUEST_BODY_KEY] == expected_body({"text": PROMPT})
    assert not (model_dir / "work" / "tripo" / "result.glb.tmp").exists()


def test_text_style_mode_uploads_style_image_as_object(
    routes: Routes, model_dir: Path, make_provider: MakeProvider
) -> None:
    provider = make_provider()
    provider.texture(
        make_request(model_dir, GuidanceMode.TEXT_STYLE), resume_task_id=None, cancel=CancelToken()
    )
    assert sent_body(routes.create)["texture_prompt"] == {
        "text": PROMPT,
        "style_image": {"file_token": "file_style"},
    }
    assert routes.upload.call_count == 2


def test_image_mode_sends_front_token_string(
    routes: Routes, model_dir: Path, make_provider: MakeProvider
) -> None:
    make_provider().texture(
        make_request(model_dir, GuidanceMode.IMAGE), resume_task_id=None, cancel=CancelToken()
    )
    assert sent_body(routes.create)["texture_prompt"] == {"image": "file_front"}
    assert routes.upload.call_count == 2


def test_images_mode_sends_view_key_array_in_order_without_missing_views(
    routes: Routes, model_dir: Path, make_provider: MakeProvider
) -> None:
    make_provider().texture(
        make_request(model_dir, GuidanceMode.IMAGES), resume_task_id=None, cancel=CancelToken()
    )
    assert sent_body(routes.create)["texture_prompt"] == {
        "images": [{"front": "file_front"}, {"left": "file_left"}, {"right": "file_right"}]
    }
    assert routes.upload.call_count == 4  # mesh + 3 views


@pytest.mark.parametrize(
    ("mode", "overrides", "message"),
    [
        (GuidanceMode.TEXT, {"prompt": None}, "requires a prompt"),
        (GuidanceMode.TEXT_STYLE, {"style_image": None}, "requires a style image"),
        (GuidanceMode.IMAGE, {"guidance_images": {}}, "front"),
        (GuidanceMode.IMAGES, {"guidance_images": {"left": Path("l.png")}}, "front"),
    ],
)
def test_build_texture_prompt_rejects_incomplete_requests(
    model_dir: Path, mode: GuidanceMode, overrides: dict[str, Any], message: str
) -> None:
    req = make_request(model_dir, mode, **overrides)
    with pytest.raises(ValueError, match=message):
        build_texture_prompt(req, lambda p: f"file_{p.stem}")


# --------------------------------------------------------------------------- resume
def test_resume_already_successful_task_makes_no_upload_or_create(
    router: respx.MockRouter, model_dir: Path, make_provider: MakeProvider
) -> None:
    upload = router.post(f"{BASE_URL}/files")
    create = router.post(f"{BASE_URL}/models/texture")
    task = router.get(f"{BASE_URL}/tasks/{TASK_ID}").mock(
        return_value=fixture_response("task_success")
    )
    router.get(url__eq=MODEL_URL).mock(return_value=_glb_response())
    router.get(url__eq=PREVIEW_URL).mock(return_value=_png_response())
    created: list[str] = []

    res = make_provider().texture(
        make_request(model_dir, GuidanceMode.TEXT),
        resume_task_id=TASK_ID,
        cancel=CancelToken(),
        on_task_created=created.append,
    )
    assert upload.call_count == 0
    assert create.call_count == 0
    assert task.call_count == 1
    assert created == [TASK_ID]
    assert res.task_id == TASK_ID
    assert res.raw_output[REQUEST_BODY_KEY] is None
    assert res.result_glb.read_bytes() == GLB_BYTES


def test_resume_running_task_polls_to_completion(
    router: respx.MockRouter, model_dir: Path, make_provider: MakeProvider
) -> None:
    events: list[str] = []

    def poll(_: httpx.Request) -> httpx.Response:
        events.append("poll")
        name = ["task_running", "task_running", "task_success"][min(events.count("poll") - 1, 2)]
        return fixture_response(name)

    router.post(f"{BASE_URL}/files")
    router.post(f"{BASE_URL}/models/texture")
    router.get(f"{BASE_URL}/tasks/{TASK_ID}").mock(side_effect=poll)
    router.get(url__eq=MODEL_URL).mock(return_value=_glb_response())
    router.get(url__eq=PREVIEW_URL).mock(return_value=_png_response())

    res = make_provider().texture(
        make_request(model_dir, GuidanceMode.TEXT),
        resume_task_id=TASK_ID,
        cancel=CancelToken(),
        on_task_created=lambda tid: events.append(f"created:{tid}"),
    )
    assert events[0] == f"created:{TASK_ID}"
    assert events.count("poll") == 3
    assert res.task_id == TASK_ID


def test_on_task_created_fires_before_first_poll(
    router: respx.MockRouter, model_dir: Path, make_provider: MakeProvider
) -> None:
    events: list[str] = []

    def create(_: httpx.Request) -> httpx.Response:
        events.append("create")
        return fixture_response("create_task")

    def poll(_: httpx.Request) -> httpx.Response:
        events.append("poll")
        return fixture_response("task_success")

    router.post(f"{BASE_URL}/files").mock(side_effect=_upload_token)
    router.post(f"{BASE_URL}/models/texture").mock(side_effect=create)
    router.get(f"{BASE_URL}/tasks/{TASK_ID}").mock(side_effect=poll)
    router.get(url__eq=MODEL_URL).mock(return_value=_glb_response())
    router.get(url__eq=PREVIEW_URL).mock(return_value=_png_response())

    make_provider().texture(
        make_request(model_dir, GuidanceMode.TEXT),
        resume_task_id=None,
        cancel=CancelToken(),
        on_task_created=lambda tid: events.append(f"created:{tid}"),
    )
    assert events == ["create", f"created:{TASK_ID}", "poll"]


def test_resume_without_callback_and_default_clock(
    router: respx.MockRouter, model_dir: Path, make_client: MakeClient
) -> None:
    router.get(f"{BASE_URL}/tasks/{TASK_ID}").mock(return_value=fixture_response("task_success"))
    router.get(url__eq=MODEL_URL).mock(return_value=_glb_response())
    router.get(url__eq=PREVIEW_URL).mock(return_value=_png_response())
    provider = TripoTextureProvider(make_client(), credit_table=CREDIT_TABLE)
    res = provider.texture(
        make_request(model_dir, GuidanceMode.TEXT), resume_task_id=TASK_ID, cancel=CancelToken()
    )
    started = datetime.fromisoformat(res.started_at)
    finished = datetime.fromisoformat(res.finished_at)
    assert started.tzinfo is not None
    assert finished >= started
    assert res.task_id == TASK_ID


# --------------------------------------------------------------------------- credits
def test_credits_fall_back_to_estimate_when_detail_has_none(
    router: respx.MockRouter, model_dir: Path, make_provider: MakeProvider
) -> None:
    router.post(f"{BASE_URL}/files").mock(side_effect=_upload_token)
    router.post(f"{BASE_URL}/models/texture").mock(return_value=fixture_response("create_task"))
    router.get(f"{BASE_URL}/tasks/{TASK_ID}").mock(
        return_value=fixture_response("task_success", credits_consumed=None)
    )
    router.get(url__eq=MODEL_URL).mock(return_value=_glb_response())
    router.get(url__eq=PREVIEW_URL).mock(return_value=_png_response())

    req = make_request(model_dir, GuidanceMode.TEXT_STYLE, quality="detailed")
    provider = make_provider()
    res = provider.texture(req, resume_task_id=None, cancel=CancelToken())
    assert res.credits_consumed == provider.estimate_credits(req) == 22.0


def test_estimate_credits_adds_style_surcharge_only_for_text_style(
    model_dir: Path, make_provider: MakeProvider
) -> None:
    provider = make_provider()
    assert provider.estimate_credits(make_request(model_dir, GuidanceMode.TEXT)) == 10.0
    assert provider.estimate_credits(make_request(model_dir, GuidanceMode.TEXT_STYLE)) == 12.0
    assert provider.estimate_credits(make_request(model_dir, GuidanceMode.IMAGES)) == 10.0
    extreme = make_request(model_dir, GuidanceMode.TEXT, quality="extreme")
    assert provider.estimate_credits(extreme) == 30.0


# --------------------------------------------------------------------------- downloads
def test_download_retries_once_with_fresh_url_after_403(
    routes: Routes, model_dir: Path, make_provider: MakeProvider
) -> None:
    routes.download.side_effect = [_forbidden(), _glb_response()]
    routes.task.side_effect = [
        fixture_response("task_success"),
        fixture_response("task_success"),  # re-fetched for a fresh URL
    ]
    res = make_provider().texture(
        make_request(model_dir, GuidanceMode.TEXT), resume_task_id=None, cancel=CancelToken()
    )
    assert routes.download.call_count == 2
    assert routes.task.call_count == 2
    assert res.result_glb.read_bytes() == GLB_BYTES


def test_download_gives_up_after_second_403(
    routes: Routes, model_dir: Path, make_provider: MakeProvider
) -> None:
    routes.download.side_effect = [_forbidden(), _forbidden()]
    routes.task.side_effect = [fixture_response("task_success"), fixture_response("task_success")]
    with pytest.raises(TripoDownloadError):
        make_provider().texture(
            make_request(model_dir, GuidanceMode.TEXT), resume_task_id=None, cancel=CancelToken()
        )
    assert routes.download.call_count == 2
    assert not (model_dir / "work" / "tripo" / "result.glb").exists()


def test_download_raises_when_refetched_detail_has_no_url(
    routes: Routes, model_dir: Path, make_provider: MakeProvider
) -> None:
    routes.download.side_effect = [_forbidden()]
    routes.task.side_effect = [
        fixture_response("task_success"),
        fixture_response("task_success", output={}),
    ]
    with pytest.raises(TripoDownloadError, match="download rejected"):
        make_provider().texture(
            make_request(model_dir, GuidanceMode.TEXT), resume_task_id=None, cancel=CancelToken()
        )


def test_success_without_model_url_is_a_download_error(
    routes: Routes, model_dir: Path, make_provider: MakeProvider
) -> None:
    routes.task.side_effect = [
        fixture_response("task_success", output={"rendered_image_url": PREVIEW_URL})
    ]
    with pytest.raises(TripoDownloadError, match=r"no output\.model_url"):
        make_provider().texture(
            make_request(model_dir, GuidanceMode.TEXT), resume_task_id=None, cancel=CancelToken()
        )


def test_preview_absent_gives_none(
    routes: Routes, model_dir: Path, make_provider: MakeProvider
) -> None:
    routes.task.side_effect = [fixture_response("task_success", output={"model_url": MODEL_URL})]
    res = make_provider().texture(
        make_request(model_dir, GuidanceMode.TEXT), resume_task_id=None, cancel=CancelToken()
    )
    assert res.preview_png is None
    assert routes.preview.call_count == 0


def test_preview_failure_degrades_to_none(
    routes: Routes, model_dir: Path, make_provider: MakeProvider
) -> None:
    routes.preview.side_effect = [_forbidden(), _forbidden()]
    routes.task.side_effect = [fixture_response("task_success"), fixture_response("task_success")]
    res = make_provider().texture(
        make_request(model_dir, GuidanceMode.TEXT), resume_task_id=None, cancel=CancelToken()
    )
    assert res.preview_png is None
    assert res.result_glb.exists()
    assert routes.preview.call_count == 2


def test_preview_retry_succeeds_with_fresh_url(
    routes: Routes, model_dir: Path, make_provider: MakeProvider
) -> None:
    routes.preview.side_effect = [_forbidden(), _png_response()]
    routes.task.side_effect = [fixture_response("task_success"), fixture_response("task_success")]
    res = make_provider().texture(
        make_request(model_dir, GuidanceMode.TEXT), resume_task_id=None, cancel=CancelToken()
    )
    assert res.preview_png is not None
    assert res.preview_png.read_bytes() == PNG_BYTES


def test_preview_refetch_without_url_gives_none(
    routes: Routes, model_dir: Path, make_provider: MakeProvider
) -> None:
    routes.preview.side_effect = [_forbidden()]
    routes.task.side_effect = [
        fixture_response("task_success"),
        fixture_response("task_success", output={"model_url": MODEL_URL}),
    ]
    res = make_provider().texture(
        make_request(model_dir, GuidanceMode.TEXT), resume_task_id=None, cancel=CancelToken()
    )
    assert res.preview_png is None


def test_work_dir_for_overrides_output_location(
    routes: Routes, model_dir: Path, make_provider: MakeProvider, tmp_path: Path
) -> None:
    out = tmp_path / "elsewhere"
    provider = make_provider(work_dir_for=lambda req: out / req.model_id)
    res = provider.texture(
        make_request(model_dir, GuidanceMode.TEXT), resume_task_id=None, cancel=CancelToken()
    )
    assert res.result_glb == out / "ant" / "result.glb"
    assert res.preview_png == out / "ant" / "preview.png"


# --------------------------------------------------------------------------- failures
def test_task_failure_propagates_after_task_id_was_announced(
    routes: Routes, model_dir: Path, make_provider: MakeProvider
) -> None:
    routes.task.side_effect = [fixture_response("task_failed")]
    created: list[str] = []
    with pytest.raises(TripoTaskFailedError):
        make_provider().texture(
            make_request(model_dir, GuidanceMode.TEXT),
            resume_task_id=None,
            cancel=CancelToken(),
            on_task_created=created.append,
        )
    assert created == [TASK_ID]
    assert routes.download.call_count == 0


def test_cancel_before_polling_raises_cancelled(
    routes: Routes, model_dir: Path, make_provider: MakeProvider
) -> None:
    cancel = CancelToken()
    cancel.set()
    with pytest.raises(CancelledError):
        make_provider().texture(
            make_request(model_dir, GuidanceMode.TEXT), resume_task_id=None, cancel=cancel
        )
    assert routes.create.call_count == 1
    assert routes.task.call_count == 0
