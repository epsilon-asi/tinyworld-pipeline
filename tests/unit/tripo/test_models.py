from __future__ import annotations

import pytest
from pydantic import ValidationError

from tinyworld_pipeline.tripo.models import (
    TERMINAL_FAILURE_STATUSES,
    ApiEnvelope,
    BalanceData,
    TaskCreateData,
    TaskDetail,
    TexturePrompt,
    TextureTaskRequest,
    TripoFileRef,
    envelope_of,
)


def test_file_ref_needs_exactly_one_of_token_or_url() -> None:
    assert TripoFileRef(file_token="f").model_dump(exclude_none=True) == {"file_token": "f"}
    assert TripoFileRef(url="https://x/y.png").url == "https://x/y.png"
    with pytest.raises(ValidationError):
        TripoFileRef()
    with pytest.raises(ValidationError):
        TripoFileRef(file_token="f", url="u")


def test_text_with_style_image_dumps_in_official_shape() -> None:
    prompt = TexturePrompt(text="cyberpunk", style_image=TripoFileRef(file_token="file_1"))
    assert prompt.model_dump(exclude_none=True) == {
        "text": "cyberpunk",
        "style_image": {"file_token": "file_1"},
    }


def test_images_view_key_array_rules() -> None:
    ok = TexturePrompt(images=[{"front": "f"}, {"left": "l"}])
    assert ok.model_dump(exclude_none=True) == {"images": [{"front": "f"}, {"left": "l"}]}

    with pytest.raises(ValidationError, match="front"):
        TexturePrompt(images=[{"left": "l"}, {"back": "b"}])
    with pytest.raises(ValidationError, match="two views"):
        TexturePrompt(images=[{"front": "f"}])
    with pytest.raises(ValidationError, match="exactly one view key"):
        TexturePrompt(images=[{"front": "f", "left": "l"}])
    with pytest.raises(ValidationError, match="duplicate"):
        TexturePrompt(images=[{"front": "f"}, {"front": "g"}])
    with pytest.raises(ValidationError):
        TexturePrompt(images=[{"front": "f"}, {"top": "t"}])


def test_text_image_images_are_mutually_exclusive_and_style_needs_text() -> None:
    with pytest.raises(ValidationError, match="exactly one"):
        TexturePrompt(text="a", image="b")
    with pytest.raises(ValidationError, match="exactly one"):
        TexturePrompt()
    with pytest.raises(ValidationError, match="style_image"):
        TexturePrompt(image="b", style_image=TripoFileRef(url="u"))


def test_texture_task_request_body_omits_none_and_never_sets_compress() -> None:
    body = TextureTaskRequest(
        input="file_abc",
        model="v3.5-20260815",
        texture_prompt=TexturePrompt(text="ant"),
        texture_seed=1337,
        texture_quality="detailed",
    ).model_dump(exclude_none=True)

    assert body == {
        "input": "file_abc",
        "model": "v3.5-20260815",
        "texture_prompt": {"text": "ant"},
        "pbr": True,
        "texture_seed": 1337,
        "texture_alignment": "original_image",
        "texture_quality": "detailed",
        "bake": True,
        "delight": True,
    }
    assert "compress" not in body


def test_envelope_success_and_failure() -> None:
    ok = envelope_of(TaskCreateData).model_validate({"code": 0, "data": {"task_id": "task_1"}})
    assert ok.ok
    assert ok.data is not None
    assert ok.data.task_id == "task_1"

    bad = ApiEnvelope[BalanceData].model_validate(
        {"code": 2000, "status": "error", "message": "bad", "suggestion": "fix", "request_id": "r"}
    )
    assert not bad.ok
    assert bad.data is None
    assert bad.suggestion == "fix"


def test_task_detail_statuses_and_normalisation_fields() -> None:
    detail = TaskDetail.model_validate(
        {
            "task_id": "task_1",
            "type": "texture_model",
            "status": "success",
            "progress": 100,
            "output": {
                "model_url": "https://cdn/x.glb",
                "normalize_scale": 0.5,
                "normalize_translation": [0, 1, 2],
                "future_field": "ignored but kept",
            },
            "credits_consumed": 10.0,
            "created_at": "2026-11-02T00:00:00Z",
        }
    )
    assert detail.is_success
    assert not detail.is_terminal_failure
    assert detail.output is not None
    assert detail.output.normalize_scale == 0.5
    assert detail.output.normalize_translation == (0.0, 1.0, 2.0)

    for status in TERMINAL_FAILURE_STATUSES:
        failed = TaskDetail(task_id="t", type="texture_model", status=status)
        assert failed.is_terminal_failure
    assert {"failed", "cancelled", "banned", "expired"} == TERMINAL_FAILURE_STATUSES
    assert not TaskDetail(task_id="t", type="x", status="unknown").is_terminal_failure
    with pytest.raises(ValidationError):
        TaskDetail(task_id="t", type="x", status="done")
