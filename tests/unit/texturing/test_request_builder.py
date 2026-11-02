"""``build_texture_request`` / ``texture_stage_fingerprint`` (plan 10 §1, §8)."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest

from fakes.glb_builder import cube
from grounded_pipeline.domain.errors import StagePrereqError
from grounded_pipeline.domain.model_spec import GuidanceImages, ResolvedModel, TextureOverrides
from grounded_pipeline.texturing.provider import GuidanceMode
from grounded_pipeline.texturing.request_builder import (
    PROVIDER_TAG,
    build_texture_request,
    guidance_mode,
    resolve_guidance_images,
    resolve_style_image,
    texture_stage_fingerprint,
)

from .conftest import DEFAULT_PROMPT, make_library, make_resolved, write_png

ALL_VIEWS = GuidanceImages(
    front=Path("references/front.png"),
    left=Path("references/left.png"),
    back=Path("references/back.png"),
    right=Path("references/right.png"),
)
FRONT_LEFT = GuidanceImages(front=Path("references/front.png"), left=Path("references/left.png"))


def _static(model: ResolvedModel) -> Path:
    return model.doc.folder / "work" / "static.glb"


def _write_views(model: ResolvedModel, images: GuidanceImages) -> None:
    for view in ("front", "left", "back", "right"):
        rel: Path | None = getattr(images, view)
        if rel is not None:
            write_png(model.doc.folder / rel, (ord(view[0]), 0, 0, 255))


def _build(model: ResolvedModel, root: Path) -> Any:
    return build_texture_request(model, static_glb=_static(model), library_root=root)


def _fingerprint(model: ResolvedModel, root: Path) -> str:
    return texture_stage_fingerprint(model, static_glb=_static(model), library_root=root)


# --------------------------------------------------------------------------- modes
def test_text_mode(library_root: Path) -> None:
    model = make_resolved(library_root, texture={"prompt_mode": "text"})
    req = _build(model, library_root)
    assert req.mode is GuidanceMode.TEXT
    assert req.prompt == DEFAULT_PROMPT
    assert req.style_image is None
    assert req.guidance_images == {}
    assert req.mesh_path == _static(model)
    assert req.model_id == "ant"
    assert (req.model_version, req.quality, req.seed) == ("v3.5-20260815", "standard", 1337)
    assert (req.pbr, req.bake, req.delight, req.alignment) == (True, True, True, "original_image")


def test_text_style_mode_uses_library_style_image(library_root: Path) -> None:
    model = make_resolved(library_root)  # default prompt_mode is text+style
    req = _build(model, library_root)
    assert req.mode is GuidanceMode.TEXT_STYLE
    assert req.prompt == DEFAULT_PROMPT
    assert req.style_image == library_root / "style" / "references" / "primary.png"
    assert req.guidance_images == {}


def test_image_mode_uses_only_front(library_root: Path) -> None:
    model = make_resolved(
        library_root,
        texture={"prompt_mode": "image"},
        front_texture=TextureOverrides(guidance_images=ALL_VIEWS),
    )
    _write_views(model, ALL_VIEWS)
    req = _build(model, library_root)
    assert req.mode is GuidanceMode.IMAGE
    assert req.prompt is None
    assert req.style_image is None
    assert req.guidance_images == {"front": model.doc.folder / "references" / "front.png"}


def test_images_mode_uses_present_views_only(library_root: Path) -> None:
    model = make_resolved(
        library_root,
        texture={"prompt_mode": "images"},
        front_texture=TextureOverrides(guidance_images=FRONT_LEFT),
    )
    _write_views(model, FRONT_LEFT)
    req = _build(model, library_root)
    assert req.mode is GuidanceMode.IMAGES
    assert req.prompt is None
    assert set(req.guidance_images) == {"front", "left"}
    assert req.guidance_images["left"] == model.doc.folder / "references" / "left.png"


def test_image_mode_without_guidance_images_is_a_prereq_error(library_root: Path) -> None:
    model = make_resolved(library_root, texture={"prompt_mode": "images"})
    with pytest.raises(StagePrereqError, match="guidance_images"):
        _build(model, library_root)


def test_resolve_guidance_images_empty_for_text_modes(library_root: Path) -> None:
    model = make_resolved(
        library_root,
        texture={"prompt_mode": "text"},
        front_texture=TextureOverrides(guidance_images=ALL_VIEWS),
    )
    assert guidance_mode(model) is GuidanceMode.TEXT
    assert resolve_guidance_images(model) == {}
    assert resolve_style_image(model, library_root=library_root) is None


# --------------------------------------------------------------------------- style precedence
def test_model_style_png_overrides_library_image(library_root: Path) -> None:
    model = make_resolved(library_root)
    override = write_png(model.doc.folder / "references" / "style.png", (0, 0, 250, 255))
    req = _build(model, library_root)
    assert req.style_image == override


def test_explicit_model_style_image_beats_implicit_override(library_root: Path) -> None:
    custom = Path("references/custom.png")
    model = make_resolved(
        library_root,
        texture={"style_image": str(custom)},
        front_texture=TextureOverrides(style_image=custom),
    )
    write_png(model.doc.folder / "references" / "style.png")
    expected = write_png(model.doc.folder / custom)
    req = _build(model, library_root)
    assert req.style_image == expected  # model-relative fallback, not references/style.png


def test_library_relative_wins_when_it_exists(library_root: Path) -> None:
    rel = Path("style/references/alt.png")
    model = make_resolved(library_root, texture={"style_image": str(rel)})
    expected = write_png(library_root / rel)
    assert _build(model, library_root).style_image == expected


def test_absolute_style_image_is_kept(library_root: Path, tmp_path: Path) -> None:
    absolute = write_png(tmp_path / "elsewhere" / "style.png")
    model = make_resolved(library_root, texture={"style_image": str(absolute)})
    assert _build(model, library_root).style_image == absolute


def test_missing_style_image_is_a_prereq_error(library_root: Path) -> None:
    model = make_resolved(library_root, texture={"style_image": "style/references/nope.png"})
    with pytest.raises(StagePrereqError, match="missing"):
        _build(model, library_root)


def test_missing_mesh_is_a_prereq_error(library_root: Path) -> None:
    model = make_resolved(library_root)
    with pytest.raises(StagePrereqError, match=r"static\.glb"):
        build_texture_request(
            model,
            static_glb=model.doc.folder / "work" / "static.glb.missing",
            library_root=library_root,
        )


# --------------------------------------------------------------------------- cache key
@pytest.mark.parametrize(
    "change",
    [
        {"prompt": "a blue ant"},
        {"texture": {"seed": 1}},
        {"texture": {"quality": "detailed"}},
        {"texture": {"pbr": False}},
        {"texture": {"delight": False}},
        {"texture": {"alignment": "geometry"}},
        {"texture": {"model_version": "v3.0-20250812"}},
        {"mesh": cube(scale=2.0)},
        {"model_id": "bee"},
    ],
    ids=lambda c: next(iter(c)) if "texture" not in c else next(iter(c["texture"])),
)
def test_request_sha_changes_with_inputs(tmp_path: Path, change: dict[str, Any]) -> None:
    root_a = make_library(tmp_path / "a")
    root_b = make_library(tmp_path / "b")
    base = _build(make_resolved(root_a), root_a).request_sha256
    changed = _build(make_resolved(root_b, **change), root_b).request_sha256
    assert base != changed


def test_request_sha_changes_with_style_image_bytes(tmp_path: Path) -> None:
    root_a = make_library(tmp_path / "a")
    root_b = make_library(tmp_path / "b")
    write_png(root_b / "style" / "references" / "primary.png", (99, 99, 99, 255))
    assert (
        _build(make_resolved(root_a), root_a).request_sha256
        != _build(make_resolved(root_b), root_b).request_sha256
    )


def test_request_sha_ignores_tripo_section_and_location(tmp_path: Path) -> None:
    root_a = make_library(tmp_path / "a")
    sha_a = _build(make_resolved(root_a), root_a).request_sha256
    root_b = tmp_path / "moved" / "b"
    shutil.copytree(root_a, root_b)
    model_b = make_resolved(root_b, tripo={"poll_interval_s": 9.0, "max_credits_per_run": 3})
    assert _build(model_b, root_b).request_sha256 == sha_a


def test_request_sha_is_deterministic(library_root: Path) -> None:
    model = make_resolved(library_root)
    assert _build(model, library_root) == _build(model, library_root)


# --------------------------------------------------------------------------- fingerprint
def test_fingerprint_is_stable_and_differs_from_request_sha(library_root: Path) -> None:
    model = make_resolved(library_root)
    fp = _fingerprint(model, library_root)
    assert fp == _fingerprint(model, library_root)
    assert fp != _build(model, library_root).request_sha256
    assert PROVIDER_TAG == "provider=tripo"


@pytest.mark.parametrize(
    "change",
    [
        {"prompt": "a blue ant"},
        {"texture": {"seed": 1}},
        {"texture": {"quality": "detailed"}},
        {"texture": {"prompt_mode": "text"}},
        {"mesh": cube(scale=2.0)},
    ],
    ids=["prompt", "seed", "quality", "mode", "mesh"],
)
def test_fingerprint_changes_with_inputs(tmp_path: Path, change: dict[str, Any]) -> None:
    root_a = make_library(tmp_path / "a")
    root_b = make_library(tmp_path / "b")
    assert _fingerprint(make_resolved(root_a), root_a) != _fingerprint(
        make_resolved(root_b, **change), root_b
    )


def test_fingerprint_changes_with_style_and_guidance_bytes(tmp_path: Path) -> None:
    root_a = make_library(tmp_path / "a")
    root_b = make_library(tmp_path / "b")
    write_png(root_b / "style" / "references" / "primary.png", (99, 99, 99, 255))
    assert _fingerprint(make_resolved(root_a), root_a) != _fingerprint(
        make_resolved(root_b), root_b
    )

    kwargs: dict[str, Any] = {
        "texture": {"prompt_mode": "images"},
        "front_texture": TextureOverrides(guidance_images=FRONT_LEFT),
    }
    model_a = make_resolved(root_a, **kwargs)
    model_b = make_resolved(root_b, **kwargs)
    _write_views(model_a, FRONT_LEFT)
    _write_views(model_b, FRONT_LEFT)
    assert _fingerprint(model_a, root_a) == _fingerprint(model_b, root_b)
    write_png(model_b.doc.folder / "references" / "left.png", (1, 1, 1, 255))
    assert _fingerprint(model_a, root_a) != _fingerprint(model_b, root_b)


def test_fingerprint_ignores_poll_interval_and_location(tmp_path: Path) -> None:
    root_a = make_library(tmp_path / "a")
    fp_a = _fingerprint(make_resolved(root_a), root_a)
    root_b = tmp_path / "moved" / "b"
    shutil.copytree(root_a, root_b)
    model_b = make_resolved(root_b, tripo={"poll_interval_s": 9.0, "poll_timeout_s": 60})
    assert _fingerprint(model_b, root_b) == fp_a
