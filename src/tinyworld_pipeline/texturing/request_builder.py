"""Pure construction of a :class:`TextureRequest` and the ``texture`` fingerprint (plan 10 §1).

Nothing here talks to a provider. The only I/O is reading the files the request references (mesh,
style image, guidance images) to hash them: the ``request_sha256`` cache key and the stage
fingerprint are both content-addressed so that moving a library does not invalidate anything.

Hashing is implemented locally (same algorithms as ``pipeline/hashing.py``: streamed SHA-256 of
files, canonical JSON of config sections, ``"\\x1f"``-joined parts) because the ``texturing``
layer must not import ``pipeline``. ``tests/unit/pipeline/test_texture_stage.py`` pins the two
implementations to each other.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path, PurePosixPath

from pydantic import BaseModel

from tinyworld_pipeline.domain.errors import StagePrereqError
from tinyworld_pipeline.domain.ids import Sha256
from tinyworld_pipeline.domain.model_spec import ResolvedModel
from tinyworld_pipeline.texturing.provider import GuidanceMode, TextureRequest

MODEL_STYLE_OVERRIDE = PurePosixPath("references/style.png")
"""A file at this model-relative path overrides the library style image (plan 05 §1)."""

PROVIDER_TAG = "provider=tripo"
"""Last fingerprint part: swapping providers must re-run the stage."""

VIEW_ORDER: tuple[str, ...] = ("front", "left", "back", "right")
"""Guidance views in the order Tripo lists them; ``front`` is mandatory."""

_CHUNK = 1 << 20
_SEPARATOR = "\x1f"
_IMAGE_MODES = frozenset({GuidanceMode.IMAGE, GuidanceMode.IMAGES})


# --------------------------------------------------------------------------- hashing helpers
def _file_sha(path: Path) -> Sha256:
    if not path.is_file():
        raise StagePrereqError(
            f"texture input file missing: {path}",
            detail={"path": str(path)},
            hint="Re-run from the `prepare` stage or restore the referenced reference image.",
        )
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(data: object) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _cfg_dump(section: BaseModel) -> str:
    return _canonical_json(section.model_dump(mode="json", exclude_none=True))


def _sha_many(parts: Iterable[str]) -> Sha256:
    return hashlib.sha256(_SEPARATOR.join(parts).encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- path resolution
def guidance_mode(model: ResolvedModel) -> GuidanceMode:
    """The effective :class:`GuidanceMode` (``texture.prompt_mode`` after layering)."""
    return GuidanceMode(model.config.texture.prompt_mode)


def resolve_style_image(model: ResolvedModel, *, library_root: Path) -> Path | None:
    """Absolute path of the style image, or ``None`` unless the mode is ``text+style``.

    Mirrors the library validation rule (plan 05 §1, 07 §1): when ``model.md`` did not set
    ``texture.style_image`` and ``<model>/references/style.png`` exists, that file wins;
    otherwise ``texture.style_image`` is taken relative to the library root, falling back to the
    model folder when only the model-relative file exists (validation already rejected the
    ambiguous case where both exist).

    Args:
        model: Resolved model (effective config + document).
        library_root: Absolute library root.

    Returns:
        The style image path (not checked for existence when neither candidate exists, so the
        hashing step reports the missing file), or ``None`` when no style image is used.
    """
    if guidance_mode(model) is not GuidanceMode.TEXT_STYLE:
        return None
    front = model.doc.front.texture
    model_set = front is not None and "style_image" in front.model_fields_set
    implicit = model.doc.folder / Path(MODEL_STYLE_OVERRIDE)
    if not model_set and implicit.is_file():
        return implicit
    rel = model.config.texture.style_image
    if rel.is_absolute():
        return rel
    from_library = library_root / rel
    from_model = model.doc.folder / rel
    if not from_library.is_file() and from_model.is_file():
        return from_model
    return from_library


def resolve_guidance_images(model: ResolvedModel) -> dict[str, Path]:
    """Absolute guidance image paths keyed by view, per the effective mode.

    ``image`` mode uses only ``front``; ``images`` mode uses every view that is set; text modes
    use none. Paths in ``model.md`` are relative to the model folder.

    Raises:
        StagePrereqError: An image mode without ``texture.guidance_images`` in ``model.md``.
    """
    mode = guidance_mode(model)
    if mode not in _IMAGE_MODES:
        return {}
    front = model.doc.front.texture
    images = front.guidance_images if front is not None else None
    if images is None:
        raise StagePrereqError(
            f"texture.prompt_mode is {mode.value!r} but model.md has no texture.guidance_images",
            detail={"model_id": model.doc.model_id, "prompt_mode": mode.value},
            hint="Add texture.guidance_images.front (and more views) or switch prompt_mode.",
        )
    views = VIEW_ORDER[:1] if mode is GuidanceMode.IMAGE else VIEW_ORDER
    folder = model.doc.folder
    out: dict[str, Path] = {}
    for view in views:
        rel: Path | None = getattr(images, view)
        if rel is not None:
            out[view] = rel if rel.is_absolute() else folder / rel
    return out


# --------------------------------------------------------------------------- public builders
def build_texture_request(
    model: ResolvedModel, *, static_glb: Path, library_root: Path
) -> TextureRequest:
    """Assemble the provider-neutral request for one model (pure apart from hashing inputs).

    Args:
        model: Resolved model (output of ``discover``).
        static_glb: Absolute path of ``work/static.glb`` produced by ``prepare``.
        library_root: Absolute library root (``texture.style_image`` is relative to it).

    Returns:
        A :class:`TextureRequest` with absolute paths and a ``request_sha256`` cache key computed
        over every field with each path replaced by the SHA-256 of the file it points to.

    Raises:
        StagePrereqError: A referenced file (mesh, style image, guidance image) is missing, or an
            image mode has no guidance images configured.
    """
    texture = model.config.texture
    mode = guidance_mode(model)
    prompt = None if mode in _IMAGE_MODES else model.composed_prompt.text
    style_image = resolve_style_image(model, library_root=library_root)
    guidance = resolve_guidance_images(model)
    fields: dict[str, object] = {
        "model_id": model.doc.model_id,
        "mesh_path": static_glb,
        "mode": mode,
        "prompt": prompt,
        "style_image": style_image,
        "guidance_images": guidance,
        "model_version": texture.model_version,
        "quality": texture.quality,
        "pbr": texture.pbr,
        "bake": texture.bake,
        "delight": texture.delight,
        "alignment": texture.alignment,
        "seed": texture.seed,
    }
    hashed = {
        **fields,
        "mode": mode.value,
        "mesh_path": _file_sha(static_glb),
        "style_image": None if style_image is None else _file_sha(style_image),
        "guidance_images": {view: _file_sha(path) for view, path in guidance.items()},
    }
    sha = hashlib.sha256(_canonical_json(hashed).encode("utf-8")).hexdigest()
    return TextureRequest.model_validate({**fields, "request_sha256": sha})


def texture_stage_fingerprint(
    model: ResolvedModel, *, static_glb: Path, library_root: Path
) -> Sha256:
    """The ``texture`` stage input fingerprint (plan 10 §1, normative composition).

    ``sha_many(file_sha(static.glb), composed_prompt.sha256, file_sha(style_image) if used,
    *file_sha(guidance images in view order), cfg_dump(config.texture), "provider=tripo")``.
    Poll intervals, budgets and other ``[tripo]`` settings are deliberately excluded.

    Raises:
        StagePrereqError: A referenced file is missing.
    """
    parts: list[str] = [_file_sha(static_glb), model.composed_prompt.sha256]
    style_image = resolve_style_image(model, library_root=library_root)
    if style_image is not None:
        parts.append(_file_sha(style_image))
    guidance = resolve_guidance_images(model)
    parts.extend(_file_sha(guidance[view]) for view in VIEW_ORDER if view in guidance)
    parts.append(_cfg_dump(model.config.texture))
    parts.append(PROVIDER_TAG)
    return _sha_many(parts)


__all__ = [
    "MODEL_STYLE_OVERRIDE",
    "PROVIDER_TAG",
    "VIEW_ORDER",
    "build_texture_request",
    "guidance_mode",
    "resolve_guidance_images",
    "resolve_style_image",
    "texture_stage_fingerprint",
]
