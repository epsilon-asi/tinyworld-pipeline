"""``gp validate`` rules — every code in plan 05 §8.

Decision helpers are pure (they take already-inspected facts); the only I/O is reading image
headers and testing file existence. Problems are returned, never raised, except by
:func:`raise_if_errors`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from PIL import Image, UnidentifiedImageError

from tinyworld_pipeline.config.layering import CliOverrides, resolve_config
from tinyworld_pipeline.domain.errors import ConfigError, Problem, UsageError, ValidationError
from tinyworld_pipeline.domain.ids import MODEL_ID_PATTERN, is_valid_model_id
from tinyworld_pipeline.domain.library_config import LibraryConfig
from tinyworld_pipeline.domain.model_spec import GuidanceImages, ModelDocument
from tinyworld_pipeline.domain.style_spec import BASE_PROMPT_SECTION
from tinyworld_pipeline.library.discovery import Library, find_animation_fbx, find_primary_fbx
from tinyworld_pipeline.library.frontmatter_io import E_FRONTMATTER
from tinyworld_pipeline.texturing.prompt_builder import MODEL_PROMPT_SECTION, compose_with_warnings

E_MODEL_ID = "E_MODEL_ID"
E_MODEL_MD_MISSING = "E_MODEL_MD_MISSING"
E_PROMPT_MISSING = "E_PROMPT_MISSING"
E_STYLE_IMAGE_MISSING = "E_STYLE_IMAGE_MISSING"
E_GUIDANCE_IMAGES = "E_GUIDANCE_IMAGES"
E_REF_IMAGE_FORMAT = "E_REF_IMAGE_FORMAT"
W_LICENSE_MISSING = "W_LICENSE_MISSING"
E_UNITY_DIR = "E_UNITY_DIR"
E_PATH_AMBIGUOUS = "E_PATH_AMBIGUOUS"
W_GUIDANCE_UNVERIFIED = "W_GUIDANCE_UNVERIFIED"
E_STYLE_BASE_PROMPT_MISSING = "E_STYLE_BASE_PROMPT_MISSING"

REF_IMAGE_MIN_PX = 256
REF_IMAGE_MAX_BYTES = 20 * 1024 * 1024
MIN_GUIDANCE_VIEWS = 2
MODEL_STYLE_OVERRIDE = Path("references/style.png")
TEXT_MODES: frozenset[str] = frozenset({"text", "text+style"})
IMAGE_MODES: frozenset[str] = frozenset({"image", "images"})

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
_JPEG_MAGIC = b"\xff\xd8\xff"

BlenderCheck = Callable[[LibraryConfig], list[Problem]]
"""Injected Blender presence/version check (``blender/locate.py``, WP-03); ``None`` skips it."""


# --------------------------------------------------------------------------- images
@dataclass(frozen=True, slots=True)
class ImageInfo:
    """Facts about a reference image file, gathered by :func:`inspect_image`."""

    exists: bool
    format: Literal["png", "jpeg"] | None
    width: int
    height: int
    size_bytes: int


def inspect_image(path: Path) -> ImageInfo:
    """Read magic bytes, pixel size (header only, via Pillow) and file size of ``path``."""
    if not path.is_file():
        return ImageInfo(False, None, 0, 0, 0)
    size = path.stat().st_size
    with path.open("rb") as fh:
        head = fh.read(len(_PNG_MAGIC))
    fmt: Literal["png", "jpeg"] | None = None
    if head.startswith(_PNG_MAGIC):
        fmt = "png"
    elif head.startswith(_JPEG_MAGIC):
        fmt = "jpeg"
    if fmt is None:
        return ImageInfo(True, None, 0, 0, size)
    try:
        with Image.open(path) as im:
            width, height = im.size
    except (UnidentifiedImageError, OSError):
        return ImageInfo(True, None, 0, 0, size)
    return ImageInfo(True, fmt, width, height, size)


def image_format_problems(info: ImageInfo, *, model_id: str | None, path: str) -> list[Problem]:
    """``E_REF_IMAGE_FORMAT`` findings for an existing image (pure)."""
    if info.format is None:
        msg, hint = "reference image is not a PNG or JPEG file", "re-export as PNG or JPEG"
    elif min(info.width, info.height) < REF_IMAGE_MIN_PX:
        msg = f"reference image is {info.width}x{info.height} px (minimum {REF_IMAGE_MIN_PX})"
        hint = "use a larger image"
    elif info.size_bytes > REF_IMAGE_MAX_BYTES:
        msg = f"reference image is {info.size_bytes} bytes (max {REF_IMAGE_MAX_BYTES})"
        hint = "downscale or re-encode the image"
    else:
        return []
    return [
        Problem(
            code=E_REF_IMAGE_FORMAT,
            severity="error",
            model_id=model_id,
            path=path,
            message=msg,
            hint=hint,
        )
    ]


# --------------------------------------------------------------------------- style image
def resolve_style_image(
    lib: Library, doc: ModelDocument, config: LibraryConfig
) -> tuple[Path | None, list[Problem]]:
    """Locate the effective style image (plan 05 §1, 07 §1 path rule).

    ``texture.style_image`` is library-root-relative; ``model.md`` may instead name a
    model-relative path, detected when the file exists under the model folder and not under the
    library root (both → ``E_PATH_AMBIGUOUS``). When the model did not set ``texture.style_image``
    and ``references/style.png`` exists in the model folder, it overrides the library image.

    Returns:
        ``(absolute path or None, problems)`` — ``E_PATH_AMBIGUOUS`` or ``E_STYLE_IMAGE_MISSING``.
    """
    texture_front = doc.front.texture
    model_set = texture_front is not None and "style_image" in texture_front.model_fields_set
    implicit = doc.folder / MODEL_STYLE_OVERRIDE
    if not model_set and implicit.is_file():
        return implicit.resolve(), []
    rel = config.texture.style_image
    candidates = [rel] if rel.is_absolute() else [lib.paths.root / rel, doc.folder / rel]
    existing = [c.resolve() for c in candidates if c.is_file()]
    existing = sorted(set(existing))
    label = "model.md:texture.style_image" if model_set else "library.toml:texture.style_image"
    if len(existing) > 1:
        return None, [
            Problem(
                code=E_PATH_AMBIGUOUS,
                severity="error",
                model_id=doc.model_id,
                path=label,
                message=(
                    f"{rel.as_posix()!r} exists under both the library root and the model folder"
                ),
                hint="rename one of the files or use an absolute path",
            )
        ]
    if existing:
        return existing[0], []
    return None, [
        Problem(
            code=E_STYLE_IMAGE_MISSING,
            severity="error",
            model_id=doc.model_id,
            path=label,
            message=f"style image {rel.as_posix()!r} not found (prompt_mode is text+style)",
            hint="add the image or switch texture.prompt_mode to 'text'",
        )
    ]


# --------------------------------------------------------------------------- per-model checks
def _check_license(doc: ModelDocument) -> list[Problem]:
    if doc.front.source.license:
        return []
    source_dir = doc.folder / "source"
    has_file = source_dir.is_dir() and any(
        p.is_file() and p.name.lower().startswith("license") for p in source_dir.iterdir()
    )
    if has_file:
        return []
    return [
        Problem(
            code=W_LICENSE_MISSING,
            severity="warning",
            model_id=doc.model_id,
            path="model.md:source.license",
            message="no source.license and no source/LICENSE* file",
            hint="record the pack licence, e.g. source.license: 'CC0 — Quaternius'",
        )
    ]


def _check_prompt(lib: Library, doc: ModelDocument, config: LibraryConfig) -> list[Problem]:
    problems: list[Problem] = []
    mode = config.texture.prompt_mode
    if mode in TEXT_MODES and not doc.sections.get(MODEL_PROMPT_SECTION):
        problems.append(
            Problem(
                code=E_PROMPT_MISSING,
                severity="error",
                model_id=doc.model_id,
                path="model.md:## prompt",
                message=f"prompt_mode {mode!r} requires a non-empty '## prompt' section",
                hint="add a '## prompt' section to model.md",
            )
        )
    if lib.style is None:
        return problems
    try:
        _composed, warnings = compose_with_warnings(lib.style, doc, doc.front.prompt)
    except ValidationError as exc:  # includes PromptTooLongError
        problems.extend(exc.problems)
        return problems
    problems.extend(warnings)
    return problems


def _check_style_image(lib: Library, doc: ModelDocument, config: LibraryConfig) -> list[Problem]:
    if config.texture.prompt_mode != "text+style":
        return []
    path, problems = resolve_style_image(lib, doc, config)
    if path is None:
        return problems
    label = f"texture.style_image ({path.name})"
    return image_format_problems(inspect_image(path), model_id=doc.model_id, path=label)


def _guidance_problem(doc: ModelDocument, path: str, message: str, hint: str) -> Problem:
    return Problem(
        code=E_GUIDANCE_IMAGES,
        severity="error",
        model_id=doc.model_id,
        path=path,
        message=message,
        hint=hint,
    )


def _check_guidance_files(doc: ModelDocument, images: GuidanceImages) -> list[Problem]:
    problems: list[Problem] = []
    for view, rel in images.model_dump().items():
        if rel is None:
            continue
        label = f"model.md:texture.guidance_images.{view}"
        info = inspect_image(doc.folder / rel)
        if not info.exists:
            problems.append(
                _guidance_problem(doc, label, f"{rel} not found", "paths are model-folder relative")
            )
        elif info.size_bytes > REF_IMAGE_MAX_BYTES:
            problems.append(
                _guidance_problem(
                    doc, label, f"{rel} exceeds {REF_IMAGE_MAX_BYTES} bytes", "downscale the image"
                )
            )
        else:
            problems.extend(image_format_problems(info, model_id=doc.model_id, path=label))
    return problems


def _check_guidance(doc: ModelDocument, config: LibraryConfig) -> list[Problem]:
    mode = config.texture.prompt_mode
    if mode not in IMAGE_MODES:
        return []
    problems = [
        Problem(
            code=W_GUIDANCE_UNVERIFIED,
            severity="warning",
            model_id=doc.model_id,
            path="model.md:texture.prompt_mode",
            message=f"prompt_mode {mode!r} is not yet verified against Tripo (spike S1, plan 18)",
            hint="prefer 'text+style' until S1 confirms image guidance on uploaded models",
        )
    ]
    images = doc.front.texture.guidance_images if doc.front.texture is not None else None
    key = "model.md:texture.guidance_images"
    if images is None:
        problems.append(
            _guidance_problem(
                doc, key, f"prompt_mode {mode!r} requires texture.guidance_images.front", "add it"
            )
        )
        return problems
    views = sum(1 for v in images.model_dump().values() if v is not None)
    if mode == "images" and views < MIN_GUIDANCE_VIEWS:
        problems.append(
            _guidance_problem(
                doc, key, f"'images' mode needs at least {MIN_GUIDANCE_VIEWS} views", "add a view"
            )
        )
    problems.extend(_check_guidance_files(doc, images))
    return problems


def _check_source(doc: ModelDocument) -> list[Problem]:
    primary, problems = find_primary_fbx(doc)
    _anims, anim_problems = find_animation_fbx(doc, primary=primary)
    return [*problems, *anim_problems]


def _model_id_problem(model_id: str) -> Problem:
    return Problem(
        code=E_MODEL_ID,
        severity="error",
        model_id=None,
        path=f"models/{model_id}/",
        message=f"folder name {model_id!r} does not match {MODEL_ID_PATTERN}",
        hint="use lower-case letters, digits and underscores, 2-64 characters",
    )


def _load_model(
    lib: Library, model_id: str, env: Mapping[str, str], cli: CliOverrides
) -> tuple[ModelDocument, LibraryConfig, list[Problem]] | list[Problem]:
    """Parse ``model.md`` and layer its config; a list means "stop here, these are the problems"."""
    if not is_valid_model_id(model_id):
        return [_model_id_problem(model_id)]
    try:
        doc = lib.model_document(model_id)
    except ValidationError as exc:
        return exc.problems
    try:
        config, warnings = resolve_config(lib.config, doc.front, env, cli)
    except ConfigError as exc:
        return [
            Problem(
                code=E_FRONTMATTER,
                severity="error",
                model_id=model_id,
                path="model.md",
                message=f"effective configuration is invalid: {exc.message}",
                hint=exc.hint,
            )
        ]
    return doc, config, warnings


def validate_model(
    lib: Library,
    model_id: str,
    *,
    env: Mapping[str, str],
    cli: CliOverrides,
) -> list[Problem]:
    """Run every per-model rule of plan 05 §8 for one model folder.

    Args:
        lib: Loaded library.
        model_id: Folder name under ``models/`` (may be invalid → ``E_MODEL_ID``).
        env: Process environment (``GP_*`` config overrides, plan 07 §1).
        cli: CLI overrides (plan 08 §3).

    Returns:
        All findings (errors and warnings) for this model, deterministic order.
    """
    loaded = _load_model(lib, model_id, env, cli)
    if isinstance(loaded, list):
        return loaded
    doc, config, problems = loaded
    problems = list(problems)
    problems.extend(_check_source(doc))
    problems.extend(_check_license(doc))
    problems.extend(_check_prompt(lib, doc, config))
    problems.extend(_check_style_image(lib, doc, config))
    problems.extend(_check_guidance(doc, config))
    return problems


# --------------------------------------------------------------------------- library-level checks
def validate_library_level(
    lib: Library, *, blender_check: BlenderCheck | None = None
) -> list[Problem]:
    """Rules that do not depend on a model: ``library.toml``, ``style.md``, Unity dir, Blender."""
    problems = list(lib.load_problems)
    if lib.style is not None and not lib.style.sections.get(BASE_PROMPT_SECTION):
        problems.append(
            Problem(
                code=E_STYLE_BASE_PROMPT_MISSING,
                severity="error",
                path="style/style.md:## base_prompt",
                message="style.md has no non-empty '## base_prompt' section",
                hint="add a '## base_prompt' section describing the shared look",
            )
        )
    unity_dir = lib.config.library.unity_import_dir
    if unity_dir is not None and not unity_dir.expanduser().is_dir():
        problems.append(
            Problem(
                code=E_UNITY_DIR,
                severity="error",
                path="library.toml:library.unity_import_dir",
                message=f"unity_import_dir {unity_dir.as_posix()!r} is not an existing directory",
                hint="create it, fix the path, or leave the key empty to disable Unity sync",
            )
        )
    if blender_check is not None:
        problems.extend(blender_check(lib.config))
    return problems


def validate_library(
    lib: Library,
    model_ids: list[str] | None = None,
    *,
    env: Mapping[str, str],
    cli: CliOverrides,
    blender_check: BlenderCheck | None = None,
) -> list[Problem]:
    """Validate the library and every (or the selected) model (plan 05 §8, 08 §2 ``gp validate``).

    Args:
        lib: Loaded library.
        model_ids: Restrict to these ids (unknown id → :class:`UsageError`); ``None`` = all.
        env: Process environment (``GP_*`` config overrides).
        cli: CLI overrides.
        blender_check: Optional injected Blender presence/version check.

    Returns:
        Library-level problems first, then per-model problems in id order.

    Raises:
        UsageError: ``model_ids`` names a folder that does not exist.
    """
    problems = validate_library_level(lib, blender_check=blender_check)
    for model_id in lib.select(ids=model_ids):
        problems.extend(validate_model(lib, model_id, env=env, cli=cli))
    return problems


def raise_if_errors(problems: list[Problem]) -> None:
    """Raise :class:`ValidationError` (carrying *all* problems) if any has severity ``error``."""
    if any(p.severity == "error" for p in problems):
        raise ValidationError(problems)


__all__ = [
    "BlenderCheck",
    "ImageInfo",
    "UsageError",
    "image_format_problems",
    "inspect_image",
    "raise_if_errors",
    "resolve_style_image",
    "validate_library",
    "validate_library_level",
    "validate_model",
]
