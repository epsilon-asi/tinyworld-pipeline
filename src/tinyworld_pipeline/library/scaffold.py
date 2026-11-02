"""Scaffolding for ``gp init`` and ``gp new-model`` (plan 08 §2, 05 §3-§5).

``library.toml`` is generated from the pydantic defaults (never hand-written) so it can never drift
from :class:`LibraryConfig`; keys whose default is ``None`` (TOML has no null) are emitted as
commented-out lines.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Literal, get_args, get_origin

import frontmatter
import tomli_w
from pydantic import BaseModel

from tinyworld_pipeline.domain.errors import UsageError
from tinyworld_pipeline.domain.ids import MODEL_ID_PATTERN, is_valid_model_id
from tinyworld_pipeline.domain.library_config import SECTION_MODELS, LibraryConfig
from tinyworld_pipeline.library.paths import LibraryPaths, ModelPaths

CopyMode = Literal["copy", "move", "link"]

GITIGNORE_TEMPLATE = """\
# tinyworld_pipeline: disposable intermediates, state and deliverables
work/
.pipeline/
output/
"""

STYLE_MD_TEMPLATE = """\
---
schema_version: 1
name: {name}
primary_reference: references/primary.png
negative_terms: ["outline", "cel shading", "pixel art", "photorealistic"]
---

## base_prompt
Stylized hand-painted game asset, smooth soft shading, saturated vibrant colours,
subtle painted gradients, clean readable shapes, no outlines, PBR materials.

## material_notes
Describe how materials should read (gloss, matte, wet, metallic) — optional.

## palette
List hex colours with names — optional; only used when a model's prompt.include_sections
lists `palette`.
"""

MODEL_MD_BODY_TEMPLATE = """\
## prompt
Describe {display_name}: dominant colours, materials, distinctive details, how light should
read on its surfaces. Keep it specific; the shared style comes from style/style.md.

## notes
Created by `gp new-model` from {source_name}. Record licence, quirks and decisions here.
"""

#: Short descriptions written next to each key (plan 05 §3). Missing keys get a type-only comment.
KEY_COMMENTS: dict[str, str] = {
    "library.name": "human-readable library name",
    "library.unity_import_dir": 'e.g. "D:/Game/Assets/Art/Creatures"; unset = no Unity sync',
    "library.concurrency": "models processed in parallel",
    "library.blender_parallelism": "simultaneous Blender processes",
    "blender.executable": "unset = auto-detect (GP_BLENDER__EXECUTABLE env wins)",
    "blender.allow_unsupported": "permit Blender versions outside 4.2/4.5 LTS",
    "blender.job_timeout_s": "per Blender job",
    "blender.keep_blend_files": "keep work/prepared.blend after success",
    "geometry.strategy": "v1: only subdivide",
    "geometry.subdivision_levels": "0 = geometry untouched (texture-only run)",
    "geometry.shapekeys": "skip = leave that mesh unsubdivided",
    "geometry.merge_material_slots": "single material slot -> single texture set",
    "geometry.include_unskinned": "export unskinned meshes as-is (never subdivided)",
    "geometry.max_triangles": "abort prepare if the subdivided result would exceed this",
    "uv.palette_texture_max_px": "source texture <= this -> treated as palette -> unwrap",
    "uv.overlap_ratio_threshold": "UV overlap above this -> unwrap",
    "texture.bake": "Tripo-side bake flag (material effects -> base textures)",
    "texture.alignment": "only matters with image guidance",
    "texture.seed": "texture_seed; per-model override allowed",
    "texture.style_image": "relative to the library root",
    "verify.position_quantum": "model units",
    "verify.allow_geometry_change": "if Tripo remeshed: false = fail, true = bake anyway",
    "bake.cage_extrusion_ratio": "x bounding-box diagonal",
    "unity.source_asset_root": 'Unity path of the vendor import, e.g. "Assets/ThirdParty/Pack"',
    "tripo.max_credits_per_run": "0 = unlimited",
    "tripo.credit_table": "estimates only; real credits_consumed is recorded per task",
}


# --------------------------------------------------------------------------- library.toml
def _literal_choices(annotation: Any) -> list[str]:  # noqa: ANN401  # pydantic annotation
    if get_origin(annotation) is Literal:
        return [repr(a) if not isinstance(a, str) else f'"{a}"' for a in get_args(annotation)]
    for arg in get_args(annotation):
        found = _literal_choices(arg)
        if found:
            return found
    return []


def _comment(section: str, key: str, model: type[BaseModel]) -> str:
    parts = []
    text = KEY_COMMENTS.get(f"{section}.{key}")
    if text:
        parts.append(text)
    annotation = model.model_fields[key].annotation
    choices = _literal_choices(annotation)
    if len(choices) > 1:
        label = "any of" if get_origin(annotation) is list else "one of"
        parts.append(f"{label}: " + " | ".join(choices))
    return "  # " + "; ".join(parts) if parts else ""


def _scalar_line(section: str, key: str, value: Any, model: type[BaseModel]) -> str:  # noqa: ANN401
    comment = _comment(section, key, model)
    if value is None:
        return f'# {key} = ""{comment or "  # unset"}'
    return tomli_w.dumps({key: value}).rstrip("\n") + comment


def render_library_toml(config: LibraryConfig | None = None) -> str:
    """Render a fully commented ``library.toml`` with every key of ``config`` (defaults if None).

    Scalar keys are emitted in model-field order; nested tables (``[tripo.credit_table]``) follow
    their section. ``None`` values become commented-out lines so the file round-trips through
    the schema unchanged.
    """
    cfg = config or LibraryConfig()
    lines = [
        "# tinyworld_pipeline library configuration (generated by `gp init`; plan 05 §3).",
        "# Every [section].key can be overridden per model in model.md front matter,",
        "# by GP_<SECTION>__<KEY> environment variables, and by CLI flags.",
        "",
        f"schema_version = {cfg.schema_version}",
    ]
    for section, model in SECTION_MODELS.items():
        values: dict[str, Any] = getattr(cfg, section).model_dump(mode="json")
        lines += ["", f"[{section}]"]
        tables = {k: v for k, v in values.items() if isinstance(v, dict)}
        for key, value in values.items():
            if key not in tables:
                lines.append(_scalar_line(section, key, value, model))
        for key, table in tables.items():
            comment = _comment(section, key, model)
            lines += ["", f"[{section}.{key}]{comment}"]
            lines.append(tomli_w.dumps(table).rstrip("\n"))
    return "\n".join(lines) + "\n"


def init_library(path: Path, *, force: bool = False) -> LibraryPaths:
    """Scaffold a new asset library at ``path`` (``gp init``).

    Creates ``library.toml`` (all defaults, commented), ``style/style.md``,
    ``style/references/.gitkeep``, ``models/.gitkeep``, ``generative/.gitkeep`` and a
    ``.gitignore``. Existing template files other than ``library.toml`` are left untouched.

    Args:
        path: Library root (created if missing).
        force: Overwrite an existing ``library.toml`` (only the toml is overwritten).

    Returns:
        The library paths.

    Raises:
        UsageError: ``library.toml`` already exists and ``force`` is false (exit 2).
    """
    paths = LibraryPaths.from_root(path)
    if paths.library_toml.exists() and not force:
        raise UsageError(
            f"{paths.library_toml} already exists",
            hint="pass --force to overwrite library.toml (other files are kept)",
        )
    for folder in (paths.root, paths.style_references, paths.models_dir, paths.generative_dir):
        folder.mkdir(parents=True, exist_ok=True)
    paths.library_toml.write_text(render_library_toml(), encoding="utf-8")
    if not paths.style_md.exists():
        paths.style_md.write_text(STYLE_MD_TEMPLATE.format(name=paths.root.name), encoding="utf-8")
    for keep in (paths.style_references, paths.models_dir, paths.generative_dir):
        (keep / ".gitkeep").touch()
    if not paths.gitignore.exists():
        paths.gitignore.write_text(GITIGNORE_TEMPLATE, encoding="utf-8")
    return paths


# --------------------------------------------------------------------------- new-model
def default_display_name(model_id: str) -> str:
    """``worker_ant`` → ``Worker Ant``."""
    return " ".join(part.capitalize() for part in model_id.split("_") if part)


def render_model_md(
    *,
    display_name: str,
    primary: str,
    anim_globs: tuple[str, ...] = (),
    source_name: str,
) -> str:
    """Render the ``model.md`` template (front matter + ``## prompt`` / ``## notes``)."""
    source: dict[str, Any] = {"primary": primary}
    if anim_globs:
        source["animations"] = list(anim_globs)
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "display_name": display_name,
        "source": source,
        "prompt": {"include_sections": ["base_prompt"]},
        "tags": [],
    }
    body = MODEL_MD_BODY_TEMPLATE.format(display_name=display_name, source_name=source_name)
    text: str = frontmatter.dumps(frontmatter.Post(body, **metadata), sort_keys=False)
    return text + "\n"


def _place_source(source_fbx: Path, dest: Path, mode: CopyMode) -> None:
    if mode == "copy":
        shutil.copy2(source_fbx, dest)
    elif mode == "move":
        shutil.move(str(source_fbx), str(dest))
    else:
        dest.symlink_to(source_fbx.resolve())


def new_model(
    lib_root: Path,
    model_id: str,
    source_fbx: Path,
    *,
    display_name: str | None = None,
    anim_globs: tuple[str, ...] = (),
    mode: CopyMode = "copy",
) -> ModelPaths:
    """Scaffold ``models/<model_id>/`` with a ``model.md`` template and the source FBX.

    Args:
        lib_root: Library root.
        model_id: New model id (validated against the ``model_id`` regex).
        source_fbx: The rigged FBX to bring in.
        display_name: Human name; defaults to a title-cased ``model_id``.
        anim_globs: Explicit ``source.animations`` globs to write (default list when empty).
        mode: ``copy`` (default), ``move`` or ``link`` (symlink) the FBX into ``source/``.

    Returns:
        The new model's paths.

    Raises:
        UsageError: Invalid id, missing source file, or the folder already exists (exit 2).
    """
    if not is_valid_model_id(model_id):
        raise UsageError(
            f"invalid model id {model_id!r}", hint=f"model ids must match {MODEL_ID_PATTERN}"
        )
    if not source_fbx.is_file():
        raise UsageError(f"source FBX not found: {source_fbx}")
    paths = LibraryPaths.from_root(lib_root).model(model_id)
    if paths.folder.exists():
        raise UsageError(
            f"{paths.folder} already exists", hint="choose another id or remove the folder"
        )
    paths.source.mkdir(parents=True)
    paths.references.mkdir()
    dest = paths.source / source_fbx.name
    _place_source(source_fbx, dest, mode)
    paths.model_md.write_text(
        render_model_md(
            display_name=display_name or default_display_name(model_id),
            primary=f"source/{source_fbx.name}",
            anim_globs=anim_globs,
            source_name=source_fbx.name,
        ),
        encoding="utf-8",
    )
    return paths
