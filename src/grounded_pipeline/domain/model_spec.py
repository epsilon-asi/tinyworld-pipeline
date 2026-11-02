"""Per-model specification: ``model.md`` front matter and the resolved model (plan 06 §3).

Override classes are explicit, hand-written partial mirrors of the section models (every field
``Optional`` with default ``None``; ADR-0003). ``tests/unit/domain/test_overrides_parity.py``
asserts they never drift from the sections they mirror.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field

from grounded_pipeline.domain._base import FrozenModel
from grounded_pipeline.domain.ids import ModelId
from grounded_pipeline.domain.library_config import (
    BakeMap,
    FbxScaleOption,
    LibraryConfig,
    PromptMode,
    RouteName,
    TextureQuality,
)
from grounded_pipeline.domain.prompt import ComposedPrompt
from grounded_pipeline.domain.style_spec import StyleDocument


class SourceSpec(FrozenModel):
    primary: Path | None = None  # relative to model folder
    animations: list[str] = Field(default_factory=lambda: ["source/animations/*.fbx"])  # globs
    license: str | None = None


class GuidanceImages(FrozenModel):
    front: Path
    left: Path | None = None
    back: Path | None = None
    right: Path | None = None


# --------------------------------------------------------------------------- override mirrors
class GeometryOverrides(FrozenModel):
    strategy: Literal["subdivide"] | None = None
    subdivision_levels: int | None = Field(None, ge=0, le=3)
    subdivision_type: Literal["catmull", "simple"] | None = None
    shade_smooth: bool | None = None
    shapekeys: Literal["fail", "drop", "skip"] | None = None
    merge_material_slots: bool | None = None
    keep_only_one_uv_layer: bool | None = None
    include_unskinned: bool | None = None
    max_triangles: int | None = Field(None, ge=1_000)


class UvOverrides(FrozenModel):
    mode: Literal["auto", "keep", "unwrap"] | None = None
    palette_texture_max_px: int | None = Field(None, ge=1)
    overlap_ratio_threshold: float | None = Field(None, ge=0, le=1)
    smart_project_angle_deg: float | None = Field(None, gt=0, lt=90)
    island_margin: float | None = Field(None, ge=0, le=1)
    pack_margin: float | None = Field(None, ge=0, le=1)


class TextureOverrides(FrozenModel):
    provider: Literal["tripo"] | None = None
    model_version: str | None = None
    quality: TextureQuality | None = None
    pbr: bool | None = None
    bake: bool | None = None
    delight: bool | None = None
    alignment: Literal["original_image", "geometry"] | None = None
    prompt_mode: PromptMode | None = None
    seed: int | None = Field(None, ge=0, le=2**31 - 1)
    style_image: Path | None = None
    guidance_images: GuidanceImages | None = None  # extra: not part of TextureSection


class VerifyOverrides(FrozenModel):
    route: RouteName | None = None
    position_quantum: float | None = Field(None, gt=0)
    uv_quantum: float | None = Field(None, gt=0)
    allow_geometry_change: bool | None = None


class BakeOverrides(FrozenModel):
    resolution: Literal[512, 1024, 2048, 4096] | None = None
    samples: int | None = Field(None, ge=1, le=256)
    margin_px: int | None = Field(None, ge=0, le=64)
    cage_extrusion_ratio: float | None = Field(None, ge=0)
    max_ray_distance_ratio: float | None = Field(None, ge=0)
    maps: list[BakeMap] | None = None


class ExportOverrides(FrozenModel):
    fbx_scale_option: FbxScaleOption | None = None
    roundtrip_companion_animations: bool | None = None
    embed_textures: bool | None = None


class UnityOverrides(FrozenModel):
    avatar_mode: Literal["copy_from_source", "create_from_model"] | None = None
    source_asset_root: str | None = None
    shader: str | None = None
    source_asset_path: str | None = None  # extra: Unity path to the vendor FBX (avatar copy)


OVERRIDE_EXTRA_FIELDS: dict[str, frozenset[str]] = {
    "texture": frozenset({"guidance_images"}),
    "unity": frozenset({"source_asset_path"}),
}
"""Fields an override class has beyond its section model (parity test and layering skip them)."""


class PromptSpec(FrozenModel):
    include_sections: list[str] = Field(default_factory=lambda: ["base_prompt"])
    template: str = "{base_prompt}\n\n{model_prompt}"


class ModelFrontMatter(FrozenModel):
    """YAML front matter of ``model.md``. Unknown keys are errors (``E_UNKNOWN_KEY``)."""

    schema_version: Literal[1] = 1
    display_name: str
    source: SourceSpec = SourceSpec()
    geometry: GeometryOverrides | None = None
    uv: UvOverrides | None = None
    texture: TextureOverrides | None = None
    verify: VerifyOverrides | None = None
    bake: BakeOverrides | None = None
    export: ExportOverrides | None = None
    unity: UnityOverrides | None = None
    prompt: PromptSpec = PromptSpec()
    tags: list[str] = Field(default_factory=list)


class ModelDocument(FrozenModel):
    """Parsed ``model.md``."""

    model_id: ModelId
    folder: Path  # absolute
    front: ModelFrontMatter
    sections: dict[str, str]  # lower-cased heading → body ("prompt", "notes", …)


class ResolvedModel(FrozenModel):
    """``ModelDocument`` + effective config + resolved absolute paths. Output of ``discover``."""

    doc: ModelDocument
    config: LibraryConfig  # effective, after layering
    primary_fbx: Path
    animation_fbx: list[Path]
    style: StyleDocument
    composed_prompt: ComposedPrompt
