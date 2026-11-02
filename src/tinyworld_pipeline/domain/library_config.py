"""Library configuration: one pydantic model per ``library.toml`` section (plan 05 §3, 06 §2).

The field defaults here are the single source of default values (``config/defaults.py`` only
re-exports an instance).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, HttpUrl

from tinyworld_pipeline.domain._base import FrozenModel

TextureQuality = Literal["fast", "standard", "detailed", "extreme"]
PromptMode = Literal["text", "text+style", "image", "images"]
RouteName = Literal["auto", "extract", "bake"]
BakeMap = Literal["basecolor", "normal", "metallic", "roughness", "occlusion", "emission"]
FbxScaleOption = Literal["FBX_SCALE_NONE", "FBX_SCALE_UNITS", "FBX_SCALE_CUSTOM", "FBX_SCALE_ALL"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR"]

TRIPO_DEFAULT_BASE_URL = "https://openapi.tripo3d.ai/v3"
TRIPO_DEFAULT_MODEL_VERSION = "v3.5-20260815"


class LibrarySection(FrozenModel):
    name: str = "library"
    unity_import_dir: Path | None = None
    concurrency: int = Field(2, ge=1, le=16)
    blender_parallelism: int = Field(1, ge=1, le=8)


class BlenderSection(FrozenModel):
    executable: Path | None = None
    allow_unsupported: bool = False
    job_timeout_s: int = Field(1800, ge=60)
    keep_blend_files: bool = True


class GeometrySection(FrozenModel):
    strategy: Literal["subdivide"] = "subdivide"  # extension seam (plan 09 §2.9)
    subdivision_levels: int = Field(1, ge=0, le=3)
    subdivision_type: Literal["catmull", "simple"] = "catmull"
    shade_smooth: bool = True
    shapekeys: Literal["fail", "drop", "skip"] = "fail"
    merge_material_slots: bool = True
    keep_only_one_uv_layer: bool = True
    include_unskinned: bool = True
    max_triangles: int = Field(500_000, ge=1_000)


class UvSection(FrozenModel):
    mode: Literal["auto", "keep", "unwrap"] = "auto"
    palette_texture_max_px: int = Field(128, ge=1)
    overlap_ratio_threshold: float = Field(0.20, ge=0, le=1)
    smart_project_angle_deg: float = Field(66.0, gt=0, lt=90)
    island_margin: float = Field(0.02, ge=0, le=1)
    pack_margin: float = Field(0.01, ge=0, le=1)


class TextureSection(FrozenModel):
    provider: Literal["tripo"] = "tripo"
    model_version: str = TRIPO_DEFAULT_MODEL_VERSION
    quality: TextureQuality = "standard"
    pbr: bool = True
    bake: bool = True
    delight: bool = True
    alignment: Literal["original_image", "geometry"] = "original_image"
    prompt_mode: PromptMode = "text+style"
    seed: int = Field(1337, ge=0, le=2**31 - 1)
    style_image: Path = Path("style/references/primary.png")


class VerifySection(FrozenModel):
    route: RouteName = "auto"
    position_quantum: float = Field(1e-4, gt=0)
    uv_quantum: float = Field(1e-4, gt=0)
    allow_geometry_change: bool = False


def _default_bake_maps() -> list[BakeMap]:
    return ["basecolor", "normal", "metallic", "roughness", "occlusion", "emission"]


def _default_credit_table() -> dict[str, float]:
    return {"fast": 10.0, "standard": 10.0, "detailed": 20.0, "extreme": 30.0, "style_image": 0.0}


class BakeSection(FrozenModel):
    resolution: Literal[512, 1024, 2048, 4096] = 2048
    samples: int = Field(16, ge=1, le=256)
    margin_px: int = Field(16, ge=0, le=64)
    cage_extrusion_ratio: float = Field(0.01, ge=0)
    max_ray_distance_ratio: float = Field(0.05, ge=0)
    maps: list[BakeMap] = Field(default_factory=_default_bake_maps)


class ExportSection(FrozenModel):
    fbx_scale_option: FbxScaleOption = "FBX_SCALE_ALL"
    roundtrip_companion_animations: bool = True
    embed_textures: bool = False


class UnitySection(FrozenModel):
    avatar_mode: Literal["copy_from_source", "create_from_model"] = "copy_from_source"
    source_asset_root: str = ""
    shader: str = "Universal Render Pipeline/Lit"


class TripoSection(FrozenModel):
    base_url: HttpUrl = HttpUrl(TRIPO_DEFAULT_BASE_URL)
    poll_interval_s: float = Field(2.0, ge=0.5)
    poll_timeout_s: int = Field(900, ge=30)
    max_concurrent_tasks: int = Field(4, ge=1, le=16)
    min_request_interval_s: float = Field(0.25, ge=0)
    max_credits_per_run: float = Field(0, ge=0)
    use_presign_above_mb: int = Field(60, ge=1, le=150)
    credit_table: dict[str, float] = Field(default_factory=_default_credit_table)


class LoggingSection(FrozenModel):
    level: LogLevel = "INFO"
    json_console: bool = False


class LibraryConfig(FrozenModel):
    """Whole ``library.toml``; also the *effective* per-model configuration after layering."""

    schema_version: Literal[1] = 1
    library: LibrarySection = LibrarySection()
    blender: BlenderSection = BlenderSection()
    geometry: GeometrySection = GeometrySection()
    uv: UvSection = UvSection()
    texture: TextureSection = TextureSection()
    verify: VerifySection = VerifySection()
    bake: BakeSection = BakeSection()
    export: ExportSection = ExportSection()
    unity: UnitySection = UnitySection()
    tripo: TripoSection = TripoSection()
    logging: LoggingSection = LoggingSection()


SECTION_MODELS: dict[str, type[FrozenModel]] = {
    "library": LibrarySection,
    "blender": BlenderSection,
    "geometry": GeometrySection,
    "uv": UvSection,
    "texture": TextureSection,
    "verify": VerifySection,
    "bake": BakeSection,
    "export": ExportSection,
    "unity": UnitySection,
    "tripo": TripoSection,
    "logging": LoggingSection,
}
"""Section name → section model, in ``library.toml`` order."""
