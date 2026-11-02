"""Raw texture set value objects shared by the extract and bake routes (plan 11 §5.1).

Both routes (``gltf.extract`` and the Blender ``bake_textures`` job) produce the same shape before
URP packing (``textures.urp_pack``); it lives in ``domain`` so neither adapter package has to import
the other.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class TextureFactors:
    """glTF material scalars that the packer folds into the maps or the manifest."""

    base_color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    metallic: float = 1.0
    roughness: float = 1.0
    emissive: tuple[float, float, float] = (0.0, 0.0, 0.0)
    normal_scale: float = 1.0
    occlusion_strength: float = 1.0
    emissive_strength: float = 1.0


@dataclass(frozen=True)
class RawTextureSet:
    """PNG paths (``None`` when the source lacks the map) plus factors, from either route."""

    basecolor: Path | None
    metallic_roughness: Path | None
    normal: Path | None
    occlusion: Path | None
    emissive: Path | None
    factors: TextureFactors
    source: Literal["extract", "bake"]
