from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from grounded_pipeline.domain.textures import RawTextureSet, TextureFactors


def test_texture_factors_defaults_are_gltf_defaults() -> None:
    factors = TextureFactors()
    assert factors.base_color == (1.0, 1.0, 1.0, 1.0)
    assert (factors.metallic, factors.roughness) == (1.0, 1.0)
    assert factors.emissive == (0.0, 0.0, 0.0)
    assert (factors.normal_scale, factors.occlusion_strength, factors.emissive_strength) == (
        1.0,
        1.0,
        1.0,
    )


def test_raw_texture_set_construction_and_frozen(tmp_path: Path) -> None:
    raw = RawTextureSet(
        basecolor=tmp_path / "basecolor.png",
        metallic_roughness=None,
        normal=tmp_path / "normal.png",
        occlusion=None,
        emissive=None,
        factors=TextureFactors(metallic=0.5),
        source="bake",
    )
    assert raw.basecolor == tmp_path / "basecolor.png"
    assert raw.metallic_roughness is None
    assert raw.factors.metallic == 0.5
    assert raw.source == "bake"
    with pytest.raises(dataclasses.FrozenInstanceError):
        raw.source = "extract"  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        raw.factors.metallic = 1.0  # type: ignore[misc]
    assert raw == dataclasses.replace(raw)
