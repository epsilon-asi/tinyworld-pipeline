"""Pack a raw glTF texture set into URP Lit (metallic workflow) maps (plan 11 section 6).

Channel table (all maths on ``0..1`` floats, output 8-bit PNG):

* ``<id>_basecolor.png`` (RGBA): ``rgb = base.rgb * baseColorFactor.rgb``;
  ``a = base.a * baseColorFactor.a`` (alpha kept; URP opaque ignores it).
* ``<id>_normal.png`` (RGB): copy. glTF and Unity are both +Y / OpenGL, so no green flip;
  ``normalTexture.scale`` goes to ``_BumpScale``.
* ``<id>_metallic_smoothness.png`` (RGBA): ``R = mr.b * metallicFactor``; ``G = B = 0``;
  ``A = 1 - mr.g * roughnessFactor``.
* ``<id>_occlusion.png`` (RGBA): ``R = G = B = occ.r`` (URP samples G); ``A = 1``; strength goes
  to ``_OcclusionStrength``.
* ``<id>_emission.png`` (RGB): copy; ``_EmissionColor = emissiveFactor * emissiveStrength``.

Missing inputs produce missing outputs. Without a metallic-roughness map the manifest falls back to
``_Metallic = 0`` and ``_Smoothness = 0.5``. When input sizes differ every map is Lanczos-resampled
to the largest. The input :class:`RawTextureSet` lives in ``domain.textures`` so both the extract
and bake routes feed the same packer.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from grounded_pipeline.domain.errors import TexturePackError
from grounded_pipeline.domain.textures import RawTextureSet
from grounded_pipeline.textures.image_io import RgbaArray, read_rgba, resample, write_png

METALLIC_FALLBACK = 0.0
SMOOTHNESS_FALLBACK_WITHOUT_MAP = 0.5
SMOOTHNESS_MULTIPLIER_WITH_MAP = 1.0


@dataclass(frozen=True)
class UrpTextureSet:
    """Packed URP maps plus the material scalars the Unity manifest needs."""

    basecolor: Path | None
    normal: Path | None
    metallic_smoothness: Path | None
    occlusion: Path | None
    emission: Path | None
    resolution: tuple[int, int] | None  # (width, height) shared by every written map
    bump_scale: float  # _BumpScale
    occlusion_strength: float  # _OcclusionStrength
    emission_color: tuple[float, float, float]  # _EmissionColor, linear = factor x strength
    metallic_fallback: float  # _Metallic
    smoothness_fallback: float  # _Smoothness

    def maps(self) -> dict[str, Path]:
        """Only the maps that were written, keyed by URP slot name."""
        candidates = {
            "basecolor": self.basecolor,
            "normal": self.normal,
            "metallic_smoothness": self.metallic_smoothness,
            "occlusion": self.occlusion,
            "emission": self.emission,
        }
        return {k: v for k, v in candidates.items() if v is not None}


def pack_urp(raw: RawTextureSet, out_dir: Path, model_id: str) -> UrpTextureSet:
    """Convert a raw texture set into URP maps named ``<model_id>_<map>.png`` in ``out_dir``.

    Args:
        raw: Raw PNG paths plus factors from the extract or bake route.
        out_dir: Destination directory (created).
        model_id: File-name prefix.

    Returns:
        Paths of the maps written and the scalar material properties.

    Raises:
        TexturePackError: An input cannot be read or has an unexpected shape.
    """
    factors = raw.factors
    inputs = _load_inputs(raw)
    if inputs:
        target = max((img.shape[1], img.shape[0]) for img in inputs.values())
        inputs = {k: resample(v, target) for k, v in inputs.items()}
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    if "basecolor" in inputs:
        base = inputs["basecolor"] * np.asarray(factors.base_color, dtype=np.float32)
        written["basecolor"] = _write(out_dir, model_id, "basecolor", base, 4)
    if "normal" in inputs:
        written["normal"] = _write(out_dir, model_id, "normal", inputs["normal"], 3)
    if "metallic_roughness" in inputs:
        mr = inputs["metallic_roughness"]
        packed = np.zeros_like(mr)
        packed[..., 0] = mr[..., 2] * np.float32(factors.metallic)
        packed[..., 3] = np.float32(1.0) - mr[..., 1] * np.float32(factors.roughness)
        written["metallic_smoothness"] = _write(out_dir, model_id, "metallic_smoothness", packed, 4)
    if "occlusion" in inputs:
        occ = inputs["occlusion"]
        packed = np.ones_like(occ)
        packed[..., :3] = occ[..., :1]
        written["occlusion"] = _write(out_dir, model_id, "occlusion", packed, 4)
    if "emissive" in inputs:
        written["emission"] = _write(out_dir, model_id, "emission", inputs["emissive"], 3)
    has_mr = "metallic_smoothness" in written
    strength = factors.emissive_strength
    return UrpTextureSet(
        basecolor=written.get("basecolor"),
        normal=written.get("normal"),
        metallic_smoothness=written.get("metallic_smoothness"),
        occlusion=written.get("occlusion"),
        emission=written.get("emission"),
        resolution=target if inputs else None,
        bump_scale=factors.normal_scale,
        occlusion_strength=factors.occlusion_strength,
        emission_color=(
            factors.emissive[0] * strength,
            factors.emissive[1] * strength,
            factors.emissive[2] * strength,
        ),
        metallic_fallback=METALLIC_FALLBACK,
        smoothness_fallback=(
            SMOOTHNESS_MULTIPLIER_WITH_MAP if has_mr else SMOOTHNESS_FALLBACK_WITHOUT_MAP
        ),
    )


def merge_metallic_roughness(metallic_png: Path, roughness_png: Path, out: Path) -> Path:
    """Combine two baked greyscale maps into a glTF-style ``metallic_roughness.png``.

    The red channel of each input is used (bakes are greyscale). Output: ``R = 0``,
    ``G = roughness``, ``B = metallic``, ``A = 1``; sizes are reconciled to the larger map.

    Args:
        metallic_png: Baked metallic map.
        roughness_png: Baked roughness map.
        out: Destination PNG path.

    Returns:
        ``out``.

    Raises:
        TexturePackError: Either input cannot be read.
    """
    metallic = _read("metallic", metallic_png)
    roughness = _read("roughness", roughness_png)
    target = max((metallic.shape[1], metallic.shape[0]), (roughness.shape[1], roughness.shape[0]))
    metallic, roughness = resample(metallic, target), resample(roughness, target)
    merged = np.zeros_like(metallic)
    merged[..., 1] = roughness[..., 0]
    merged[..., 2] = metallic[..., 0]
    merged[..., 3] = 1.0
    return write_png(out, merged, channels=4)


# ------------------------------------------------------------------------------------ helpers
def _load_inputs(raw: RawTextureSet) -> dict[str, RgbaArray]:
    paths = {
        "basecolor": raw.basecolor,
        "metallic_roughness": raw.metallic_roughness,
        "normal": raw.normal,
        "occlusion": raw.occlusion,
        "emissive": raw.emissive,
    }
    return {slot: _read(slot, path) for slot, path in paths.items() if path is not None}


def _read(slot: str, path: Path) -> RgbaArray:
    try:
        return read_rgba(path)
    except (OSError, ValueError) as exc:
        raise TexturePackError(
            f"cannot read {slot} map {path}: {exc}", detail={"map": slot, "path": str(path)}
        ) from exc


def _write(out_dir: Path, model_id: str, name: str, data: RgbaArray, channels: int) -> Path:
    try:
        return write_png(out_dir / f"{model_id}_{name}.png", data, channels=channels)
    except OSError as exc:
        raise TexturePackError(f"cannot write {name} map: {exc}", detail={"map": name}) from exc
