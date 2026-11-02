"""Tiny Pillow/numpy helpers shared by the texture packers.

Images travel through the packers as ``(H, W, 4)`` float32 RGBA arrays in ``0..1``; files are
8-bit PNGs. Nothing here applies colour management: sRGB-vs-linear is a *tagging* concern handled
by the Unity importer settings (appendix C §4); packing formulas are defined on raw 8-bit values.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt
from PIL import Image as PilImage

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
JPEG_MAGIC = b"\xff\xd8\xff"

ImageFormat = Literal["png", "jpeg"]
RGBA_CHANNELS = 4
RgbaArray = npt.NDArray[np.float32]


def detect_format(data: bytes) -> ImageFormat | None:
    """Identify PNG or JPEG from the leading magic bytes; ``None`` for anything else."""
    if data.startswith(PNG_MAGIC):
        return "png"
    if data.startswith(JPEG_MAGIC):
        return "jpeg"
    return None


def read_rgba(path: Path) -> RgbaArray:
    """Load any Pillow-readable image as float32 RGBA in ``0..1`` (grey/RGB expanded to RGBA)."""
    with PilImage.open(path) as img:
        rgba = np.asarray(img.convert("RGBA"), dtype=np.float32)
    return (rgba / np.float32(255.0)).astype(np.float32)


def to_uint8(values: npt.NDArray[np.floating]) -> npt.NDArray[np.uint8]:
    """Clip to ``0..1`` and round half-to-even to 8-bit."""
    scaled: npt.NDArray[np.float64] = np.rint(np.clip(values, 0.0, 1.0) * 255.0)
    return scaled.astype(np.uint8)


def write_png(path: Path, rgba: npt.NDArray[np.floating], *, channels: int = 4) -> Path:
    """Write an ``(H, W, >=channels)`` float array in ``0..1`` as an 8-bit PNG.

    Args:
        path: Destination; parent directories are created.
        rgba: Float image; only the first ``channels`` channels are written.
        channels: ``4`` for RGBA (default) or ``3`` for RGB.

    Returns:
        ``path``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "RGBA" if channels == RGBA_CHANNELS else "RGB"
    PilImage.fromarray(to_uint8(rgba[..., :channels]), mode=mode).save(path, format="PNG")
    return path


def image_size(path: Path) -> tuple[int, int]:
    """``(width, height)`` of an image file without decoding its pixels."""
    with PilImage.open(path) as img:
        width, height = img.size
    return int(width), int(height)


def resample(rgba: RgbaArray, size: tuple[int, int]) -> RgbaArray:
    """Lanczos-resample a float RGBA array to ``size = (width, height)``; no-op when equal."""
    width, height = size
    if rgba.shape[1] == width and rgba.shape[0] == height:
        return rgba
    planes = [
        np.asarray(
            PilImage.fromarray(np.ascontiguousarray(rgba[..., c]), mode="F").resize(
                (width, height), resample=PilImage.Resampling.LANCZOS
            ),
            dtype=np.float32,
        )
        for c in range(rgba.shape[2])
    ]
    return np.clip(np.stack(planes, axis=-1), 0.0, 1.0).astype(np.float32)
