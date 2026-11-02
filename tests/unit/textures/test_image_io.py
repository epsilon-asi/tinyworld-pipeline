from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from tinyworld_pipeline.textures.image_io import (
    detect_format,
    image_size,
    read_rgba,
    resample,
    to_uint8,
    write_png,
)


def _save(path: Path, pixels: np.ndarray, mode: str) -> Path:
    Image.fromarray(pixels, mode=mode).save(path)
    return path


def test_detect_format() -> None:
    assert detect_format(b"\x89PNG\r\n\x1a\n...") == "png"
    assert detect_format(b"\xff\xd8\xff\xe1...") == "jpeg"
    assert detect_format(b"GIF89a") is None
    assert detect_format(b"") is None


def test_read_rgba_expands_rgb_and_grey(tmp_path: Path) -> None:
    rgb = _save(tmp_path / "rgb.png", np.full((2, 3, 3), [10, 20, 30], dtype=np.uint8), "RGB")
    grey = _save(tmp_path / "l.png", np.full((2, 2), 51, dtype=np.uint8), "L")
    a = read_rgba(rgb)
    assert a.shape == (2, 3, 4)
    assert a.dtype == np.float32
    np.testing.assert_allclose(a[0, 0], [10 / 255, 20 / 255, 30 / 255, 1.0], atol=1e-7)
    b = read_rgba(grey)
    assert b.shape == (2, 2, 4)
    np.testing.assert_allclose(b[1, 1], [0.2, 0.2, 0.2, 1.0], atol=1e-7)


def test_to_uint8_rounds_and_clips() -> None:
    values = np.array([-0.5, 0.0, 0.5, 1.0, 1.7, 127.5 / 255], dtype=np.float32)
    assert to_uint8(values).tolist() == [0, 0, 128, 255, 255, 128]
    assert to_uint8(values).dtype == np.uint8


def test_write_png_rgba_and_rgb_round_trip(tmp_path: Path) -> None:
    rgba = np.zeros((2, 2, 4), dtype=np.float32)
    rgba[..., 0] = 1.0
    rgba[..., 3] = 0.5
    out4 = write_png(tmp_path / "nested" / "a.png", rgba)
    out3 = write_png(tmp_path / "b.png", rgba, channels=3)
    with Image.open(out4) as img:
        assert img.mode == "RGBA"
        assert list(np.asarray(img)[0, 0]) == [255, 0, 0, 128]
    with Image.open(out3) as img:
        assert img.mode == "RGB"
        assert list(np.asarray(img)[0, 0]) == [255, 0, 0]
    np.testing.assert_array_equal(to_uint8(read_rgba(out4)), to_uint8(rgba))


def test_image_size(tmp_path: Path) -> None:
    path = _save(tmp_path / "s.png", np.zeros((3, 5, 3), dtype=np.uint8), "RGB")
    assert image_size(path) == (5, 3)


def test_resample_identity_returns_same_array() -> None:
    rgba = np.random.default_rng(0).random((2, 2, 4), dtype=np.float32)
    assert resample(rgba, (2, 2)) is rgba


@pytest.mark.parametrize("size", [(4, 4), (1, 1), (3, 5)])
def test_resample_solid_colour_is_preserved(size: tuple[int, int]) -> None:
    rgba = np.tile(np.array([0.25, 0.5, 0.75, 1.0], dtype=np.float32), (2, 2, 1))
    out = resample(rgba, size)
    assert out.shape == (size[1], size[0], 4)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, np.tile(rgba[0, 0], (size[1], size[0], 1)), atol=1e-6)


def test_resample_output_is_clipped_to_unit_range() -> None:
    rgba = np.zeros((4, 4, 4), dtype=np.float32)
    rgba[::2, ::2] = 1.0  # checkerboard; Lanczos overshoots without clipping
    out = resample(rgba, (8, 8))
    assert out.min() >= 0.0
    assert out.max() <= 1.0
