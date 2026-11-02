from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from PIL import Image

from grounded_pipeline.domain.errors import TexturePackError
from grounded_pipeline.domain.textures import RawTextureSet, TextureFactors
from grounded_pipeline.textures.urp_pack import UrpTextureSet, merge_metallic_roughness, pack_urp

MR = (0, 128, 64, 255)  # roughness 128 in G, metallic 64 in B
BASE = (200, 100, 50, 255)
NORMAL = (128, 128, 255, 255)
OCC = (180, 180, 180, 255)
EMISSIVE = (10, 20, 30, 255)


def _png(path: Path, rgba: tuple[int, ...] | np.ndarray, size: int = 2) -> Path:
    pixels = (
        np.tile(np.asarray(rgba, dtype=np.uint8), (size, size, 1))
        if isinstance(rgba, tuple)
        else np.asarray(rgba, dtype=np.uint8)
    )
    mode = "RGBA" if pixels.shape[-1] == 4 else "RGB"
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels, mode=mode).save(path)
    return path


def _pixels(path: Path | None) -> np.ndarray:
    assert path is not None
    with Image.open(path) as img:
        return np.asarray(img.convert("RGBA")).astype(int)


def _bytes(path: Path | None) -> bytes:
    assert path is not None
    return path.read_bytes()


def _mode(path: Path | None) -> str:
    assert path is not None
    with Image.open(path) as img:
        return img.mode


def _raw(
    root: Path,
    *,
    basecolor: tuple[int, ...] | np.ndarray | None = BASE,
    metallic_roughness: tuple[int, ...] | np.ndarray | None = MR,
    normal: tuple[int, ...] | np.ndarray | None = NORMAL,
    occlusion: tuple[int, ...] | np.ndarray | None = OCC,
    emissive: tuple[int, ...] | np.ndarray | None = EMISSIVE,
    factors: TextureFactors | None = None,
    size: int = 2,
) -> RawTextureSet:
    def maybe(name: str, value: tuple[int, ...] | np.ndarray | None) -> Path | None:
        return None if value is None else _png(root / f"{name}.png", value, size)

    return RawTextureSet(
        basecolor=maybe("basecolor", basecolor),
        metallic_roughness=maybe("metallic_roughness", metallic_roughness),
        normal=maybe("normal", normal),
        occlusion=maybe("occlusion", occlusion),
        emissive=maybe("emissive", emissive),
        factors=factors or TextureFactors(),
        source="extract",
    )


# ------------------------------------------------------------------------------ formulas
def test_pack_full_set_exact_values(tmp_path: Path) -> None:
    urp = pack_urp(_raw(tmp_path / "raw"), tmp_path / "urp", "ant_worker")
    assert isinstance(urp, UrpTextureSet)
    assert urp.basecolor == tmp_path / "urp" / "ant_worker_basecolor.png"
    assert urp.normal == tmp_path / "urp" / "ant_worker_normal.png"
    assert urp.metallic_smoothness == tmp_path / "urp" / "ant_worker_metallic_smoothness.png"
    assert urp.occlusion == tmp_path / "urp" / "ant_worker_occlusion.png"
    assert urp.emission == tmp_path / "urp" / "ant_worker_emission.png"
    assert _pixels(urp.basecolor)[0, 0].tolist() == [200, 100, 50, 255]
    assert _pixels(urp.normal)[0, 0].tolist() == [128, 128, 255, 255]
    assert _mode(urp.normal) == "RGB"
    assert _pixels(urp.metallic_smoothness)[0, 0].tolist() == [64, 0, 0, 255 - 128]
    assert _mode(urp.metallic_smoothness) == "RGBA"
    assert _pixels(urp.occlusion)[0, 0].tolist() == [180, 180, 180, 255]
    assert _pixels(urp.emission)[0, 0].tolist() == [10, 20, 30, 255]
    assert _mode(urp.emission) == "RGB"
    assert urp.resolution == (2, 2)
    assert urp.bump_scale == 1.0
    assert urp.occlusion_strength == 1.0
    assert urp.emission_color == (0.0, 0.0, 0.0)
    assert urp.metallic_fallback == 0.0
    assert urp.smoothness_fallback == 1.0
    assert set(urp.maps()) == {
        "basecolor",
        "normal",
        "metallic_smoothness",
        "occlusion",
        "emission",
    }


def test_occlusion_uses_red_channel_only(tmp_path: Path) -> None:
    urp = pack_urp(_raw(tmp_path / "raw", occlusion=(90, 10, 20, 40)), tmp_path / "urp", "m")
    assert _pixels(urp.occlusion)[0, 0].tolist() == [90, 90, 90, 255]


def test_factors_are_folded_in(tmp_path: Path) -> None:
    factors = TextureFactors(
        base_color=(0.5, 0.5, 0.5, 1.0),
        metallic=0.5,
        roughness=0.5,
        emissive=(1.0, 0.5, 0.25),
        normal_scale=0.8,
        occlusion_strength=0.6,
        emissive_strength=4.0,
    )
    urp = pack_urp(_raw(tmp_path / "raw", factors=factors), tmp_path / "urp", "m")
    assert _pixels(urp.basecolor)[0, 0].tolist() == [100, 50, 25, 255]
    assert _pixels(urp.metallic_smoothness)[0, 0].tolist() == [32, 0, 0, 255 - 64]
    assert _pixels(urp.emission)[0, 0].tolist() == [10, 20, 30, 255]  # emission map is copied
    assert urp.emission_color == (4.0, 2.0, 1.0)
    assert urp.bump_scale == 0.8
    assert urp.occlusion_strength == 0.6


def test_basecolor_alpha_is_kept_and_scaled(tmp_path: Path) -> None:
    factors = TextureFactors(base_color=(1.0, 1.0, 1.0, 0.5))
    raw = _raw(tmp_path / "raw", basecolor=(10, 20, 30, 200), factors=factors)
    urp = pack_urp(raw, tmp_path / "urp", "m")
    assert _pixels(urp.basecolor)[0, 0].tolist() == [10, 20, 30, 100]
    assert _mode(urp.basecolor) == "RGBA"


# ------------------------------------------------------------------------------ missing maps
def test_missing_metallic_roughness_sets_fallbacks(tmp_path: Path) -> None:
    urp = pack_urp(_raw(tmp_path / "raw", metallic_roughness=None), tmp_path / "urp", "m")
    assert urp.metallic_smoothness is None
    assert urp.metallic_fallback == 0.0
    assert urp.smoothness_fallback == 0.5
    assert "metallic_smoothness" not in urp.maps()
    assert sorted(p.name for p in (tmp_path / "urp").iterdir()) == [
        "m_basecolor.png",
        "m_emission.png",
        "m_normal.png",
        "m_occlusion.png",
    ]


def test_only_basecolor(tmp_path: Path) -> None:
    raw = _raw(
        tmp_path / "raw", metallic_roughness=None, normal=None, occlusion=None, emissive=None
    )
    urp = pack_urp(raw, tmp_path / "urp", "m")
    assert urp.maps().keys() == {"basecolor"}
    assert urp.resolution == (2, 2)


def test_no_maps_at_all(tmp_path: Path) -> None:
    raw = _raw(
        tmp_path / "raw",
        basecolor=None,
        metallic_roughness=None,
        normal=None,
        occlusion=None,
        emissive=None,
    )
    urp = pack_urp(raw, tmp_path / "urp", "m")
    assert urp.maps() == {}
    assert urp.resolution is None
    assert urp.smoothness_fallback == 0.5
    assert (tmp_path / "urp").is_dir()


# ------------------------------------------------------------------------------ resampling
def test_differing_sizes_are_resampled_to_the_largest(tmp_path: Path) -> None:
    raw = _raw(tmp_path / "raw", size=2)
    big = _png(tmp_path / "raw" / "basecolor4.png", BASE, size=4)
    raw = RawTextureSet(**{**raw.__dict__, "basecolor": big})
    urp = pack_urp(raw, tmp_path / "urp", "m")
    assert urp.resolution == (4, 4)
    for path in urp.maps().values():
        assert _pixels(path).shape == (4, 4, 4)
    # Solid colours survive Lanczos exactly.
    assert _pixels(urp.metallic_smoothness)[3, 3].tolist() == [64, 0, 0, 127]


def test_largest_is_chosen_by_area_and_non_square_sizes_work(tmp_path: Path) -> None:
    wide = np.tile(np.asarray(BASE, dtype=np.uint8), (2, 8, 1))
    tall = np.tile(np.asarray(NORMAL, dtype=np.uint8), (3, 3, 1))
    raw = _raw(
        tmp_path / "raw",
        basecolor=wide,
        normal=tall,
        metallic_roughness=None,
        occlusion=None,
        emissive=None,
    )
    urp = pack_urp(raw, tmp_path / "urp", "m")
    assert urp.resolution == (8, 2)
    assert _pixels(urp.normal).shape == (2, 8, 4)


# ------------------------------------------------------------------------------ errors
def test_unreadable_input_raises_texture_pack_error(tmp_path: Path) -> None:
    raw = _raw(tmp_path / "raw")
    assert raw.basecolor is not None
    raw.basecolor.write_bytes(b"not an image")
    with pytest.raises(TexturePackError) as info:
        pack_urp(raw, tmp_path / "urp", "m")
    assert info.value.code == "E_TEXTURE_PACK"
    assert info.value.detail["map"] == "basecolor"


def test_missing_input_file_raises(tmp_path: Path) -> None:
    raw = _raw(tmp_path / "raw", normal=None)
    raw = RawTextureSet(**{**raw.__dict__, "normal": tmp_path / "raw" / "gone.png"})
    with pytest.raises(TexturePackError, match="normal"):
        pack_urp(raw, tmp_path / "urp", "m")


def test_unwritable_output_raises(tmp_path: Path) -> None:
    blocker = tmp_path / "urp"
    blocker.write_text("file, not a directory")
    with pytest.raises((TexturePackError, OSError)):
        pack_urp(_raw(tmp_path / "raw"), blocker, "m")


# ------------------------------------------------------------------------------ merge
def test_merge_metallic_roughness(tmp_path: Path) -> None:
    metallic = _png(tmp_path / "metallic.png", (64, 64, 64, 255))
    roughness = _png(tmp_path / "roughness.png", (128, 128, 128, 255))
    out = merge_metallic_roughness(metallic, roughness, tmp_path / "raw" / "metallic_roughness.png")
    assert out == tmp_path / "raw" / "metallic_roughness.png"
    assert _pixels(out)[0, 0].tolist() == [0, 128, 64, 255]
    # Packing the merged map yields the same result as packing an extracted one.
    raw = RawTextureSet(None, out, None, None, None, TextureFactors(), "bake")
    urp = pack_urp(raw, tmp_path / "urp", "m")
    assert _pixels(urp.metallic_smoothness)[0, 0].tolist() == [64, 0, 0, 127]


def test_merge_reconciles_sizes_and_reports_errors(tmp_path: Path) -> None:
    metallic = _png(tmp_path / "metallic.png", (64, 64, 64, 255), size=2)
    roughness = _png(tmp_path / "roughness.png", (128, 128, 128, 255), size=4)
    out = merge_metallic_roughness(metallic, roughness, tmp_path / "mr.png")
    assert _pixels(out).shape == (4, 4, 4)
    with pytest.raises(TexturePackError, match="roughness"):
        merge_metallic_roughness(metallic, tmp_path / "nope.png", tmp_path / "mr2.png")


# ------------------------------------------------------------------------------ properties
rgba_2x2 = st.lists(st.integers(0, 255), min_size=16, max_size=16).map(
    lambda v: np.asarray(v, dtype=np.uint8).reshape(2, 2, 4)
)
unit = st.floats(0.0, 1.0, allow_nan=False)


@settings(max_examples=60, deadline=None)
@given(mr=rgba_2x2)
def test_metallic_smoothness_alpha_is_exactly_255_minus_g(mr: np.ndarray) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        raw = _raw(
            root / "raw",
            basecolor=None,
            metallic_roughness=mr,
            normal=None,
            occlusion=None,
            emissive=None,
        )
        urp = pack_urp(raw, root / "urp", "m")
        out = _pixels(urp.metallic_smoothness)
        np.testing.assert_array_equal(out[..., 3], 255 - mr[..., 1].astype(int))
        np.testing.assert_array_equal(out[..., 0], mr[..., 2].astype(int))
        assert not out[..., 1].any()
        assert not out[..., 2].any()


@settings(max_examples=60, deadline=None)
@given(base=rgba_2x2, normal=rgba_2x2, emissive=rgba_2x2)
def test_identity_factors_copy_exactly_and_are_idempotent(
    base: np.ndarray, normal: np.ndarray, emissive: np.ndarray
) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        raw = _raw(
            root / "raw",
            basecolor=base,
            metallic_roughness=None,
            normal=normal,
            occlusion=None,
            emissive=emissive,
        )
        first = pack_urp(raw, root / "urp1", "m")
        np.testing.assert_array_equal(_pixels(first.basecolor), base.astype(int))
        np.testing.assert_array_equal(_pixels(first.normal)[..., :3], normal[..., :3].astype(int))
        np.testing.assert_array_equal(
            _pixels(first.emission)[..., :3], emissive[..., :3].astype(int)
        )
        again = RawTextureSet(
            first.basecolor, None, first.normal, None, first.emission, TextureFactors(), "extract"
        )
        second = pack_urp(again, root / "urp2", "m")
        for name in ("basecolor", "normal", "emission"):
            assert _bytes(getattr(second, name)) == _bytes(getattr(first, name))


@settings(max_examples=60, deadline=None)
@given(
    mr=rgba_2x2,
    base=rgba_2x2,
    metallic=unit,
    roughness=unit,
    tint=st.tuples(unit, unit, unit, unit),
)
def test_channel_ranges_preserved_under_arbitrary_factors(
    mr: np.ndarray,
    base: np.ndarray,
    metallic: float,
    roughness: float,
    tint: tuple[float, float, float, float],
) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        factors = TextureFactors(base_color=tint, metallic=metallic, roughness=roughness)
        raw = _raw(
            root / "raw",
            basecolor=base,
            metallic_roughness=mr,
            normal=None,
            occlusion=None,
            emissive=None,
            factors=factors,
        )
        urp = pack_urp(raw, root / "urp", "m")
        ms, bc = _pixels(urp.metallic_smoothness), _pixels(urp.basecolor)
        assert ms.min() >= 0
        assert ms.max() <= 255
        assert (ms[..., 0] <= mr[..., 2].astype(int)).all()  # metallic never exceeds the map
        assert (ms[..., 3] >= 255 - mr[..., 1].astype(int)).all()  # smoothness never below 1 - g
        assert (bc <= base.astype(int)).all()  # tint can only darken
