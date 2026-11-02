"""WP-08 acceptance: ``bake_textures`` under a real Blender (plan 11 §8, 16 §WP-08).

``prepared.blend`` comes from the real ``prepare_model`` job on ``rigged_cube.fbx`` (no
subdivision, UVs kept). The "provider result" is the same cube re-exported by
``tools/make_bake_fixture.py`` with ``cube_texture.png`` as base colour (nearest sampling), a flat
normal map, roughness 0.5 and metallic 0.25. Baking that onto the prepared cube must reproduce the
texture: the baked base colour equals the nearest-upscaled 4x4 texture (mean abs error < 2/255 on
covered texels away from the quadrant seams), the normal map is flat ``(128, 128, 255) ± 3``, the
roughness bake is ``128 ± 3`` and the EMIT-trick metallic bake is ``64 ± 3`` (R-14: no colour
management on data maps). Base colour is baked through the EMIT trick as well: on 4.5.12 a
``DIFFUSE``/``COLOR`` bake of a Principled BSDF with metallic 0.25 returned ``(221, 0, 0)`` for a
pure red texel (``(1 - metallic)`` times the dielectric layer's ~4 % loss), a mean error of
13/255 against the plan's < 2/255 bar.
"""

from __future__ import annotations

import os
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from ulid import ULID

from grounded_pipeline.blender.protocol import (
    BakeTexturesData,
    BakeTexturesParams,
    JobRequest,
    JobResult,
    PrepareModelData,
    PrepareModelParams,
)
from grounded_pipeline.blender.runner import SubprocessBlenderRunner, default_entry_script
from grounded_pipeline.domain.library_config import BakeSection, GeometrySection, UvSection
from grounded_pipeline.domain.ports import CancelToken

pytestmark = pytest.mark.blender

ROOT = Path(__file__).resolve().parents[3]
FIXTURE_DIR = ROOT / "tests" / "fixtures" / "fbx"
FIXTURE_TOOL = ROOT / "tools" / "make_bake_fixture.py"
SOURCE_FBX = FIXTURE_DIR / "rigged_cube.fbx"
TEXTURE_PNG = FIXTURE_DIR / "cube_texture.png"
RESOLUTION = 512  # smallest BakeSection.resolution
SAMPLES = 4
MARGIN_PX = 4
SEAM_BAND_PX = MARGIN_PX + 2  # texels this close to the texture's quadrant seams are ignored
ROUGHNESS, METALLIC = 0.5, 0.25
TOLERANCE = 3


@pytest.fixture(scope="module")
def runner(blender_exe: Path) -> SubprocessBlenderRunner:
    return SubprocessBlenderRunner(
        blender_exe,
        default_entry_script(),
        threading.BoundedSemaphore(1),
        os.environ,
        poll_interval_s=0.2,
    )


def _prepare(runner: SubprocessBlenderRunner, work: Path) -> PrepareModelParams:
    """Real ``prepare_model`` with levels 0 and the source UVs kept (palette threshold lowered)."""
    params = PrepareModelParams(
        source_fbx=SOURCE_FBX,
        out_blend=work / "prepared.blend",
        out_static_glb=work / "static.glb",
        out_report=work / "prepare_report.json",
        geometry=GeometrySection(subdivision_levels=0),
        uv=UvSection(palette_texture_max_px=2),
        material_name="cube_rigged_mat",
        palette_hint_textures=[],
    )
    req = JobRequest(
        job="prepare_model",
        job_id=str(ULID()),
        model_id="cube_rigged",
        params=params.model_dump(mode="json"),
    )
    result = runner.run(req, timeout_s=300, log_path=work / "prepare.log", cancel=CancelToken())
    assert result.ok, result.error
    data = PrepareModelData.model_validate(result.data)
    assert data.uv.decision == "kept"
    return params


def _textured_source(blender_exe: Path, out_glb: Path) -> Path:
    cmd = [
        str(blender_exe),
        "-b",
        "--factory-startup",
        "--python-exit-code",
        "2",
        "--python",
        str(FIXTURE_TOOL),
        "--",
        str(SOURCE_FBX),
        str(TEXTURE_PNG),
        str(out_glb),
        "--roughness",
        str(ROUGHNESS),
        "--metallic",
        str(METALLIC),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300, check=False)  # noqa: S603  # trusted local executable
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    assert out_glb.is_file()
    return out_glb


def _bake(
    runner: SubprocessBlenderRunner, work: Path, prepared: PrepareModelParams, source_glb: Path
) -> tuple[BakeTexturesData, JobResult, BakeTexturesParams]:
    report = PrepareModelData.model_validate_json(prepared.out_report.read_text())
    params = BakeTexturesParams(
        prepared_blend=prepared.out_blend,
        source_glb=source_glb,
        out_dir=work / "textures" / "raw",
        bake=BakeSection(
            resolution=RESOLUTION,
            samples=SAMPLES,
            margin_px=MARGIN_PX,
            maps=["basecolor", "normal", "roughness", "metallic", "occlusion", "emission"],
        ),
        bbox_diagonal=report.bbox_diagonal,
    )
    req = JobRequest(
        job="bake_textures",
        job_id=str(ULID()),
        model_id="cube_rigged",
        params=params.model_dump(mode="json"),
    )
    result = runner.run(req, timeout_s=300, log_path=work / "bake.log", cancel=CancelToken())
    assert result.ok, result.error
    return BakeTexturesData.model_validate(result.data), result, params


def _rgb(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"), dtype=np.int32)


def _coverage(roughness: np.ndarray) -> np.ndarray:
    """Texels the bake wrote (unwritten texels stay black after ``use_clear``)."""
    return roughness.max(axis=-1) > 0


def _expected_basecolor() -> np.ndarray:
    with Image.open(TEXTURE_PNG) as img:
        upscaled = img.convert("RGB").resize((RESOLUTION, RESOLUTION), Image.Resampling.NEAREST)
    return np.asarray(upscaled, dtype=np.int32)


def _away_from_seams() -> np.ndarray:
    axis = np.arange(RESOLUTION)
    clear = np.abs(axis - RESOLUTION / 2) > SEAM_BAND_PX
    return clear[:, None] & clear[None, :]


# --------------------------------------------------------------------------- test
def test_bake_textured_cube_onto_itself(
    runner: SubprocessBlenderRunner, blender_exe: Path, tmp_path: Path
) -> None:
    started = time.monotonic()
    work = tmp_path / "work"
    prepared = _prepare(runner, work)
    source_glb = _textured_source(blender_exe, tmp_path / "source" / "textured_cube.glb")

    data, result, params = _bake(runner, work, prepared, source_glb)
    elapsed = time.monotonic() - started

    assert sorted(data.maps) == ["basecolor", "metallic", "normal", "roughness"]
    assert sorted(data.bake_seconds) == sorted(data.maps)
    assert all(seconds > 0.0 for seconds in data.bake_seconds.values())
    assert data.resolution == RESOLUTION
    assert data.samples == SAMPLES
    assert sorted(result.warnings) == [
        "emission: source material has no Emission Color data; skipped",
        "occlusion: source material has no Occlusion data; skipped",
    ]
    for name, path in data.maps.items():
        assert path == params.out_dir / f"{name}.png"
        assert path.is_file(), name
        with Image.open(path) as img:
            assert img.size == (RESOLUTION, RESOLUTION), name

    roughness = _rgb(data.maps["roughness"])
    covered = _coverage(roughness)
    assert covered.mean() > 0.3  # the cube cross fills a good part of the atlas
    assert abs(roughness[covered].mean() - ROUGHNESS * 255) <= TOLERANCE
    assert roughness[covered].std() <= 1.0

    metallic = _rgb(data.maps["metallic"])
    assert abs(metallic[covered].mean() - METALLIC * 255) <= TOLERANCE  # R-14: exact data values
    assert metallic[covered].std() <= 1.0

    normal = _rgb(data.maps["normal"])
    mean_normal = normal[covered].mean(axis=0)
    assert np.all(np.abs(mean_normal - np.array([128, 128, 255])) <= TOLERANCE), mean_normal

    basecolor = _rgb(data.maps["basecolor"])
    mask = covered & _away_from_seams()
    error = np.abs(basecolor - _expected_basecolor())[mask]
    assert error.mean() < 2.0, error.mean()
    assert (error.max(axis=-1) <= TOLERANCE).mean() > 0.98

    assert not (work / "prepared.blend.tmp").exists()
    log = (work / "bake.log").read_text()
    assert "[entry] job=bake_textures" in log
    assert "[entry] ok=True exit=0" in log
    assert elapsed < 120.0, elapsed
