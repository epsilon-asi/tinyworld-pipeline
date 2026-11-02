from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from fakes.glb_builder import MaterialSpec, cube, textured_set, write_glb
from tinyworld_pipeline.domain.errors import StagePrereqError
from tinyworld_pipeline.domain.textures import RawTextureSet, TextureFactors
from tinyworld_pipeline.pipeline.context import ModelContext, PipelineContext
from tinyworld_pipeline.pipeline.stages.apply import (
    APPLY_REPORT,
    PREPARED_BLEND,
    RAW_DIR,
    URP_DIR,
    ApplyStage,
    bake_route_not_implemented,
    read_route,
)
from tinyworld_pipeline.pipeline.stages.apply_bake import run_bake_route
from tinyworld_pipeline.pipeline.stages.verify import RESULT_GLB, VERIFY_REPORT


def _set_route(folder: Path, route: str) -> None:
    (folder / VERIFY_REPORT).parent.mkdir(parents=True, exist_ok=True)
    (folder / VERIFY_REPORT).write_text(json.dumps({"route": route}), encoding="utf-8")


@pytest.fixture
def extractable(rigged: ModelContext) -> ModelContext:
    images, material = textured_set()
    write_glb(rigged.paths.folder / RESULT_GLB, cube(images=images, materials=[material]))
    _set_route(rigged.paths.folder, "extract")
    return rigged


def test_extract_route_packs_urp_and_writes_report(
    extractable: ModelContext, ctx: PipelineContext
) -> None:
    folder = extractable.paths.folder
    stale = folder / URP_DIR / "stale.png"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"x")

    outcome = ApplyStage().run(extractable, ctx)

    urp = folder / URP_DIR
    names = sorted(p.name for p in urp.iterdir())
    assert names == [
        "cube_rigged_basecolor.png",
        "cube_rigged_emission.png",
        "cube_rigged_metallic_smoothness.png",
        "cube_rigged_normal.png",
        "cube_rigged_occlusion.png",
    ]
    assert outcome.outputs == [*(f"work/textures/urp/{n}" for n in names), "work/apply_report.json"]
    assert outcome.extra == {"route": "extract", "fallback_to_bake": False, "resolution": [2, 2]}
    report = json.loads((folder / APPLY_REPORT).read_text(encoding="utf-8"))
    assert report["route"] == "extract"
    assert report["fallback_to_bake"] is False
    assert report["raw"]["basecolor"] == "work/textures/raw/basecolor.png"
    assert report["urp"]["basecolor"] == "work/textures/urp/cube_rigged_basecolor.png"
    assert report["factors"]["base_color"] == [1.0, 1.0, 1.0, 1.0]
    assert report["resolution"] == [2, 2]
    assert report["source"] == "extract"
    assert report["material"]["bump_scale"] == 1.0
    with Image.open(urp / "cube_rigged_basecolor.png") as img:
        assert img.size == (2, 2)


def test_multi_material_falls_back_to_injected_bake_route(
    rigged: ModelContext, ctx: PipelineContext
) -> None:
    folder = rigged.paths.folder
    images, material = textured_set()
    other = MaterialSpec(name="other", base_color_image=1)
    write_glb(folder / RESULT_GLB, cube(primitives=2, images=images, materials=[material, other]))
    _set_route(folder, "extract")
    calls: list[str] = []

    def fake_bake(model: ModelContext, _ctx: PipelineContext) -> RawTextureSet:
        calls.append(model.model_id)
        raw_dir = model.paths.folder / RAW_DIR
        assert raw_dir.is_dir()
        assert not any(raw_dir.iterdir())  # cleared before baking
        path = raw_dir / "basecolor.png"
        Image.new("RGBA", (4, 4), (10, 20, 30, 255)).save(path)
        return RawTextureSet(
            basecolor=path,
            metallic_roughness=None,
            normal=None,
            occlusion=None,
            emissive=None,
            factors=TextureFactors(),
            source="bake",
        )

    outcome = ApplyStage(bake_route=fake_bake).run(rigged, ctx)

    assert calls == ["cube_rigged"]
    assert outcome.extra == {"route": "bake", "fallback_to_bake": True, "resolution": [4, 4]}
    assert outcome.outputs == [
        "work/textures/urp/cube_rigged_basecolor.png",
        "work/apply_report.json",
    ]
    report = json.loads((folder / APPLY_REPORT).read_text(encoding="utf-8"))
    assert report["fallback_to_bake"] is True
    assert report["source"] == "bake"


def test_bake_route_default_is_the_wp08_implementation(
    rigged: ModelContext, ctx: PipelineContext
) -> None:
    folder = rigged.paths.folder
    write_glb(folder / RESULT_GLB, cube())
    _set_route(folder, "bake")
    assert ApplyStage().bake_route is run_bake_route
    stage = ApplyStage(bake_route=bake_route_not_implemented)  # the seam is still injectable
    with pytest.raises(NotImplementedError, match="WP-08"):
        stage.run(rigged, ctx)


def test_fingerprint_depends_on_route_and_bake_inputs(
    extractable: ModelContext, ctx: PipelineContext
) -> None:
    folder = extractable.paths.folder
    stage = ApplyStage()
    extract_fp = stage.input_fingerprint(extractable, ctx)
    blend = folder / PREPARED_BLEND
    blend.write_bytes(b"BLENDER-v405")
    assert stage.input_fingerprint(extractable, ctx) == extract_fp  # blend irrelevant on extract
    _set_route(folder, "bake")
    bake_fp = stage.input_fingerprint(extractable, ctx)
    assert bake_fp != extract_fp
    blend.write_bytes(b"BLENDER-v405-changed")
    assert stage.input_fingerprint(extractable, ctx) != bake_fp
    write_glb(folder / RESULT_GLB, cube(scale=3.0))
    _set_route(folder, "extract")
    assert stage.input_fingerprint(extractable, ctx) != extract_fp


def test_missing_or_invalid_verify_report_is_prereq_error(
    rigged: ModelContext, ctx: PipelineContext
) -> None:
    folder = rigged.paths.folder
    write_glb(folder / RESULT_GLB, cube())
    stage = ApplyStage()
    with pytest.raises(StagePrereqError, match="missing"):
        stage.input_fingerprint(rigged, ctx)
    with pytest.raises(StagePrereqError):
        stage.run(rigged, ctx)
    _set_route(folder, "teleport")
    with pytest.raises(StagePrereqError, match="no valid route"):
        read_route(folder)
    (folder / VERIFY_REPORT).write_text("[]", encoding="utf-8")
    with pytest.raises(StagePrereqError, match="no valid route"):
        read_route(folder)
