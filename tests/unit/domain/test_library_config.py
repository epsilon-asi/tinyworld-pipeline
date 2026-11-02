from __future__ import annotations

import pytest
from pydantic import ValidationError

from grounded_pipeline.domain.library_config import (
    SECTION_MODELS,
    BakeSection,
    GeometrySection,
    LibraryConfig,
    TripoSection,
    UvSection,
)


def test_defaults_match_plan_05_section_3() -> None:
    cfg = LibraryConfig()

    assert cfg.library.concurrency == 2
    assert cfg.blender.job_timeout_s == 1800
    assert cfg.geometry.subdivision_levels == 1
    assert cfg.geometry.shapekeys == "fail"
    assert cfg.uv.palette_texture_max_px == 128
    assert cfg.texture.model_version == "v3.5-20260815"
    assert cfg.texture.prompt_mode == "text+style"
    assert cfg.texture.seed == 1337
    assert cfg.verify.route == "auto"
    assert cfg.bake.resolution == 2048
    assert cfg.bake.maps == [
        "basecolor",
        "normal",
        "metallic",
        "roughness",
        "occlusion",
        "emission",
    ]
    assert cfg.export.fbx_scale_option == "FBX_SCALE_ALL"
    assert cfg.unity.shader == "Universal Render Pipeline/Lit"
    assert str(cfg.tripo.base_url) == "https://openapi.tripo3d.ai/v3"
    assert cfg.tripo.credit_table == {
        "fast": 10,
        "standard": 10,
        "detailed": 20,
        "extreme": 30,
        "style_image": 0,
    }
    assert cfg.logging.level == "INFO"


def test_section_models_cover_every_config_field() -> None:
    assert set(SECTION_MODELS) == set(LibraryConfig.model_fields) - {"schema_version"}


def test_unknown_key_is_rejected() -> None:
    with pytest.raises(ValidationError, match="extra_forbidden"):
        GeometrySection.model_validate({"subdivision_level": 2})


def test_models_are_frozen() -> None:
    cfg = LibraryConfig()
    with pytest.raises(ValidationError):
        cfg.geometry.subdivision_levels = 3  # type: ignore[misc]


@pytest.mark.parametrize(
    ("model", "data"),
    [
        (GeometrySection, {"subdivision_levels": 4}),
        (GeometrySection, {"subdivision_type": "loop"}),
        (GeometrySection, {"max_triangles": 10}),
        (UvSection, {"smart_project_angle_deg": 90}),
        (UvSection, {"overlap_ratio_threshold": 1.5}),
        (BakeSection, {"resolution": 3000}),
        (BakeSection, {"maps": ["albedo"]}),
        (TripoSection, {"base_url": "not a url"}),
        (TripoSection, {"poll_interval_s": 0.1}),
        (TripoSection, {"use_presign_above_mb": 0}),
    ],
)
def test_out_of_range_values_are_rejected(model: type[object], data: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        model.model_validate(data)  # type: ignore[attr-defined]


def test_default_mutable_fields_are_not_shared_between_instances() -> None:
    a, b = BakeSection(), BakeSection()
    assert a.maps == b.maps
    assert a.maps is not b.maps


def test_json_round_trip_is_stable() -> None:
    cfg = LibraryConfig.model_validate({"texture": {"quality": "detailed"}, "bake": {"samples": 4}})
    again = LibraryConfig.model_validate_json(cfg.model_dump_json())
    assert again == cfg
    assert again.texture.quality == "detailed"
