from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from tinyworld_pipeline.domain.model_spec import (
    GuidanceImages,
    ModelDocument,
    ModelFrontMatter,
    PromptSpec,
    SourceSpec,
    TextureOverrides,
)

PLAN_EXAMPLE = {
    "schema_version": 1,
    "display_name": "Worker Ant",
    "source": {
        "primary": "source/Ant.fbx",
        "animations": ["source/animations/*.fbx"],
        "license": "CC0 — Quaternius",
    },
    "geometry": {"subdivision_levels": 1, "shapekeys": "drop"},
    "uv": {"mode": "auto"},
    "texture": {
        "seed": 4242,
        "quality": "detailed",
        "prompt_mode": "text+style",
        "guidance_images": {
            "front": "references/front.png",
            "left": "references/left.png",
            "back": "references/back.png",
            "right": "references/right.png",
        },
    },
    "prompt": {
        "include_sections": ["base_prompt", "material_notes"],
        "template": "{base_prompt}\n\n{model_prompt}\n\n{material_notes}",
    },
    "unity": {"source_asset_path": "Assets/ThirdParty/Quaternius/Ant.fbx"},
    "tags": ["insect", "enemy", "tier1"],
}


def test_plan_example_front_matter_parses() -> None:
    front = ModelFrontMatter.model_validate(PLAN_EXAMPLE)

    assert front.display_name == "Worker Ant"
    assert front.source.primary == Path("source/Ant.fbx")
    assert front.geometry is not None
    assert front.geometry.shapekeys == "drop"
    assert front.geometry.model_fields_set == {"subdivision_levels", "shapekeys"}
    assert front.texture is not None
    assert front.texture.guidance_images == GuidanceImages(
        front=Path("references/front.png"),
        left=Path("references/left.png"),
        back=Path("references/back.png"),
        right=Path("references/right.png"),
    )
    assert front.unity is not None
    assert front.unity.source_asset_path == "Assets/ThirdParty/Quaternius/Ant.fbx"
    assert front.bake is None


def test_minimal_front_matter_only_needs_display_name() -> None:
    front = ModelFrontMatter(display_name="Bee")

    assert front.source == SourceSpec()
    assert front.source.animations == ["source/animations/*.fbx"]
    assert front.prompt == PromptSpec()
    assert front.prompt.template == "{base_prompt}\n\n{model_prompt}"
    assert front.tags == []


@pytest.mark.parametrize(
    "data",
    [
        {"display_name": "x", "textur": {"seed": 1}},
        {"display_name": "x", "texture": {"qualty": "fast"}},
        {"display_name": "x", "geometry": {"levels": 2}},
        {"display_name": "x", "schema_version": 2},
    ],
)
def test_unknown_or_wrong_keys_fail(data: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        ModelFrontMatter.model_validate(data)


def test_override_constraints_mirror_sections() -> None:
    with pytest.raises(ValidationError):
        TextureOverrides(seed=-1)
    with pytest.raises(ValidationError):
        GuidanceImages.model_validate({"left": "a.png"})  # front is required


def test_model_document_requires_valid_model_id() -> None:
    front = ModelFrontMatter(display_name="x")
    ModelDocument(
        model_id="ant_worker", folder=Path("/lib/models/ant_worker"), front=front, sections={}
    )
    with pytest.raises(ValidationError):
        ModelDocument(model_id="Ant-Worker", folder=Path("/x"), front=front, sections={})
