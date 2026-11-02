"""``gp validate``: exit 0 (clean / warnings) and 3 (errors), Blender check integration."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from tinyworld_pipeline.blender.locate import BlenderInfo
from tinyworld_pipeline.cli.wiring import W_BLENDER_VERSION
from tinyworld_pipeline.domain.errors import BlenderNotFoundError
from tinyworld_pipeline.domain.library_config import BlenderSection

from .conftest import Harness, MakeHarness


def test_validate_clean_library(harness: Harness, lib: Path) -> None:
    result, document = harness.json("validate", "--library", str(lib))
    assert result.exit_code == 0
    assert "library is valid" in harness.text
    assert [m["model_id"] for m in document["models"]] == ["cube_palette", "cube_rigged"]
    assert all(m["status"] == "ok" for m in document["models"])
    assert document["problems"] == []


def test_validate_selected_model_only(harness: Harness, lib: Path) -> None:
    _, document = harness.json("validate", "--library", str(lib), "--model", "cube_rigged")
    assert [m["model_id"] for m in document["models"]] == ["cube_rigged"]


def test_validate_error_exits_3_with_table(harness: Harness, lib: Path) -> None:
    (lib / "models" / "cube_palette" / "source" / "cube_palette.fbx").unlink()
    result, document = harness.json("validate", "--library", str(lib))
    assert result.exit_code == 3
    assert document["error"]["code"] == "E_VALIDATION"
    palette = next(m for m in document["models"] if m["model_id"] == "cube_palette")
    assert palette["status"] == "error"
    assert palette["problems"][0]["code"] == "E_SOURCE_MISSING"
    assert "E_SOURCE_MISSING" in harness.text
    assert "error" in harness.text


def test_validate_unknown_model_is_usage_error(harness: Harness, lib: Path) -> None:
    result = harness.invoke("validate", "--library", str(lib), "--model", "ghost")
    assert result.exit_code == 2
    assert "unknown model id" in harness.text


def test_validate_reports_missing_blender_as_error(make_harness: MakeHarness, lib: Path) -> None:
    def missing(
        explicit: Path | None, env: Mapping[str, str], section: BlenderSection
    ) -> BlenderInfo:
        raise BlenderNotFoundError("No Blender executable found.", hint="install 4.5 LTS")

    harness = make_harness(resolver=missing)
    result, document = harness.json("validate", "--library", str(lib))
    assert result.exit_code == 3
    codes = [p["code"] for p in document["problems"]]
    assert "E_BLENDER_NOT_FOUND" in codes
    library_row = next(m for m in document["models"] if m["model_id"] is None)
    assert library_row["status"] == "error"


def test_validate_tolerated_blender_version_is_warning(
    make_harness: MakeHarness, lib: Path
) -> None:
    def tolerated(
        explicit: Path | None, env: Mapping[str, str], section: BlenderSection
    ) -> BlenderInfo:
        return BlenderInfo(path=Path("/fake/blender"), version=(4, 3, 2))

    harness = make_harness(resolver=tolerated)
    result, document = harness.json("validate", "--library", str(lib))
    assert result.exit_code == 0
    assert [p["code"] for p in document["problems"]] == [W_BLENDER_VERSION]
    assert "W_BLENDER_VERSION" in harness.text
    assert "warning" in harness.text
