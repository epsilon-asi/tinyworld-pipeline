"""Structural checks on the Unity package (plan 12 §4): JSON parses, C# files are well-formed."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
PACKAGE = ROOT / "unity" / "com.studio.grounded-pipeline"
EDITOR = PACKAGE / "Editor"
TESTS = PACKAGE / "Tests" / "Editor"
GOLDEN = ROOT / "tests" / "fixtures" / "manifest_golden.json"
CS_FILES = sorted(PACKAGE.rglob("*.cs"))
EXPECTED_EDITOR_FILES = {
    "GroundedPipeline.Editor.asmdef",
    "GroundedPipelinePostprocessor.cs",
    "PipelineManifest.cs",
    "ManifestLocator.cs",
    "UrpMaterialBuilder.cs",
    "TextureImportRules.cs",
    "PipelineMenu.cs",
}


def strip_code(source: str) -> str:
    """Remove comments and string/char literals so braces inside them do not count."""
    out: list[str] = []
    i = 0
    n = len(source)
    while i < n:
        two = source[i : i + 2]
        if two == "//":
            i = source.find("\n", i)
            i = n if i < 0 else i
        elif two == "/*":
            end = source.find("*/", i + 2)
            i = n if end < 0 else end + 2
        elif two == '@"':
            i += 2
            while i < n:
                if source[i] == '"':
                    if source[i : i + 2] == '""':
                        i += 2
                        continue
                    i += 1
                    break
                i += 1
        elif source[i] == '"' or source[i] == "'":
            quote = source[i]
            i += 1
            while i < n and source[i] != quote:
                i += 2 if source[i] == "\\" else 1
            i += 1
        else:
            out.append(source[i])
            i += 1
    return "".join(out)


# ------------------------------------------------------------------------------ package.json
def test_package_json_matches_plan_12_4_3() -> None:
    doc = json.loads((PACKAGE / "package.json").read_text(encoding="utf-8"))
    assert doc["name"] == "com.studio.grounded-pipeline"
    assert doc["version"] == "0.1.0"
    assert doc["displayName"] == "Grounded Pipeline Importer"
    assert doc["unity"] == "6000.0"
    assert doc["dependencies"] == {"com.unity.render-pipelines.universal": "17.0.0"}


def test_expected_files_exist() -> None:
    assert {
        p.name for p in EDITOR.iterdir() if p.suffix in {".cs", ".asmdef"}
    } == EXPECTED_EDITOR_FILES
    assert (TESTS / "GroundedPipeline.Editor.Tests.asmdef").exists()
    assert (TESTS / "ManifestLoadTests.cs").exists()
    assert (PACKAGE / "README.md").exists()


# ------------------------------------------------------------------------------ asmdefs
def test_editor_asmdef_is_editor_only_and_references_urp() -> None:
    doc = json.loads((EDITOR / "GroundedPipeline.Editor.asmdef").read_text(encoding="utf-8"))
    assert doc["name"] == "GroundedPipeline.Editor"
    assert doc["includePlatforms"] == ["Editor"]
    assert {
        "Unity.RenderPipelines.Universal.Runtime",
        "Unity.RenderPipelines.Universal.Editor",
    } <= set(doc["references"])
    defines = {d["define"]: d["name"] for d in doc.get("versionDefines", [])}
    assert defines == {"GP_URP": "com.unity.render-pipelines.universal"}


def test_tests_asmdef_references_editor_assembly_and_test_runner() -> None:
    doc = json.loads((TESTS / "GroundedPipeline.Editor.Tests.asmdef").read_text(encoding="utf-8"))
    assert doc["includePlatforms"] == ["Editor"]
    assert {"GroundedPipeline.Editor", "UnityEditor.TestRunner", "UnityEngine.TestRunner"} <= set(
        doc["references"]
    )
    assert "nunit.framework.dll" in doc["precompiledReferences"]
    assert "UNITY_INCLUDE_TESTS" in doc["defineConstraints"]


# ------------------------------------------------------------------------------ C# files
@pytest.mark.parametrize("cs_file", CS_FILES, ids=[p.name for p in CS_FILES])
def test_cs_file_declares_namespace_and_is_brace_balanced(cs_file: Path) -> None:
    source = cs_file.read_text(encoding="utf-8")
    expected_ns = (
        "GroundedPipeline.Editor.Tests" if cs_file.parent == TESTS else "GroundedPipeline.Editor"
    )
    assert re.search(rf"^namespace {re.escape(expected_ns)}\s*$", source, re.MULTILINE), cs_file
    assert "#if UNITY_EDITOR" not in source  # Editor asmdef already restricts the platform
    code = strip_code(source)
    for open_, close in ("{}", "()", "[]"):
        assert code.count(open_) == code.count(close), f"{cs_file.name}: unbalanced {open_}{close}"
    assert '"' not in code  # every literal was consumed by strip_code
    assert "'" not in code


def test_postprocessor_implements_plan_12_4_2() -> None:
    source = (EDITOR / "GroundedPipelinePostprocessor.cs").read_text(encoding="utf-8")
    for needle in (
        "AssetPostprocessor",
        "GetPostprocessOrder() => 100",
        "OnPreprocessTexture()",
        "OnPreprocessModel()",
        "OnPostprocessModel(GameObject",
        "OnPostprocessAllAssets(",
        "ModelImporterAnimationType.Generic",
        "ModelImporterMaterialImportMode.None",
        "ModelImporterNormals.Import",
        "importBlendShapes = false",
        "bakeAxisConversion = false",
        "useFileScale = true",
        "DependsOnSourceAsset(",
        "DependsOnArtifact(",
        "ModelImporterAvatarSetup.CopyFromOther",
        "ModelImporterAvatarSetup.CreateFromThisModel",
        "ImportAssetOptions.ForceUpdate",
        '$"[GroundedPipeline] {path}: {exception.Message}"',
    ):
        assert needle in source, needle
    # Every Unity callback body is protected.
    assert source.count("catch (Exception exception)") >= 4


def test_texture_rules_and_material_builder_follow_appendix_c() -> None:
    rules = (EDITOR / "TextureImportRules.cs").read_text(encoding="utf-8")
    for needle in (
        "TextureImporterType.NormalMap",
        "sRGBTexture = texture.srgb",
        "mipmapEnabled = true",
        "alphaIsTransparency = false",
        "TextureImporterCompression.Compressed",
        "TextureImporterAlphaSource.FromInput",
    ):
        assert needle in rules, needle
    builder = (EDITOR / "UrpMaterialBuilder.cs").read_text(encoding="utf-8")
    for needle in (
        "Shader.Find(material.shader)",
        "SetTexture(texture.slot",
        "SetFloat(entry.name, entry.value)",
        "SetColor(color.name",
        "EnableKeyword(keyword)",
        "MaterialGlobalIlluminationFlags.RealtimeEmissive",
        "AssetDatabase.SaveAssetIfDirty",
        "AssetDatabase.CreateAsset",
    ):
        assert needle in builder, needle


def test_menu_has_all_four_entries() -> None:
    menu = (EDITOR / "PipelineMenu.cs").read_text(encoding="utf-8")
    items = set(re.findall(r'\[MenuItem\(Root \+ "([^"]+)"\)\]', menu))
    assert items == {
        "Reimport All Pipeline Models",
        "Validate Manifests",
        "Open Log",
        "Dump Clip Paths",
    }
    assert "AnimationUtility.GetCurveBindings(clip)" in menu


def test_embedded_golden_json_in_cs_test_matches_fixture() -> None:
    source = (TESTS / "ManifestLoadTests.cs").read_text(encoding="utf-8")
    match = re.search(r'GoldenJson = @"(.*?)";\s*\n\s*// GOLDEN-END', source, re.DOTALL)
    assert match is not None, "GOLDEN-BEGIN/END block missing"
    embedded = match.group(1).replace('""', '"') + "\n"
    assert embedded == GOLDEN.read_text(encoding="utf-8")
