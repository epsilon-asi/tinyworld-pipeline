"""Build a textured source GLB for the ``bake_textures`` integration test (plan 11 §8).

Runs **inside** Blender (stdlib + ``bpy`` only)::

    /Applications/Blender.app/Contents/MacOS/Blender -b --factory-startup --python-exit-code 2 \\
        --python tools/make_bake_fixture.py -- <source.fbx> <texture.png> <out.glb> \\
        [--roughness 0.5] [--metallic 0.25]

The FBX is imported with the pipeline's ``FBX_IMPORT_KWARGS`` so the mesh lands exactly where
``prepare_model`` puts it; every mesh is shaded smooth (as ``prepare`` does), its armature
binding is removed, and it gets one Principled material: ``texture.png`` as base colour sampled
with ``Closest`` interpolation (so every baked texel is one of the texture's colours), a flat
tangent normal map (so the source *has* a normal input and bakes to ``(128, 128, 255)``), and
constant roughness/metallic. The result is exported with ``GLB_EXPORT_KWARGS`` plus materials and
images, i.e. the same geometry and UVs as ``static.glb`` but textured, standing in for a provider
result that kept our UVs.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import bpy

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tinyworld_pipeline.blender_jobs.fbx_settings import (
    FBX_IMPORT_KWARGS,
    GLB_EXPORT_KWARGS,
)

log = logging.getLogger("make_bake_fixture")

MATERIAL_NAME = "bake_source_mat"
FLAT_NORMAL = (0.5, 0.5, 1.0, 1.0)
FLAT_NORMAL_SIZE = 4


def _parse(argv: list[str]) -> argparse.Namespace:
    args = argv[argv.index("--") + 1 :] if "--" in argv else argv[1:]
    parser = argparse.ArgumentParser(prog="make_bake_fixture.py")
    parser.add_argument("source_fbx", type=Path)
    parser.add_argument("texture_png", type=Path)
    parser.add_argument("out_glb", type=Path)
    parser.add_argument("--roughness", type=float, default=0.5)
    parser.add_argument("--metallic", type=float, default=0.25)
    return parser.parse_args(args)


def _supported_kwargs(op: Any, kwargs: dict[str, Any]) -> dict[str, Any]:  # noqa: ANN401
    known = {prop.identifier for prop in op.get_rna_type().properties}
    return {key: value for key, value in kwargs.items() if key in known}


def _socket(node: bpy.types.Node, name: str, *, output: bool = False) -> bpy.types.NodeSocket:
    sockets = node.outputs if output else node.inputs
    assert sockets is not None  # shader nodes always have socket collections
    return sockets[name]


def _image_node(
    tree: bpy.types.ShaderNodeTree, image: bpy.types.Image, colorspace: str
) -> bpy.types.ShaderNodeTexImage:
    node = tree.nodes.new("ShaderNodeTexImage")
    assert isinstance(node, bpy.types.ShaderNodeTexImage)
    settings = image.colorspace_settings
    assert settings is not None
    settings.name = colorspace  # type: ignore[assignment]  # colour-space names are runtime enums
    node.image = image
    node.interpolation = "Closest"
    return node


def _flat_normal_image(out_dir: Path) -> bpy.types.Image:
    """A tiny constant ``(0.5, 0.5, 1)`` normal map saved beside the GLB (exporter needs a file)."""
    image = bpy.data.images.new("flat_normal", FLAT_NORMAL_SIZE, FLAT_NORMAL_SIZE, alpha=False)
    image.pixels = list(FLAT_NORMAL) * (FLAT_NORMAL_SIZE * FLAT_NORMAL_SIZE)  # type: ignore[assignment]  # stubs type pixels as float
    image.filepath_raw = str(out_dir / "flat_normal.png")
    image.file_format = "PNG"
    image.save()
    return image


def _material(
    texture: Path, out_dir: Path, roughness: float, metallic: float
) -> bpy.types.Material:
    material = bpy.data.materials.new(MATERIAL_NAME)
    material.use_nodes = True
    tree = material.node_tree
    assert tree is not None
    tree.nodes.clear()
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    bsdf = tree.nodes.new("ShaderNodeBsdfPrincipled")
    tree.links.new(_socket(bsdf, "BSDF", output=True), _socket(output, "Surface"))
    base = _image_node(tree, bpy.data.images.load(str(texture)), "sRGB")
    tree.links.new(_socket(base, "Color", output=True), _socket(bsdf, "Base Color"))
    normal_tex = _image_node(tree, _flat_normal_image(out_dir), "Non-Color")
    normal_map = tree.nodes.new("ShaderNodeNormalMap")
    tree.links.new(_socket(normal_tex, "Color", output=True), _socket(normal_map, "Color"))
    tree.links.new(_socket(normal_map, "Normal", output=True), _socket(bsdf, "Normal"))
    _socket(bsdf, "Roughness").default_value = roughness  # type: ignore[attr-defined]  # NodeSocketFloat
    _socket(bsdf, "Metallic").default_value = metallic  # type: ignore[attr-defined]  # NodeSocketFloat
    return material


def build(
    source_fbx: Path, texture: Path, out_glb: Path, roughness: float, metallic: float
) -> None:
    """Import, retexture and export (see the module docstring)."""
    out_glb.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=str(source_fbx), **FBX_IMPORT_KWARGS)
    meshes = [obj for obj in bpy.data.objects if obj.type == "MESH"]
    if not meshes:
        raise RuntimeError(f"{source_fbx} contains no mesh")
    for armature in (obj for obj in bpy.data.objects if obj.type == "ARMATURE"):
        data = armature.data
        assert isinstance(data, bpy.types.Armature)
        data.pose_position = "REST"
    material = _material(texture, out_glb.parent, roughness, metallic)
    for obj in meshes:
        mesh = obj.data
        assert isinstance(mesh, bpy.types.Mesh)
        mesh.polygons.foreach_set("use_smooth", [True] * len(mesh.polygons))
        mesh.update()
        for modifier in list(obj.modifiers):
            if modifier.type == "ARMATURE":
                obj.modifiers.remove(modifier)
        mesh.materials.clear()
        mesh.materials.append(material)
    for obj in bpy.data.objects:
        obj.select_set(obj in meshes)
    view_layer = bpy.context.view_layer
    assert view_layer is not None
    view_layer.objects.active = meshes[0]
    kwargs = {
        **GLB_EXPORT_KWARGS,
        "export_materials": "EXPORT",
        "export_image_format": "AUTO",
        "export_texcoords": True,
    }
    with bpy.context.temp_override(selected_objects=meshes):
        bpy.ops.export_scene.gltf(
            filepath=str(out_glb), **_supported_kwargs(bpy.ops.export_scene.gltf, kwargs)
        )
    log.info("wrote %s", out_glb)


def main(argv: list[str]) -> int:
    """CLI entry point; returns the process exit code."""
    logging.basicConfig(level=logging.INFO)
    ns = _parse(argv)
    build(ns.source_fbx, ns.texture_png, ns.out_glb, ns.roughness, ns.metallic)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
