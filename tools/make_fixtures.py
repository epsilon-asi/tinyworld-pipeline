"""Regenerate the committed FBX fixtures in ``tests/fixtures/fbx`` (plan ``15`` §5).

Runs **inside** Blender (stdlib + ``bpy``/``bmesh`` only)::

    /Applications/Blender.app/Contents/MacOS/Blender -b --factory-startup \\
        --python tools/make_fixtures.py -- tests/fixtures/fbx [--check]

Fixtures (all share one deterministically built rig ``Armature`` = ``Root`` -> ``Spine`` ->
``Head``, see ``tests/fixtures/fbx/README.md``):

- ``rigged_cube.fbx``      subdivided cube, skinned, default UVs, ``cube_mat`` ->
                           ``cube_texture.png``, 24-frame action ``idle``
- ``rigged_cube@walk.fbx`` armature only, 24-frame action ``walk`` (companion clip)
- ``palette_cube.fbx``     same cube, all UVs collapsed onto swatches of ``palette.png`` (16x4),
                           fully overlapping per cube side, ``palette_mat``
- ``shapekey_cube.fbx``    same cube plus shape key ``blink`` (top vertices pulled down)

Export uses the pipeline's ``FBX_EXPORT_KWARGS`` (plan ``12`` §1.2) with ``FBX_SCALE_ALL`` and
``path_mode="RELATIVE"`` / ``embed_textures=False`` so the FBX references the PNG beside it.

Determinism: geometry, weights, UVs and keys are built without randomness, but Blender's FBX
exporter derives element UUIDs from Python's ``hash()`` of strings (salted per process) and stamps
the creation time. Run with ``PYTHONHASHSEED=0`` and ``--python-use-system-env`` (Blender ignores
the environment otherwise) and only the three ``CreationTimeStamp`` bytes differ between runs.
``--check`` sidesteps bytes entirely: it regenerates into a temporary directory and compares the
structural summaries produced by ``tools/fixture_probe.py``.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
import tempfile
import zlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import bmesh
import bpy

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fixture_probe import summarize_many  # needs the sys.path entry above

# Plan 12 §1.2 with cfg.export.fbx_scale_option="FBX_SCALE_ALL" and embed_textures=False.
FBX_EXPORT_KWARGS: dict[str, Any] = {
    "use_selection": True,
    "object_types": {"ARMATURE", "MESH"},
    "use_mesh_modifiers": True,
    "mesh_smooth_type": "FACE",
    "use_tspace": False,
    "add_leaf_bones": False,
    "primary_bone_axis": "Y",
    "secondary_bone_axis": "X",
    "armature_nodetype": "NULL",
    "use_armature_deform_only": False,
    "bake_anim": True,
    "bake_anim_use_all_bones": True,
    "bake_anim_use_nla_strips": False,
    "bake_anim_use_all_actions": True,
    "bake_anim_force_startend_keying": True,
    "bake_anim_step": 1.0,
    "bake_anim_simplify_factor": 0.0,
    "apply_scale_options": "FBX_SCALE_ALL",
    "global_scale": 1.0,
    "apply_unit_scale": True,
    "use_space_transform": True,
    "bake_space_transform": False,
    "axis_forward": "-Z",
    "axis_up": "Y",
    "path_mode": "RELATIVE",
    "embed_textures": False,
    "use_custom_props": False,
    "use_metadata": True,
}

FIXTURE_NAMES = ("rigged_cube.fbx", "rigged_cube@walk.fbx", "palette_cube.fbx", "shapekey_cube.fbx")
CUBE_TEXTURE = "cube_texture.png"
PALETTE_TEXTURE = "palette.png"

BONE_LENGTH = 0.5
BONES = ("Root", "Spine", "Head")  # chain, bottom to top along +Z
FRAME_START, FRAME_END = 1, 24
Rgb = tuple[int, int, int]
Keys = dict[str, list[tuple[int, tuple[float, float, float]]]]  # bone -> [(frame, euler deg)]

IDLE_KEYS: Keys = {
    "Spine": [(1, (0.0, 0.0, 0.0)), (12, (5.0, 0.0, 0.0)), (24, (0.0, 0.0, 0.0))],
    "Head": [(1, (0.0, 0.0, 0.0)), (12, (0.0, 0.0, -8.0)), (24, (0.0, 0.0, 0.0))],
}
WALK_KEYS: Keys = {
    "Spine": [
        (1, (0.0, -10.0, 0.0)),
        (8, (0.0, 10.0, 0.0)),
        (16, (0.0, -10.0, 0.0)),
        (24, (0.0, 10.0, 0.0)),
    ],
    "Head": [(1, (15.0, 0.0, 0.0)), (12, (-15.0, 0.0, 0.0)), (24, (15.0, 0.0, 0.0))],
}


# --------------------------------------------------------------------------- textures


def _png_bytes(width: int, height: int, pixels: Sequence[Rgb]) -> bytes:
    """Minimal deterministic RGB8 PNG (no ancillary chunks, so no timestamps)."""

    def chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    rows = b"".join(
        b"\x00" + bytes(c for px in pixels[y * width : (y + 1) * width] for c in px)
        for y in range(height)
    )
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(rows, 9))
        + chunk(b"IEND", b"")
    )


def write_textures(out_dir: Path) -> None:
    """Write ``cube_texture.png`` (4x4, four colour quadrants) and ``palette.png`` (16x4, 64)."""
    quadrants: list[Rgb] = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    cube_px = [quadrants[(y // 2) * 2 + (x // 2)] for y in range(4) for x in range(4)]
    (out_dir / CUBE_TEXTURE).write_bytes(_png_bytes(4, 4, cube_px))
    palette_px: list[Rgb] = [(x * 17, y * 85, 255 - x * 17) for y in range(4) for x in range(16)]
    (out_dir / PALETTE_TEXTURE).write_bytes(_png_bytes(16, 4, palette_px))


# --------------------------------------------------------------------------- scene building


def _scene() -> bpy.types.Scene:
    scene = bpy.context.scene
    assert scene is not None  # always set in background mode
    return scene


def _view_layer() -> bpy.types.ViewLayer:
    view_layer = bpy.context.view_layer
    assert view_layer is not None  # always set in background mode
    return view_layer


def reset_scene() -> None:
    """Empty factory scene; also pins the frame range every fixture uses."""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = _scene()
    scene.frame_start, scene.frame_end = FRAME_START, FRAME_END
    scene.frame_set(FRAME_START)


def build_armature() -> bpy.types.Object:
    """``Armature`` object with the connected chain Root -> Spine -> Head, each 0.5 along +Z."""
    data = bpy.data.armatures.new("Armature")
    obj = bpy.data.objects.new("Armature", data)
    _scene().collection.objects.link(obj)
    _view_layer().objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    parent = None
    for i, name in enumerate(BONES):
        bone = data.edit_bones.new(name)
        bone.head = (0.0, 0.0, i * BONE_LENGTH)
        bone.tail = (0.0, 0.0, (i + 1) * BONE_LENGTH)
        bone.parent = parent
        bone.use_connect = parent is not None
        parent = bone
    bpy.ops.object.mode_set(mode="OBJECT")
    return obj


def build_cube_mesh(name: str) -> bpy.types.Mesh:
    """Unit cube subdivided once (26 verts / 24 quads), default cube UVs, z in [0.25, 1.25]."""
    bm = bmesh.new()
    bm.loops.layers.uv.new("UVMap")
    bmesh.ops.create_cube(bm, size=1.0, calc_uvs=True)
    bmesh.ops.subdivide_edges(bm, edges=list(bm.edges), cuts=1, use_grid_fill=True)
    bmesh.ops.translate(bm, verts=list(bm.verts), vec=(0.0, 0.0, 0.75))
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    return mesh


def _weights_for_height(z: float) -> dict[str, float]:
    """Lower ring -> Root, middle -> mostly Spine, upper -> mostly Head (sums to 1)."""
    if z < BONE_LENGTH:
        return {"Root": 1.0}
    if z < 2 * BONE_LENGTH:
        return {"Root": 0.25, "Spine": 0.75}
    return {"Spine": 0.25, "Head": 0.75}


def skin_to_armature(obj: bpy.types.Object, armature: bpy.types.Object) -> None:
    """Parent ``obj`` to ``armature`` with an Armature modifier and height-banded vertex groups."""
    mesh = obj.data
    assert isinstance(mesh, bpy.types.Mesh)
    groups = {name: obj.vertex_groups.new(name=name) for name in BONES}
    for vertex in mesh.vertices:
        for name, weight in _weights_for_height(vertex.co.z).items():
            groups[name].add([vertex.index], weight, "REPLACE")
    obj.parent = armature
    modifier = obj.modifiers.new("Armature", "ARMATURE")
    assert isinstance(modifier, bpy.types.ArmatureModifier)
    modifier.object = armature


def make_material(name: str, image_path: Path) -> bpy.types.Material:
    """Principled material whose Base Color is an Image Texture node reading ``image_path``."""
    image = bpy.data.images.load(str(image_path))  # 8-bit PNG -> sRGB by default
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    tree = material.node_tree
    assert tree is not None
    bsdf = tree.nodes["Principled BSDF"]
    tex = tree.nodes.new("ShaderNodeTexImage")
    assert isinstance(tex, bpy.types.ShaderNodeTexImage)
    tex.image = image
    tex.interpolation = "Closest"
    tex.location = (-400.0, 300.0)
    outputs = tex.outputs
    assert outputs is not None
    tree.links.new(outputs["Color"], bsdf.inputs["Base Color"])
    return material


def build_cube(
    name: str, armature: bpy.types.Object, material: bpy.types.Material
) -> bpy.types.Object:
    """Skinned, textured cube object linked to the scene."""
    mesh = build_cube_mesh(name)
    mesh.materials.append(material)
    obj = bpy.data.objects.new(name, mesh)
    _scene().collection.objects.link(obj)
    skin_to_armature(obj, armature)
    return obj


def _side_index(normal: Sequence[float]) -> int:
    """0..5 for +X, -X, +Y, -Y, +Z, -Z (dominant normal axis)."""
    axis = max(range(3), key=lambda i: abs(normal[i]))
    return axis * 2 + (0 if normal[axis] > 0 else 1)


def collapse_uvs_to_palette(obj: bpy.types.Object) -> None:
    """Map every face of a cube side onto one tiny quad in a palette swatch (overlap ratio 1.0)."""
    mesh = obj.data
    assert isinstance(mesh, bpy.types.Mesh)
    layer = mesh.uv_layers[0]
    half = 0.01  # swatch is 1/16 x 1/4 UV units; the quad stays well inside it
    for poly in mesh.polygons:
        side = _side_index(poly.normal.to_tuple())
        cu = (2 * side + 1 + 0.5) / 16.0
        cv = (side % 4 + 0.5) / 4.0
        corners = (
            (cu - half, cv - half),
            (cu + half, cv - half),
            (cu + half, cv + half),
            (cu - half, cv + half),
        )
        for k, loop_index in enumerate(poly.loop_indices):
            layer.data[loop_index].uv = corners[k % 4]


def add_blink_shape_key(obj: bpy.types.Object) -> None:
    """``Basis`` + ``blink`` (top vertex ring moved 0.2 down)."""
    obj.shape_key_add(name="Basis", from_mix=False)
    key = obj.shape_key_add(name="blink", from_mix=False)
    mesh = obj.data
    assert isinstance(mesh, bpy.types.Mesh)
    for vertex in mesh.vertices:
        if vertex.co.z > 1.0:
            key.data[vertex.index].co.z -= 0.2  # type: ignore[attr-defined]  # ShapeKeyPoint.co is a Vector


def add_action(armature: bpy.types.Object, name: str, keys: Keys) -> None:
    """Key XYZ-euler rotations on pose bones; the auto-created action is renamed to ``name``."""
    pose = armature.pose
    assert pose is not None
    for bone_name, frames in keys.items():
        pbone = pose.bones[bone_name]
        pbone.rotation_mode = "XYZ"
        for frame, degrees in frames:
            pbone.rotation_euler = tuple(math.radians(d) for d in degrees)
            pbone.keyframe_insert("rotation_euler", frame=frame)
        pbone.rotation_euler = (0.0, 0.0, 0.0)
    anim = armature.animation_data
    assert anim is not None
    assert anim.action is not None
    anim.action.name = name
    anim.action.use_fake_user = True
    # Unassign so the armature evaluates in rest pose at export time. The exporter writes bone Model
    # nodes from the *current pose* (only skin clusters use the rest pose), and the importer takes
    # the rest pose of an un-skinned armature from those nodes - a posed export would give the
    # mesh-less ``@walk`` clip a different rest pose than the model. ``bake_anim_use_all_actions``
    # still exports the unassigned action.
    anim.action = None
    _scene().frame_set(FRAME_START)


def export_fbx(path: Path, objects: Sequence[bpy.types.Object]) -> None:
    """Select exactly ``objects`` (armature active) and export with the pipeline kwargs."""
    bpy.ops.object.select_all(action="DESELECT")
    for obj in objects:
        obj.select_set(True)
    _view_layer().objects.active = objects[0]
    bpy.ops.export_scene.fbx(filepath=str(path), **FBX_EXPORT_KWARGS)


# --------------------------------------------------------------------------- fixtures


def make_fixture(out_dir: Path, name: str) -> Path:
    """Build and export one fixture by file name into ``out_dir``."""
    reset_scene()
    armature = build_armature()
    objects: list[bpy.types.Object] = [armature]
    if name == "rigged_cube@walk.fbx":
        add_action(armature, "walk", WALK_KEYS)
    else:
        texture = PALETTE_TEXTURE if name == "palette_cube.fbx" else CUBE_TEXTURE
        mat_name = "palette_mat" if name == "palette_cube.fbx" else "cube_mat"
        cube = build_cube("Cube", armature, make_material(mat_name, out_dir / texture))
        if name == "palette_cube.fbx":
            collapse_uvs_to_palette(cube)
        elif name == "shapekey_cube.fbx":
            add_blink_shape_key(cube)
        add_action(armature, "idle", IDLE_KEYS)
        objects.append(cube)
    path = out_dir / name
    export_fbx(path, objects)
    return path


def generate(out_dir: Path) -> list[Path]:
    """Write textures and all four FBX fixtures into ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    write_textures(out_dir)
    return [make_fixture(out_dir, name) for name in FIXTURE_NAMES]


def check(out_dir: Path) -> int:
    """Regenerate into a temp dir and compare structural summaries with the committed files."""
    with tempfile.TemporaryDirectory(prefix="gp_fixtures_") as tmp:
        fresh = generate(Path(tmp))
        fresh_summary = summarize_many(fresh)
        committed_summary = summarize_many([out_dir / p.name for p in fresh])
    drift = [name for name in FIXTURE_NAMES if fresh_summary[name] != committed_summary[name]]
    for name in drift:
        print(f"DRIFT {name}")
        print(
            json.dumps(
                {"committed": committed_summary[name], "fresh": fresh_summary[name]},
                indent=2,
                sort_keys=True,
            )
        )
    print("fixtures match" if not drift else f"{len(drift)} fixture(s) drifted")
    return 1 if drift else 0


def main(argv: list[str]) -> int:
    """Entry point; ``argv`` is everything after Blender's ``--`` separator."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", type=Path, help="tests/fixtures/fbx")
    parser.add_argument("--check", action="store_true", help="compare instead of overwrite")
    args = parser.parse_args(argv)
    out_dir = args.out_dir.resolve()
    if args.check:
        return check(out_dir)
    for path in generate(out_dir):
        print(f"wrote {path} ({path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []))
