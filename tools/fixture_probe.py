"""Blender-side probe: import FBX files with the pipeline's import kwargs and summarise them.

Runs **inside** Blender (stdlib + ``bpy`` only)::

    blender -b --factory-startup --python tools/fixture_probe.py -- a.fbx [b.fbx] [--out out.json]

Each file is imported into a fresh, empty scene with ``FBX_IMPORT_KWARGS`` (plan ``09`` §2.6) and
reduced to a JSON-serialisable structural summary (armatures, bones in fingerprint order, actions,
meshes, images). The summary is written to ``--out`` when given; otherwise it is printed between
``GP_PROBE_JSON_BEGIN`` / ``GP_PROBE_JSON_END`` marker lines so callers can find it in Blender's
stdout.

Used by ``tests/integration/blender/test_fixtures_load.py`` and ``tools/make_fixtures.py --check``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import bpy

# Mirror of plan 09 §2.6 (NORMATIVE). Kept verbatim so the fixtures are read exactly as the pipeline
# will read vendor files.
FBX_IMPORT_KWARGS: dict[str, Any] = {
    "use_anim": True,
    "anim_offset": 1.0,
    "ignore_leaf_bones": False,
    "force_connect_children": False,
    "automatic_bone_orientation": False,
    "primary_bone_axis": "Y",
    "secondary_bone_axis": "X",
    "use_custom_normals": True,
    "use_image_search": True,
    "use_prepost_rot": True,
    "global_scale": 1.0,
    "bake_space_transform": False,
    "axis_forward": "-Z",
    "axis_up": "Y",
}

_ROUND = 4


def reset_scene() -> None:
    """Start from an empty factory scene so nothing leaks between probes."""
    bpy.ops.wm.read_factory_settings(use_empty=True)


def _vec(values: Any) -> list[float]:  # noqa: ANN401 - mathutils vectors are untyped at this boundary
    return [round(float(v), _ROUND) + 0.0 for v in values]  # "+ 0.0" folds -0.0 into 0.0


def _bones_depth_first(armature: bpy.types.Armature) -> list[bpy.types.Bone]:
    """Depth-first traversal, roots and children sorted by name (plan 09 §2.7)."""
    out: list[bpy.types.Bone] = []

    def visit(bone: bpy.types.Bone) -> None:
        out.append(bone)
        for child in sorted(bone.children, key=lambda b: b.name):
            visit(child)

    for root in sorted((b for b in armature.bones if b.parent is None), key=lambda b: b.name):
        visit(root)
    return out


def _armature_summary(obj: bpy.types.Object) -> dict[str, Any]:
    armature = obj.data
    assert isinstance(armature, bpy.types.Armature)
    bones = _bones_depth_first(armature)
    return {
        "bones": [b.name for b in bones],
        "bone_parents": {b.name: (b.parent.name if b.parent else None) for b in bones},
        "bone_heads": {b.name: _vec(b.head_local) for b in bones},
        "bone_tails": {b.name: _vec(b.tail_local) for b in bones},
    }


def _unique_uv_count(mesh: bpy.types.Mesh, layer: bpy.types.MeshUVLoopLayer) -> int:
    return (
        len({(round(d.uv[0], _ROUND), round(d.uv[1], _ROUND)) for d in layer.data})
        if mesh.loops
        else 0
    )


def _uv_bbox(layer: bpy.types.MeshUVLoopLayer) -> list[float]:
    us = [d.uv[0] for d in layer.data]
    vs = [d.uv[1] for d in layer.data]
    if not us:
        return []
    return _vec((min(us), min(vs), max(us), max(vs)))


def _mesh_summary(obj: bpy.types.Object) -> dict[str, Any]:
    mesh = obj.data
    assert isinstance(mesh, bpy.types.Mesh)
    uv_layers = list(mesh.uv_layers)
    shape_keys = mesh.shape_keys
    arm_mods = [m for m in obj.modifiers if m.type == "ARMATURE"]
    weighted = sum(1 for v in mesh.vertices if any(g.weight > 0.0 for g in v.groups))
    return {
        "vertices": len(mesh.vertices),
        "faces": len(mesh.polygons),
        "triangles": sum(len(p.vertices) - 2 for p in mesh.polygons),
        "shape_keys": [kb.name for kb in shape_keys.key_blocks] if shape_keys else [],
        "uv_layers": [layer.name for layer in uv_layers],
        "uv_unique_points": _unique_uv_count(mesh, uv_layers[0]) if uv_layers else 0,
        "uv_bbox": _uv_bbox(uv_layers[0]) if uv_layers else [],
        "materials": [m.name for m in mesh.materials if m is not None],
        "vertex_groups": sorted(vg.name for vg in obj.vertex_groups),
        "weighted_vertices": weighted,
        "armature_modifier_targets": [m.object.name if m.object else None for m in arm_mods],  # type: ignore[attr-defined]  # Modifier.object is only on ArmatureModifier
        "parent": obj.parent.name if obj.parent else None,
    }


def _image_summary(image: bpy.types.Image) -> dict[str, Any]:
    path = Path(bpy.path.abspath(image.filepath)) if image.filepath else None
    return {
        "size": [int(image.size[0]), int(image.size[1])],
        "filename": path.name if path else None,
        "found": bool(path and path.exists()),
    }


def summarize_fbx(path: Path) -> dict[str, Any]:
    """Import ``path`` into a fresh scene and return its structural summary."""
    reset_scene()
    bpy.ops.import_scene.fbx(filepath=str(path), **FBX_IMPORT_KWARGS)
    objects = sorted(bpy.data.objects, key=lambda o: o.name)
    return {
        "objects": [[o.name, o.type] for o in objects],
        "armatures": {o.name: _armature_summary(o) for o in objects if o.type == "ARMATURE"},
        "meshes": {o.name: _mesh_summary(o) for o in objects if o.type == "MESH"},
        "actions": {
            a.name: {
                "frame_range": _vec(a.frame_range),
                "fcurves": len(a.fcurves),
            }
            for a in sorted(bpy.data.actions, key=lambda a: a.name)
        },
        "images": {
            i.name: _image_summary(i) for i in sorted(bpy.data.images, key=lambda i: i.name)
        },
    }


def summarize_many(paths: list[Path]) -> dict[str, Any]:
    """``{file_name: summary}`` for every path, in the order given."""
    return {p.name: summarize_fbx(p) for p in paths}


def main(argv: list[str]) -> int:
    """Entry point; ``argv`` is everything after Blender's ``--`` separator."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fbx", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    summary = summarize_many(args.fbx)
    text = json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if args.out is not None:
        args.out.write_text(text, encoding="utf-8")
    else:
        print("GP_PROBE_JSON_BEGIN")
        print(text, end="")
        print("GP_PROBE_JSON_END")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []))
