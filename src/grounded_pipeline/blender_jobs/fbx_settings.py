"""Import/export operator settings shared by every Blender job (plan 09 §2.5-2.6, 12 §1.2).

This is the ONE place these dictionaries live. Changing any of them requires an ADR (plan 19 §9).
Pure constants and functions: no ``bpy`` import, so the module is unit-testable under CPython.
"""

from __future__ import annotations

from typing import Any

FBX_IMPORT_KWARGS: dict[str, Any] = {  # NORMATIVE (plan 09 §2.6)
    "use_anim": True,
    "anim_offset": 1.0,
    "ignore_leaf_bones": False,  # vendor files should not have them; do not guess
    "force_connect_children": False,
    "automatic_bone_orientation": False,  # preserve vendor bone axes exactly
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
"""The only FBX import kwargs in the codebase; ``roundtrip_animation`` uses the same dict."""

GLB_EXPORT_KWARGS: dict[str, Any] = {  # NORMATIVE (plan 09 §2.5); glTF 2.0 exporter
    "export_format": "GLB",
    "use_selection": True,
    "export_apply": True,  # apply modifiers (none remain; safe)
    "export_texcoords": True,
    "export_normals": True,
    "export_tangents": False,
    "export_materials": "PLACEHOLDER",  # no textures; placeholder keeps primitive grouping
    "export_colors": False,  # removed in 4.2; BpyScene drops unknown keys by feature detection
    "export_attributes": False,
    "export_skins": False,  # UNRIGGED
    "export_animations": False,
    "export_morph": False,
    "export_yup": True,
    "export_image_format": "NONE",
}
"""Static (unrigged) GLB for the texturing provider."""


def fbx_export_kwargs(scale_option: str, embed_textures: bool) -> dict[str, Any]:
    """Build the NORMATIVE FBX export kwargs (plan 12 §1.2).

    Args:
        scale_option: ``cfg.export.fbx_scale_option`` (``FBX_SCALE_ALL`` by default).
        embed_textures: ``cfg.export.embed_textures``; also selects ``path_mode``.

    Returns:
        Keyword arguments for ``bpy.ops.export_scene.fbx``.
    """
    return {
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
        "apply_scale_options": scale_option,
        "global_scale": 1.0,
        "apply_unit_scale": True,
        "use_space_transform": True,
        "bake_space_transform": False,
        "axis_forward": "-Z",
        "axis_up": "Y",
        "path_mode": "COPY" if embed_textures else "RELATIVE",
        "embed_textures": embed_textures,
        "use_custom_props": False,
        "use_metadata": True,
    }
