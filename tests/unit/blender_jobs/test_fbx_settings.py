"""Snapshot of the NORMATIVE operator settings (plan 09 §2.5-2.6, 12 §1.2)."""

from __future__ import annotations

import pytest

from grounded_pipeline.blender_jobs.fbx_settings import (
    FBX_IMPORT_KWARGS,
    GLB_EXPORT_KWARGS,
    fbx_export_kwargs,
)


def test_fbx_import_kwargs_snapshot() -> None:
    assert FBX_IMPORT_KWARGS == {
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


def test_glb_export_kwargs_snapshot() -> None:
    assert GLB_EXPORT_KWARGS == {
        "export_format": "GLB",
        "use_selection": True,
        "export_apply": True,
        "export_texcoords": True,
        "export_normals": True,
        "export_tangents": False,
        "export_materials": "PLACEHOLDER",
        "export_colors": False,
        "export_attributes": False,
        "export_skins": False,
        "export_animations": False,
        "export_morph": False,
        "export_yup": True,
        "export_image_format": "NONE",
    }


@pytest.mark.parametrize(
    ("scale_option", "embed", "path_mode"),
    [("FBX_SCALE_ALL", False, "RELATIVE"), ("FBX_SCALE_UNITS", True, "COPY")],
)
def test_fbx_export_kwargs_snapshot(scale_option: str, embed: bool, path_mode: str) -> None:
    assert fbx_export_kwargs(scale_option, embed) == {
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
        "path_mode": path_mode,
        "embed_textures": embed,
        "use_custom_props": False,
        "use_metadata": True,
    }


def test_export_kwargs_are_fresh_dicts() -> None:
    assert fbx_export_kwargs("FBX_SCALE_ALL", False) is not fbx_export_kwargs(
        "FBX_SCALE_ALL", False
    )
