"""``BpyScene``: the production :class:`SceneApi` adapter over ``bpy`` (appendix B).

Only importable inside Blender (``bpy`` at module top); excluded from unit coverage and exercised
by ``tests/integration/blender``. Thin by design: translate, call the operator / data API,
translate back. No decision logic lives here.
"""

from __future__ import annotations

import contextlib
import logging
import math
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, Literal, cast

import bpy
import numpy as np
from mathutils import Matrix, Vector

from .rig_fingerprint import clean_action_name
from .scene_api import BoneInfo, ShaderInput, Uv, Vec3

log = logging.getLogger(__name__)

SubdivType = Literal["CATMULL_CLARK", "SIMPLE"]
UvSmooth = Literal[
    "NONE",
    "PRESERVE_CORNERS",
    "PRESERVE_CORNERS_AND_JUNCTIONS",
    "PRESERVE_CORNERS_JUNCTIONS_AND_CONCAVE",
    "PRESERVE_BOUNDARIES",
    "SMOOTH_ALL",
]
BakeType = Literal[
    "COMBINED",
    "AO",
    "SHADOW",
    "POSITION",
    "NORMAL",
    "UV",
    "ROUGHNESS",
    "EMIT",
    "ENVIRONMENT",
    "DIFFUSE",
    "GLOSSY",
    "TRANSMISSION",
]
_SUBDIV_TYPES: dict[str, SubdivType] = {"catmull": "CATMULL_CLARK", "simple": "SIMPLE"}
_UV_SMOOTH_MODES: frozenset[str] = frozenset(UvSmooth.__args__)  # type: ignore[attr-defined]  # Literal has __args__ at runtime
_BAKE_TYPES: frozenset[str] = frozenset(BakeType.__args__)  # type: ignore[attr-defined]  # Literal has __args__ at runtime
_PRINCIPLED_IMAGE_INPUTS = {
    "basecolor": ("Base Color", "sRGB"),
    "emission": ("Emission Color", "sRGB"),
}
_PRINCIPLED_BAKE_INPUTS = (
    "Base Color",
    "Metallic",
    "Roughness",
    "Normal",
    "Emission Color",
    "Emission Strength",
)
_GLTF_OCCLUSION_INPUT = "Occlusion"


# --------------------------------------------------------------------------- context helpers
def _scene() -> bpy.types.Scene:
    scene = bpy.context.scene
    assert scene is not None  # always set in background mode
    return scene


def _view_layer() -> bpy.types.ViewLayer:
    view_layer = bpy.context.view_layer
    assert view_layer is not None  # always set in background mode
    return view_layer


def _obj(name: str) -> bpy.types.Object:
    if name not in bpy.data.objects:
        raise KeyError(f"no object named {name!r}")
    return bpy.data.objects[name]


def _mesh(name: str) -> bpy.types.Mesh:
    data = _obj(name).data
    if not isinstance(data, bpy.types.Mesh):
        raise TypeError(f"{name!r} is not a mesh object")
    return data


def _armature(name: str) -> bpy.types.Armature:
    data = _obj(name).data
    if not isinstance(data, bpy.types.Armature):
        raise TypeError(f"{name!r} is not an armature object")
    return data


def _node_tree(material: bpy.types.Material) -> bpy.types.ShaderNodeTree:
    material.use_nodes = True
    tree = material.node_tree
    assert tree is not None  # use_nodes=True guarantees a tree
    return tree


def _vec3(values: Vector) -> Vec3:
    x, y, z = values.to_tuple()
    return (float(x), float(y), float(z))


def _mat4(matrix: Matrix) -> list[list[float]]:
    return [[float(matrix[r][c]) for c in range(4)] for r in range(4)]


def _set_colorspace(image: bpy.types.Image, name: str) -> None:
    settings = image.colorspace_settings
    assert settings is not None  # every image has colour-space settings
    settings.name = name  # type: ignore[assignment]  # colour-space names are runtime enums


def _socket(
    sockets: bpy.types.NodeInputs | bpy.types.NodeOutputs | None, name: str
) -> bpy.types.NodeSocket:
    assert sockets is not None  # shader nodes always have socket collections
    return sockets[name]


def _link_from(socket: bpy.types.NodeSocket) -> bpy.types.NodeSocket | None:
    links = socket.links
    if not links:
        return None
    return links[0].from_socket


def _unhide(objects: Iterable[bpy.types.Object]) -> None:
    """Imported objects must be visible/selectable for operators (appendix B §7)."""
    for obj in objects:
        obj.hide_viewport = False
        obj.hide_select = False
        with contextlib.suppress(RuntimeError):  # not in the view layer
            obj.hide_set(False)


def _select_only(objects: list[bpy.types.Object], active: bpy.types.Object | None) -> None:
    for obj in bpy.data.objects:
        if obj.name in _view_layer().objects:
            obj.select_set(False)
    for obj in objects:
        obj.select_set(True)
    _view_layer().objects.active = active


def _override(objects: list[bpy.types.Object], active: bpy.types.Object) -> dict[str, Any]:
    return {
        "object": active,
        "active_object": active,
        "selected_objects": objects,
        "selected_editable_objects": objects,
    }


def _supported_kwargs(op: Any, kwargs: dict[str, Any]) -> dict[str, Any]:  # noqa: ANN401
    """Drop kwargs the operator does not know (property names drift across 4.x; appendix B §2)."""
    known = {prop.identifier for prop in op.get_rna_type().properties}
    dropped = sorted(key for key in kwargs if key not in known)
    if dropped:
        log.debug("dropping unsupported operator kwargs: %s", dropped)
    return {key: value for key, value in kwargs.items() if key in known}


# --------------------------------------------------------------------------- adapter
class BpyScene:
    """``SceneApi`` over the live Blender session (see ``scene_api.py`` for the contract)."""

    # ------------------------------------------------------------------ scene lifecycle
    def reset(self) -> None:
        bpy.ops.wm.read_factory_settings(use_empty=True)

    def import_fbx(self, path: str, **kwargs: Any) -> None:  # noqa: ANN401
        before = {obj.name for obj in bpy.data.objects}
        bpy.ops.import_scene.fbx(filepath=path, **kwargs)
        _unhide(obj for obj in bpy.data.objects if obj.name not in before)

    def import_glb(self, path: str, **kwargs: Any) -> None:  # noqa: ANN401
        before = {obj.name for obj in bpy.data.objects}
        accepted = _supported_kwargs(bpy.ops.import_scene.gltf, kwargs)
        bpy.ops.import_scene.gltf(filepath=path, **accepted)
        _unhide(obj for obj in bpy.data.objects if obj.name not in before)

    def open_blend(self, path: str) -> None:
        bpy.ops.wm.open_mainfile(filepath=path)

    def save_blend(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=path)

    def blender_version(self) -> tuple[int, int, int]:
        major, minor, patch = bpy.app.version
        return (int(major), int(minor), int(patch))

    # ------------------------------------------------------------------ queries
    def armatures(self) -> list[str]:
        return sorted(obj.name for obj in bpy.data.objects if obj.type == "ARMATURE")

    def meshes(self) -> list[str]:
        return sorted(obj.name for obj in bpy.data.objects if obj.type == "MESH")

    def skinned_meshes(self, armature: str) -> list[str]:
        arm = _obj(armature)
        names: list[str] = []
        for obj in bpy.data.objects:
            if obj.type != "MESH":
                continue
            bound = any(
                isinstance(mod, bpy.types.ArmatureModifier) and mod.object == arm
                for mod in obj.modifiers
            )
            if bound or (obj.parent == arm and obj.parent_type == "ARMATURE"):
                names.append(obj.name)
        return sorted(names)

    def armature_data_name(self, armature: str) -> str:
        return _armature(armature).name

    def bones(self, armature: str) -> list[BoneInfo]:
        return [
            BoneInfo(
                name=bone.name,
                parent=bone.parent.name if bone.parent else None,
                head=_vec3(bone.head_local),
                tail=_vec3(bone.tail_local),
                matrix_local=_mat4(bone.matrix_local),
            )
            for bone in _armature(armature).bones
        ]

    def actions(self) -> list[str]:
        return sorted(
            clean_action_name(action.name)
            for action in bpy.data.actions
            if action.id_root == "OBJECT"
        )

    def mesh_counts(self, obj: str) -> tuple[int, int, int]:
        mesh = _mesh(obj)
        faces = len(mesh.polygons)
        totals = [0] * faces
        mesh.polygons.foreach_get("loop_total", totals)
        return len(mesh.vertices), faces, sum(totals) - 2 * faces

    def has_shapekeys(self, obj: str) -> bool:
        return _mesh(obj).shape_keys is not None

    def uv_layers(self, obj: str) -> list[str]:
        return [layer.name for layer in _mesh(obj).uv_layers]

    def uv_loops(self, obj: str, layer: str) -> list[Uv]:
        mesh = _mesh(obj)
        buffer = [0.0] * (2 * len(mesh.loops))
        mesh.uv_layers[layer].data.foreach_get("uv", buffer)
        return list(zip(buffer[0::2], buffer[1::2], strict=True))

    def face_loop_ranges(self, obj: str) -> list[tuple[int, int]]:
        mesh = _mesh(obj)
        starts, totals = [0] * len(mesh.polygons), [0] * len(mesh.polygons)
        mesh.polygons.foreach_get("loop_start", starts)
        mesh.polygons.foreach_get("loop_total", totals)
        return [(s, s + t) for s, t in zip(starts, totals, strict=True)]

    def material_slots(self, obj: str) -> list[str]:
        return [slot.material.name if slot.material else "" for slot in _obj(obj).material_slots]

    def image_sizes(self, obj: str) -> list[tuple[int, int]]:
        sizes: list[tuple[int, int]] = []
        for slot in _obj(obj).material_slots:
            if slot.material is not None:
                sizes.extend(s for s in _material_images(slot.material) if s not in sizes)
        return sizes

    def material_image_sizes(self, material: str) -> list[tuple[int, int]]:
        return _material_images(bpy.data.materials[material])

    def bbox_diagonal(self, objs: list[str]) -> float:
        low, high = self.object_bbox(objs)
        return math.dist(low, high)

    def object_bbox(self, objs: list[str]) -> tuple[Vec3, Vec3]:
        """Box around the world-space vertices of the MESH objects in ``objs`` (others ignored)."""
        points = [
            p
            for name in objs
            if _obj(name).type == "MESH"
            for p in self.world_vertex_positions(name)
        ]
        if not points:
            return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
        array = np.asarray(points, dtype=np.float64)
        low, high = array.min(axis=0), array.max(axis=0)
        return (float(low[0]), float(low[1]), float(low[2])), (
            float(high[0]),
            float(high[1]),
            float(high[2]),
        )

    def world_vertex_positions(self, obj: str) -> list[Vec3]:
        mesh = _mesh(obj)
        count = len(mesh.vertices)
        buffer = [0.0] * (3 * count)
        mesh.vertices.foreach_get("co", buffer)
        matrix = np.asarray(_mat4(_obj(obj).matrix_world), dtype=np.float64)
        local = np.asarray(buffer, dtype=np.float64).reshape(count, 3)
        world = local @ matrix[:3, :3].T + matrix[:3, 3]
        return [(float(x), float(y), float(z)) for x, y, z in world]

    # ------------------------------------------------------------------ mesh mutations
    def merge_material_slots(self, objs: list[str], name: str) -> None:
        objects = [_obj(o) for o in objs]
        first = next((s.material for o in objects for s in o.material_slots if s.material), None)
        if first is None:
            first = bpy.data.materials.new(name)
        if first.name != name:
            if name in bpy.data.materials:
                bpy.data.materials[name].name = f"{name}.orig"
            first.name = name
        for obj in objects:
            mesh = _mesh(obj.name)
            mesh.materials.clear()
            mesh.materials.append(first)
            mesh.polygons.foreach_set("material_index", [0] * len(mesh.polygons))

    def keep_first_uv_layer(self, obj: str) -> None:
        mesh = _mesh(obj)
        while len(mesh.uv_layers) > 1:
            mesh.uv_layers.remove(mesh.uv_layers[-1])
        if mesh.uv_layers:
            mesh.uv_layers[0].active = True
            mesh.uv_layers[0].active_render = True

    def smart_unwrap(
        self, objs: list[str], angle_deg: float, island_margin: float, pack_margin: float
    ) -> None:
        objects = [_obj(o) for o in objs]
        _select_only(objects, objects[0])
        with bpy.context.temp_override(**_override(objects, objects[0])):
            bpy.ops.object.mode_set(mode="EDIT")
            try:
                bpy.ops.mesh.select_all(action="SELECT")
                bpy.ops.uv.smart_project(
                    angle_limit=math.radians(angle_deg),
                    island_margin=island_margin,
                    area_weight=0.0,
                    correct_aspect=True,
                    scale_to_bounds=False,
                )
                bpy.ops.uv.pack_islands(rotate=True, margin=pack_margin)
            finally:
                bpy.ops.object.mode_set(mode="OBJECT")

    def clear_shapekeys(self, obj: str) -> None:
        _obj(obj).shape_key_clear()

    def add_subsurf(  # noqa: PLR0917 - normative signature (plan 09 §2.3)
        self, obj: str, name: str, levels: int, kind: str, uv_smooth: str, limit_surface: bool
    ) -> None:
        if uv_smooth not in _UV_SMOOTH_MODES:
            raise ValueError(f"unknown uv_smooth mode {uv_smooth!r}")
        mod = _obj(obj).modifiers.new(name, "SUBSURF")
        if not isinstance(mod, bpy.types.SubsurfModifier):
            raise TypeError("modifiers.new did not return a SubsurfModifier")
        mod.subdivision_type = _SUBDIV_TYPES[kind]
        mod.levels = levels
        mod.render_levels = levels
        mod.uv_smooth = cast(UvSmooth, uv_smooth)  # validated above
        mod.use_limit_surface = limit_surface

    def move_modifier_to_top(self, obj: str, name: str) -> None:
        target = _obj(obj)
        _select_only([target], target)
        with bpy.context.temp_override(**_override([target], target)):
            bpy.ops.object.modifier_move_to_index(modifier=name, index=0)

    def apply_modifier(self, obj: str, name: str) -> None:
        target = _obj(obj)
        _unhide([target])
        _select_only([target], target)
        with bpy.context.temp_override(**_override([target], target)):
            bpy.ops.object.modifier_apply(modifier=name)

    def shade_smooth(self, obj: str) -> None:
        mesh = _mesh(obj)
        mesh.polygons.foreach_set("use_smooth", [True] * len(mesh.polygons))
        mesh.update()

    # ------------------------------------------------------------------ export
    def export_glb(self, objs: list[str], path: str, **kwargs: Any) -> None:  # noqa: ANN401
        objects = [_obj(o) for o in objs]
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        accepted = _supported_kwargs(bpy.ops.export_scene.gltf, kwargs)
        restore = _disable_armature_modifiers(objects)
        try:
            _select_only(objects, objects[0] if objects else None)
            with bpy.context.temp_override(selected_objects=objects):
                bpy.ops.export_scene.gltf(filepath=path, **accepted)
        finally:
            restore()

    def export_fbx(self, objs: list[str], path: str, **kwargs: Any) -> None:  # noqa: ANN401
        objects = [_obj(o) for o in objs]
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        active = next((o for o in objects if o.type == "ARMATURE"), objects[0])
        _select_only(objects, active)
        with bpy.context.temp_override(**_override(objects, active)):
            bpy.ops.export_scene.fbx(filepath=path, **kwargs)

    def reset_pose(self, armature: str) -> None:
        arm = _obj(armature)
        anim = arm.animation_data
        if anim is not None:
            anim.action = None  # animation_data itself stays: the exporter needs it (B §8.3)
        pose = arm.pose
        assert pose is not None  # armature objects always carry a pose
        for pose_bone in pose.bones:
            pose_bone.matrix_basis = Matrix.Identity(4)

    def clean_action_names(self) -> None:
        for action in list(bpy.data.actions):
            clean = clean_action_name(action.name)
            if (
                action.id_root == "OBJECT"
                and clean != action.name
                and clean not in bpy.data.actions
            ):
                action.name = clean

    # ------------------------------------------------------------------ baking (WP-08)
    def set_render_engine_cycles(self, device: str, samples: int) -> None:
        scene = _scene()
        scene.render.engine = "CYCLES"  # type: ignore[assignment]  # stubs only list the EEVEE id
        cycles = scene.cycles  # add-on property group; untyped in the stubs
        assert cycles is not None  # Cycles is bundled and enabled with --factory-startup
        cycles.device = device
        cycles.samples = samples
        cycles.use_denoising = False
        cycles.seed = 0
        view, display = scene.view_settings, scene.display_settings
        assert view is not None  # always present on a scene
        assert display is not None
        # Colour-management names are runtime enums; the stubs only know "NONE".
        view.view_transform = "Standard"  # type: ignore[assignment]  # runtime enum
        view.look = "None"  # type: ignore[assignment]  # runtime enum
        display.display_device = "sRGB"  # type: ignore[assignment]  # runtime enum

    def configure_bake(
        self, margin_px: int, cage_extrusion: float, max_ray_distance: float
    ) -> None:
        bake = _scene().render.bake
        bake.use_selected_to_active = True
        bake.cage_extrusion = cage_extrusion
        bake.max_ray_distance = max_ray_distance
        bake.margin = margin_px
        bake.margin_type = "EXTEND"
        bake.use_clear = True
        bake.use_pass_direct = False
        bake.use_pass_indirect = False
        bake.use_pass_color = True

    def bake_map(
        self,
        targets: list[str],
        sources: list[str],
        *,
        map_name: str,
        bake_type: str,
        resolution: int,
        out_path: str,
        colorspace: str,
        emit_input: str | None,
    ) -> None:
        if bake_type not in _BAKE_TYPES:
            raise ValueError(f"unknown bake type {bake_type!r}")
        if not targets:
            raise ValueError("bake_map needs at least one target")
        target_objs, source_objs = [_obj(t) for t in targets], [_obj(s) for s in sources]
        image = bpy.data.images.new(
            map_name, resolution, resolution, alpha=False, float_buffer=False
        )
        _set_colorspace(image, colorspace)
        attached = _attach_bake_image(target_objs, image)
        restores = [_rewire_to_emission(o, emit_input) for o in source_objs] if emit_input else []
        try:
            for index, target_obj in enumerate(target_objs):
                _run_bake(target_obj, source_objs, cast(BakeType, bake_type), clear=index == 0)
            _save_png(image, out_path)
        finally:
            for restore in restores:
                restore()
            for tree, node in attached:
                tree.nodes.remove(node)
            bpy.data.images.remove(image)

    def shader_inputs(self, obj: str) -> list[ShaderInput]:
        found: list[ShaderInput] = []
        for slot in _obj(obj).material_slots:
            material = slot.material
            tree = material.node_tree if material is not None and material.use_nodes else None
            if tree is None:
                continue
            for node in tree.nodes:
                names = _bake_input_names(node)
                inputs = node.inputs
                for name in names:
                    if inputs is not None and name in inputs:
                        socket = inputs[name]
                        found.append(ShaderInput(name, bool(socket.links), _socket_value(socket)))
        return found

    def image_stats(self, path: str) -> tuple[float, float]:
        image = bpy.data.images.load(path, check_existing=False)
        try:
            width, height = int(image.size[0]), int(image.size[1])
            buffer = np.empty(width * height * image.channels, dtype=np.float32)
            cast("Any", image.pixels).foreach_get(buffer)  # stubs type ``pixels`` as a float
            rgb = buffer.reshape(-1, image.channels)[:, : min(3, image.channels)]
        finally:
            bpy.data.images.remove(image)
        return float(rgb.mean()), float(rgb.var())

    def transform_objects(self, objs: list[str], scale: float, translation: Vec3) -> None:
        transform = Matrix.Translation(Vector(translation)) @ Matrix.Scale(scale, 4)
        for name in objs:
            obj = _obj(name)
            obj.matrix_world = transform @ obj.matrix_world

    def hide_from_render(self, obj: str) -> None:
        _obj(obj).hide_render = True

    def set_pose_position(self, armature: str, rest: bool) -> None:
        _armature(armature).pose_position = "REST" if rest else "POSE"

    # ------------------------------------------------------------------ materials (WP-09)
    def bind_urp_material(
        self, objs: list[str], material_name: str, textures: dict[str, str]
    ) -> None:
        material = (
            bpy.data.materials[material_name]
            if material_name in bpy.data.materials
            else bpy.data.materials.new(material_name)
        )
        tree = _node_tree(material)
        tree.nodes.clear()
        output = tree.nodes.new("ShaderNodeOutputMaterial")
        bsdf = tree.nodes.new("ShaderNodeBsdfPrincipled")
        tree.links.new(_socket(bsdf.outputs, "BSDF"), _socket(output.inputs, "Surface"))
        for key, (input_name, colorspace) in _PRINCIPLED_IMAGE_INPUTS.items():
            if key in textures:
                image = _image_node(tree, textures[key], colorspace)
                tree.links.new(_socket(image.outputs, "Color"), _socket(bsdf.inputs, input_name))
        if "emission" in textures:
            _set_default(_socket(bsdf.inputs, "Emission Strength"), 1.0)
        if "normal" in textures:
            normal_map = tree.nodes.new("ShaderNodeNormalMap")
            image = _image_node(tree, textures["normal"], "Non-Color")
            tree.links.new(_socket(image.outputs, "Color"), _socket(normal_map.inputs, "Color"))
            tree.links.new(_socket(normal_map.outputs, "Normal"), _socket(bsdf.inputs, "Normal"))
        if "metallic_smoothness" in textures:
            _wire_metallic_smoothness(tree, bsdf, textures["metallic_smoothness"])
        for name in objs:
            mesh = _mesh(name)
            mesh.materials.clear()
            mesh.materials.append(material)
            mesh.polygons.foreach_set("material_index", [0] * len(mesh.polygons))


# --------------------------------------------------------------------------- module helpers
def _set_default(socket: bpy.types.NodeSocket, value: object) -> None:
    socket.default_value = value  # type: ignore[attr-defined]  # NodeSocket subclasses expose default_value


def _wire_metallic_smoothness(
    tree: bpy.types.ShaderNodeTree, bsdf: bpy.types.Node, path: str
) -> None:
    """Metallic ← R, Roughness ← 1 - A of the URP metallic/smoothness map (appendix B §6)."""
    image = _image_node(tree, path, "Non-Color")
    separate = tree.nodes.new("ShaderNodeSeparateColor")
    invert = tree.nodes.new("ShaderNodeInvert")
    tree.links.new(_socket(image.outputs, "Color"), _socket(separate.inputs, "Color"))
    tree.links.new(_socket(separate.outputs, "Red"), _socket(bsdf.inputs, "Metallic"))
    tree.links.new(_socket(image.outputs, "Alpha"), _socket(invert.inputs, "Color"))
    tree.links.new(_socket(invert.outputs, "Color"), _socket(bsdf.inputs, "Roughness"))


def _material_images(material: bpy.types.Material) -> list[tuple[int, int]]:
    tree = material.node_tree if material.use_nodes else None
    if tree is None:
        return []
    sizes: list[tuple[int, int]] = []
    for node in tree.nodes:
        if isinstance(node, bpy.types.ShaderNodeTexImage) and node.image is not None:
            width, height = int(node.image.size[0]), int(node.image.size[1])
            if width > 0 and height > 0:
                sizes.append((width, height))
    return sizes


def _first_material(obj: bpy.types.Object, fallback_name: str) -> bpy.types.Material:
    for slot in obj.material_slots:
        if slot.material is not None:
            return slot.material
    material = bpy.data.materials.new(fallback_name)
    mesh = obj.data
    if isinstance(mesh, bpy.types.Mesh):
        mesh.materials.append(material)
    return material


def _image_node(
    tree: bpy.types.ShaderNodeTree, path: str, colorspace: str
) -> bpy.types.ShaderNodeTexImage:
    image = bpy.data.images.load(path, check_existing=True)
    _set_colorspace(image, colorspace)
    if bpy.data.filepath:
        image.filepath = bpy.path.relpath(path)  # portable inside the model folder (appendix B §6)
    node = tree.nodes.new("ShaderNodeTexImage")
    if not isinstance(node, bpy.types.ShaderNodeTexImage):
        raise TypeError("nodes.new did not return a ShaderNodeTexImage")
    node.image = image
    return node


def _disable_armature_modifiers(objects: list[bpy.types.Object]) -> Callable[[], None]:
    """Bypass armature deformation (rest shape in the static GLB); returns the undo closure."""
    saved: list[tuple[bpy.types.Modifier, bool, bool]] = []
    for obj in objects:
        for mod in obj.modifiers:
            if mod.type == "ARMATURE":
                saved.append((mod, mod.show_viewport, mod.show_render))
                mod.show_viewport = False
                mod.show_render = False

    def restore() -> None:
        for mod, viewport, render in saved:
            mod.show_viewport = viewport
            mod.show_render = render

    return restore


def _rewire_to_emission(obj: bpy.types.Object, input_name: str) -> Callable[[], None]:
    """EMIT trick (appendix B §5): route ``input_name`` through an Emission shader; returns undo."""
    undo: list[Callable[[], None]] = []
    for slot in obj.material_slots:
        if slot.material is None:
            continue
        tree = _node_tree(slot.material)
        output = next((n for n in tree.nodes if n.type == "OUTPUT_MATERIAL"), None)
        socket = _find_input_socket(tree, input_name)
        if output is None or socket is None:
            continue
        surface = _socket(output.inputs, "Surface")
        previous = _link_from(surface)
        emission = tree.nodes.new("ShaderNodeEmission")
        source = _link_from(socket)
        if source is not None:
            tree.links.new(source, _socket(emission.inputs, "Color"))
        else:
            value = socket.default_value  # type: ignore[attr-defined]  # NodeSocket subclasses expose default_value
            rgba = (value, value, value, 1.0) if isinstance(value, float) else tuple(value)
            _set_default(_socket(emission.inputs, "Color"), rgba)
        tree.links.new(_socket(emission.outputs, "Emission"), surface)
        undo.append(_restorer(tree, emission, previous, surface))

    def restore_all() -> None:
        for fn in undo:
            fn()

    return restore_all


def _restorer(
    tree: bpy.types.ShaderNodeTree,
    emission: bpy.types.Node,
    previous: bpy.types.NodeSocket | None,
    surface: bpy.types.NodeSocket,
) -> Callable[[], None]:
    def restore() -> None:
        tree.nodes.remove(emission)
        if previous is not None:
            tree.links.new(previous, surface)

    return restore


def _find_input_socket(
    tree: bpy.types.ShaderNodeTree, input_name: str
) -> bpy.types.NodeSocket | None:
    """Principled input by name, or the glTF settings group's ``Occlusion`` input."""
    for node in tree.nodes:
        inputs = node.inputs
        if input_name in _bake_input_names(node) and inputs and input_name in inputs:
            return inputs[input_name]
    return None


def _bake_input_names(node: bpy.types.Node) -> tuple[str, ...]:
    """Socket names worth reporting for ``node``: Principled inputs or the glTF ``Occlusion``."""
    if node.type == "BSDF_PRINCIPLED":
        return _PRINCIPLED_BAKE_INPUTS
    if _is_gltf_group(node):
        return (_GLTF_OCCLUSION_INPUT,)
    return ()


def _is_gltf_group(node: bpy.types.Node) -> bool:
    """The glTF importer's settings group (``glTF Material Output`` / ``glTF Settings``)."""
    if not isinstance(node, bpy.types.ShaderNodeGroup):
        return False
    group = node.node_tree
    return node.name.startswith("glTF") or (group is not None and group.name.startswith("glTF"))


def _socket_value(socket: bpy.types.NodeSocket) -> tuple[float, ...]:
    value = getattr(socket, "default_value", None)
    if value is None:
        return ()
    if isinstance(value, (int, float)):
        return (float(value),)
    return tuple(float(v) for v in value)


def _attach_bake_image(
    targets: list[bpy.types.Object], image: bpy.types.Image
) -> list[tuple[bpy.types.ShaderNodeTree, bpy.types.ShaderNodeTexImage]]:
    """Add an active Image Texture node holding ``image`` to each distinct target material."""
    attached: list[tuple[bpy.types.ShaderNodeTree, bpy.types.ShaderNodeTexImage]] = []
    seen: set[str] = set()
    for target in targets:
        material = _first_material(target, f"{target.name}_bake")
        if material.name in seen:
            continue
        seen.add(material.name)
        tree = _node_tree(material)
        node = tree.nodes.new("ShaderNodeTexImage")
        if not isinstance(node, bpy.types.ShaderNodeTexImage):
            raise TypeError("nodes.new did not return a ShaderNodeTexImage")
        node.image = image
        tree.nodes.active = node
        attached.append((tree, node))
    return attached


def _run_bake(
    target: bpy.types.Object, sources: list[bpy.types.Object], bake_type: BakeType, *, clear: bool
) -> None:
    """One ``bpy.ops.object.bake`` call; unspecified settings come from ``scene.render.bake``."""
    objects = [*sources, target]
    kwargs: dict[str, Any] = {
        "type": bake_type,
        "use_selected_to_active": True,
        "use_clear": clear,
        "target": "IMAGE_TEXTURES",
    }
    if bake_type == "DIFFUSE":
        kwargs["pass_filter"] = {"COLOR"}
    if bake_type == "NORMAL":
        kwargs["normal_space"] = "TANGENT"
    _select_only(objects, target)
    with bpy.context.temp_override(**_override(objects, target)):
        bpy.ops.object.bake(**kwargs)


def _save_png(image: bpy.types.Image, out_path: str) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    image.filepath_raw = out_path
    image.file_format = "PNG"
    image.save()
