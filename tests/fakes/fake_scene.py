"""``FakeScene``: in-memory :class:`SceneApi` for job tests (plan 15 §3.1).

Objects are :class:`FakeObject` records; every operation mutates counts predictably:

* ``add_subsurf`` + ``apply_modifier`` → ``faces x 4^levels``, vertices by the closed-quad
  Catmull-Clark formula (``V' = V + 3F`` per level), ``tris = faces x 2``;
* ``smart_unwrap`` replaces the first UV layer by a synthetic non-overlapping atlas;
* ``export_*`` / ``save_blend`` record their calls and write small deterministic placeholder
  files so callers can hash them;
* ``mutate_rig()`` is a test hook for the rig-mutation guard.

Every call is appended to :attr:`FakeScene.calls` as ``(method, args...)`` so tests can assert
the exact order of operations. Positions are synthetic (vertex ``i`` sits at ``(i, 0, 0)``,
scaled/translated by the object's transform).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from grounded_pipeline.blender_jobs.rig_fingerprint import clean_action_name
from grounded_pipeline.blender_jobs.scene_api import BoneInfo, ShaderInput, Uv, Vec3

FAKE_GLB_HEADER = b"FAKE_GLB\n"
FAKE_FBX_HEADER = b"FAKE_FBX\n"
FAKE_BLEND_BYTES = b"FAKE_BLEND\n"
FAKE_PNG_BYTES = b"\x89PNG\r\n\x1a\nFAKE"
DEFAULT_VERSION = (4, 5, 12)
_DEFAULT_LOOPS_PER_FACE = 4


@dataclass
class FakeModifier:
    name: str
    kind: str
    levels: int = 0
    subdivision_type: str = ""
    uv_smooth: str = ""
    limit_surface: bool = False


@dataclass
class FakeObject:
    """One scene object; ``kind`` is ``"MESH"`` or ``"ARMATURE"``."""

    kind: str
    verts: int = 0
    faces: int = 0
    tris: int = 0
    uv_layers: dict[str, list[list[Uv]]] = field(default_factory=dict)  # per-face loop lists
    materials: list[str] = field(default_factory=list)
    modifiers: list[FakeModifier] = field(default_factory=list)
    shapekeys: bool = False
    parent_armature: str | None = None
    bones: list[BoneInfo] = field(default_factory=list)
    image_sizes: list[tuple[int, int]] = field(default_factory=list)
    data_name: str | None = None
    smooth: bool = False
    hide_render: bool = False
    scale: float = 1.0
    translation: Vec3 = (0.0, 0.0, 0.0)
    shader_inputs: list[ShaderInput] = field(default_factory=list)  # bake sources
    pose_rest: bool = False  # armatures: ``pose_position == "REST"``

    def face_loop_counts(self) -> list[int]:
        if self.uv_layers:
            first = next(iter(self.uv_layers.values()))
            return [len(face) for face in first]
        return [_DEFAULT_LOOPS_PER_FACE] * self.faces


class FakeScene:
    """See the module docstring. Construct with objects or use the ``scene_*`` builders."""

    def __init__(
        self,
        objects: dict[str, FakeObject] | None = None,
        *,
        actions: list[tuple[str, str]] | None = None,
        version: tuple[int, int, int] = DEFAULT_VERSION,
    ) -> None:
        self.objects: dict[str, FakeObject] = dict(objects or {})
        self.raw_actions: list[tuple[str, str]] = list(actions or [])  # (name, id_root)
        self.version = version
        self.calls: list[tuple[Any, ...]] = []
        self.exports: list[tuple[str, list[str], str, dict[str, Any]]] = []
        self.saved_blends: list[str] = []
        self.opened_blends: list[str] = []
        self.imports: list[tuple[str, str, dict[str, Any]]] = []
        self.baked: list[dict[str, Any]] = []
        self.bound_textures: dict[str, dict[str, str]] = {}
        self.pose_resets: list[str] = []
        # What the next ``import_fbx`` adds to the scene (jobs that ``reset()`` first, e.g.
        # ``roundtrip_animation``, start empty and rely on the import to populate the scene).
        self.fbx_payload: dict[str, FakeObject] = {}
        self.fbx_actions: list[tuple[str, str]] = []
        self.render: dict[str, Any] = {}
        self.bake_settings: dict[str, Any] = {}
        # Failure injection knobs for job tests.
        self.import_error: Exception | None = None
        self.export_error: Exception | None = None
        self.export_writes_file = True
        self.unwrap_zero_area = False
        # Bake knobs (WP-08): objects that ``import_glb`` adds, per-map bake failures, map names
        # whose ``image_stats`` read back as all-black, and the EMIT rewire/restore trail.
        self.glb_objects: dict[str, FakeObject] = {}
        self.bake_errors: dict[str, Exception] = {}
        self.empty_images: set[str] = set()
        self.rewires: list[tuple[str, str, str]] = []  # ("rewire"|"restore", map, input)

    # ------------------------------------------------------------------ helpers
    def _record(self, *call: Any) -> None:
        self.calls.append(call)

    def method_calls(self, name: str | None = None) -> list[tuple[Any, ...]]:
        """Recorded calls, optionally filtered by method name."""
        return [c for c in self.calls if name is None or c[0] == name]

    def call_order(self) -> list[str]:
        """Method names in call order."""
        return [str(c[0]) for c in self.calls]

    def _mesh(self, name: str) -> FakeObject:
        obj = self.objects[name]
        if obj.kind != "MESH":
            raise TypeError(f"{name!r} is not a mesh")
        return obj

    def _armature(self, name: str) -> FakeObject:
        obj = self.objects[name]
        if obj.kind != "ARMATURE":
            raise TypeError(f"{name!r} is not an armature")
        return obj

    def mutate_rig(self) -> None:
        """Test hook: perturb the first bone of the first armature (rig-mutation guard)."""
        arm = self._armature(self.armatures()[0])
        bone = arm.bones[0]
        matrix = [list(row) for row in bone.matrix_local]
        matrix[0][3] += 1.0
        arm.bones[0] = replace(bone, matrix_local=matrix)

    # ------------------------------------------------------------------ scene lifecycle
    def reset(self) -> None:
        self._record("reset")
        self.objects.clear()
        self.raw_actions.clear()

    def import_fbx(self, path: str, **kwargs: Any) -> None:
        self._record("import_fbx", path)
        self.imports.append(("fbx", path, dict(kwargs)))
        if self.import_error is not None:
            raise self.import_error
        self.objects.update(self.fbx_payload)
        self.raw_actions.extend(self.fbx_actions)

    def import_glb(self, path: str, **kwargs: Any) -> None:
        self._record("import_glb", path)
        self.imports.append(("glb", path, dict(kwargs)))
        if self.import_error is not None:
            raise self.import_error
        self.objects.update(self.glb_objects)

    def open_blend(self, path: str) -> None:
        self._record("open_blend", path)
        self.opened_blends.append(path)

    def save_blend(self, path: str) -> None:
        self._record("save_blend", path)
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(FAKE_BLEND_BYTES)
        self.saved_blends.append(path)

    def blender_version(self) -> tuple[int, int, int]:
        return self.version

    # ------------------------------------------------------------------ queries
    def armatures(self) -> list[str]:
        return sorted(n for n, o in self.objects.items() if o.kind == "ARMATURE")

    def meshes(self) -> list[str]:
        return sorted(n for n, o in self.objects.items() if o.kind == "MESH")

    def skinned_meshes(self, armature: str) -> list[str]:
        return sorted(
            n for n, o in self.objects.items() if o.kind == "MESH" and o.parent_armature == armature
        )

    def armature_data_name(self, armature: str) -> str:
        arm = self._armature(armature)
        return arm.data_name or armature

    def bones(self, armature: str) -> list[BoneInfo]:
        return list(self._armature(armature).bones)

    def actions(self) -> list[str]:
        return sorted(clean_action_name(n) for n, root in self.raw_actions if root == "OBJECT")

    def mesh_counts(self, obj: str) -> tuple[int, int, int]:
        mesh = self._mesh(obj)
        return mesh.verts, mesh.faces, mesh.tris

    def has_shapekeys(self, obj: str) -> bool:
        return self._mesh(obj).shapekeys

    def uv_layers(self, obj: str) -> list[str]:
        return list(self._mesh(obj).uv_layers)

    def uv_loops(self, obj: str, layer: str) -> list[Uv]:
        return [uv for face in self._mesh(obj).uv_layers[layer] for uv in face]

    def face_loop_ranges(self, obj: str) -> list[tuple[int, int]]:
        ranges: list[tuple[int, int]] = []
        start = 0
        for count in self._mesh(obj).face_loop_counts():
            ranges.append((start, start + count))
            start += count
        return ranges

    def material_slots(self, obj: str) -> list[str]:
        return list(self._mesh(obj).materials)

    def image_sizes(self, obj: str) -> list[tuple[int, int]]:
        return list(self._mesh(obj).image_sizes)

    def material_image_sizes(self, material: str) -> list[tuple[int, int]]:
        sizes: list[tuple[int, int]] = []
        for obj in self.objects.values():
            if material in obj.materials:
                sizes.extend(s for s in obj.image_sizes if s not in sizes)
        return sizes

    def bbox_diagonal(self, objs: list[str]) -> float:
        low, high = self.object_bbox(objs)
        return math.dist(low, high)

    def object_bbox(self, objs: list[str]) -> tuple[Vec3, Vec3]:
        points = [
            p
            for name in objs
            if self.objects[name].kind == "MESH"
            for p in self.world_vertex_positions(name)
        ]
        if not points:
            return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
        low = tuple(min(p[i] for p in points) for i in range(3))
        high = tuple(max(p[i] for p in points) for i in range(3))
        return (low[0], low[1], low[2]), (high[0], high[1], high[2])

    def world_vertex_positions(self, obj: str) -> list[Vec3]:
        mesh = self._mesh(obj)
        s, (tx, ty, tz) = mesh.scale, mesh.translation
        return [(i * s + tx, ty, tz) for i in range(mesh.verts)]

    # ------------------------------------------------------------------ mesh mutations
    def merge_material_slots(self, objs: list[str], name: str) -> None:
        self._record("merge_material_slots", tuple(objs), name)
        for obj in objs:
            self._mesh(obj).materials = [name]

    def keep_first_uv_layer(self, obj: str) -> None:
        self._record("keep_first_uv_layer", obj)
        mesh = self._mesh(obj)
        if mesh.uv_layers:
            first = next(iter(mesh.uv_layers))
            mesh.uv_layers = {first: mesh.uv_layers[first]}

    def smart_unwrap(
        self, objs: list[str], angle_deg: float, island_margin: float, pack_margin: float
    ) -> None:
        self._record("smart_unwrap", tuple(objs), angle_deg, island_margin, pack_margin)
        counts = [(name, self._mesh(name).face_loop_counts()) for name in objs]
        total = sum(len(c) for _, c in counts)
        side = max(1, math.ceil(math.sqrt(total)))
        index = 0
        for name, loop_counts in counts:
            mesh = self._mesh(name)
            layer = next(iter(mesh.uv_layers), "UVMap")
            faces: list[list[Uv]] = []
            for loops in loop_counts:
                faces.append(_atlas_face(index, side, loops, zero=self.unwrap_zero_area))
                index += 1
            mesh.uv_layers = {layer: faces}

    def clear_shapekeys(self, obj: str) -> None:
        self._record("clear_shapekeys", obj)
        self._mesh(obj).shapekeys = False

    def add_subsurf(  # noqa: PLR0917 - normative signature (plan 09 §2.3)
        self, obj: str, name: str, levels: int, kind: str, uv_smooth: str, limit_surface: bool
    ) -> None:
        self._record("add_subsurf", obj, name, levels, kind, uv_smooth, limit_surface)
        self._mesh(obj).modifiers.append(
            FakeModifier(name, "SUBSURF", levels, kind, uv_smooth, limit_surface)
        )

    def move_modifier_to_top(self, obj: str, name: str) -> None:
        self._record("move_modifier_to_top", obj, name)
        mods = self._mesh(obj).modifiers
        mod = _find_modifier(mods, name)
        mods.remove(mod)
        mods.insert(0, mod)

    def apply_modifier(self, obj: str, name: str) -> None:
        self._record("apply_modifier", obj, name)
        mesh = self._mesh(obj)
        mod = _find_modifier(mesh.modifiers, name)
        if mesh.shapekeys:
            raise RuntimeError("Modifier cannot be applied to a mesh with shape keys")
        if mod.kind == "SUBSURF":
            for _ in range(mod.levels):
                mesh.verts += 3 * mesh.faces  # closed quad mesh: V' = V + E + F, E = 2F
                mesh.faces *= 4
                mesh.uv_layers = {
                    layer: [sub for face in faces for sub in _split_face(face)]
                    for layer, faces in mesh.uv_layers.items()
                }
            mesh.tris = mesh.faces * 2
        mesh.modifiers.remove(mod)

    def shade_smooth(self, obj: str) -> None:
        self._record("shade_smooth", obj)
        self._mesh(obj).smooth = True

    # ------------------------------------------------------------------ export
    def export_glb(self, objs: list[str], path: str, **kwargs: Any) -> None:
        self._record("export_glb", tuple(objs), path)
        self._export("glb", objs, path, kwargs, FAKE_GLB_HEADER)

    def export_fbx(self, objs: list[str], path: str, **kwargs: Any) -> None:
        self._record("export_fbx", tuple(objs), path)
        self._export("fbx", objs, path, kwargs, FAKE_FBX_HEADER)

    def reset_pose(self, armature: str) -> None:
        self._record("reset_pose", armature)
        self._armature(armature)
        self.pose_resets.append(armature)

    def clean_action_names(self) -> None:
        self._record("clean_action_names")
        taken = {name for name, _root in self.raw_actions}
        renamed: list[tuple[str, str]] = []
        for name, root in self.raw_actions:
            clean = clean_action_name(name)
            if root == "OBJECT" and clean != name and clean not in taken:
                taken.add(clean)
                renamed.append((clean, root))
            else:
                renamed.append((name, root))
        self.raw_actions = renamed

    def _export(
        self, kind: str, objs: list[str], path: str, kwargs: dict[str, Any], header: bytes
    ) -> None:
        self.exports.append((kind, list(objs), path, dict(kwargs)))
        if self.export_error is not None:
            raise self.export_error
        if self.export_writes_file:
            target = Path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(header + "\n".join(objs).encode("utf-8"))

    # ------------------------------------------------------------------ baking (WP-08)
    def set_render_engine_cycles(self, device: str, samples: int) -> None:
        self._record("set_render_engine_cycles", device, samples)
        self.render = {"engine": "CYCLES", "device": device, "samples": samples}

    def configure_bake(
        self, margin_px: int, cage_extrusion: float, max_ray_distance: float
    ) -> None:
        self._record("configure_bake", margin_px, cage_extrusion, max_ray_distance)
        self.bake_settings = {
            "margin_px": margin_px,
            "cage_extrusion": cage_extrusion,
            "max_ray_distance": max_ray_distance,
        }

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
        self._record("bake_map", tuple(targets), tuple(sources), map_name)
        for target in targets:
            self._mesh(target)
        for source in sources:
            self._mesh(source)
        if emit_input is not None:
            self.rewires.append(("rewire", map_name, emit_input))
        try:
            error = self.bake_errors.get(map_name)
            if error is not None:
                raise error
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            Path(out_path).write_bytes(FAKE_PNG_BYTES)
        finally:
            if emit_input is not None:
                self.rewires.append(("restore", map_name, emit_input))
        self.baked.append(
            {
                "targets": list(targets),
                "sources": list(sources),
                "map_name": map_name,
                "bake_type": bake_type,
                "resolution": resolution,
                "out_path": out_path,
                "colorspace": colorspace,
                "emit_input": emit_input,
                "samples": self.render.get("samples"),
            }
        )

    def shader_inputs(self, obj: str) -> list[ShaderInput]:
        return list(self._mesh(obj).shader_inputs)

    def image_stats(self, path: str) -> tuple[float, float]:
        self._record("image_stats", path)
        if not Path(path).is_file():
            raise RuntimeError(f"cannot load image {path!r}")
        if Path(path).stem in self.empty_images:
            return (0.0, 0.0)
        return (0.5, 0.02)

    def set_pose_position(self, armature: str, rest: bool) -> None:
        self._record("set_pose_position", armature, rest)
        self._armature(armature).pose_rest = rest

    def transform_objects(self, objs: list[str], scale: float, translation: Vec3) -> None:
        self._record("transform_objects", tuple(objs), scale, translation)
        for name in objs:
            mesh = self._mesh(name)
            tx, ty, tz = mesh.translation
            mesh.scale *= scale
            mesh.translation = (
                tx * scale + translation[0],
                ty * scale + translation[1],
                tz * scale + translation[2],
            )

    def hide_from_render(self, obj: str) -> None:
        self._record("hide_from_render", obj)
        self.objects[obj].hide_render = True

    # ------------------------------------------------------------------ materials (WP-09)
    def bind_urp_material(
        self, objs: list[str], material_name: str, textures: dict[str, str]
    ) -> None:
        self._record("bind_urp_material", tuple(objs), material_name)
        for name in objs:
            self._mesh(name).materials = [material_name]
            self.bound_textures[name] = dict(textures)


# --------------------------------------------------------------------------- geometry helpers
def _find_modifier(mods: list[FakeModifier], name: str) -> FakeModifier:
    for mod in mods:
        if mod.name == name:
            return mod
    raise KeyError(f"no modifier named {name!r}")


def _atlas_face(index: int, side: int, loops: int, *, zero: bool) -> list[Uv]:
    """Face ``index`` as a small polygon inside cell ``index`` of a ``side x side`` grid."""
    if zero:
        return [(0.0, 0.0)] * loops
    cell = 1.0 / side
    cx, cy = (index % side + 0.5) * cell, (index // side + 0.5) * cell
    radius = cell * 0.4
    if loops == _DEFAULT_LOOPS_PER_FACE:
        return [
            (cx - radius, cy - radius),
            (cx + radius, cy - radius),
            (cx + radius, cy + radius),
            (cx - radius, cy + radius),
        ]
    return [
        (
            cx + radius * math.cos(2 * math.pi * k / loops),
            cy + radius * math.sin(2 * math.pi * k / loops),
        )
        for k in range(loops)
    ]


def _split_face(face: list[Uv]) -> list[list[Uv]]:
    """Catmull-Clark split of one UV quad into four (other polygons are copied 4x)."""
    if len(face) != _DEFAULT_LOOPS_PER_FACE:
        return [list(face) for _ in range(4)]
    (a, b, c, d) = face

    def mid(p: Uv, q: Uv) -> Uv:
        return ((p[0] + q[0]) / 2, (p[1] + q[1]) / 2)

    ab, bc, cd, da = mid(a, b), mid(b, c), mid(c, d), mid(d, a)
    centre = ((a[0] + b[0] + c[0] + d[0]) / 4, (a[1] + b[1] + c[1] + d[1]) / 4)
    return [[a, ab, centre, da], [ab, b, bc, centre], [centre, bc, c, cd], [da, centre, cd, d]]


# --------------------------------------------------------------------------- builders
def three_bone_chain() -> list[BoneInfo]:
    """``Root → Spine → Head`` along +Z, 0.5 units each (the fixture rig)."""
    bones: list[BoneInfo] = []
    names = ("Root", "Spine", "Head")
    for i, name in enumerate(names):
        z = i * 0.5
        matrix = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, z],
            [0.0, 0.0, 0.0, 1.0],
        ]
        bones.append(
            BoneInfo(
                name=name,
                parent=names[i - 1] if i else None,
                head=(0.0, 0.0, z),
                tail=(0.0, 0.0, z + 0.5),
                matrix_local=matrix,
            )
        )
    return bones


def cube_uv_atlas() -> list[list[Uv]]:
    """Six non-overlapping quads in a 3x2 grid (a perfect atlas)."""
    faces: list[list[Uv]] = []
    for i in range(6):
        u0, v0 = (i % 3) / 3 + 0.02, (i // 3) / 2 + 0.02
        u1, v1 = u0 + 1 / 3 - 0.04, v0 + 0.5 - 0.04
        faces.append([(u0, v0), (u1, v0), (u1, v1), (u0, v1)])
    return faces


def cube_uv_palette() -> list[list[Uv]]:
    """Six faces collapsed onto the same tiny quad (overlap ratio 1.0)."""
    quad: list[Uv] = [(0.10, 0.10), (0.12, 0.10), (0.12, 0.12), (0.10, 0.12)]
    return [list(quad) for _ in range(6)]


def cube_object(
    *,
    armature: str | None = "Armature",
    uv: list[list[Uv]] | None = None,
    extra_uv_layers: dict[str, list[list[Uv]]] | None = None,
    materials: tuple[str, ...] = ("cube_mat",),
    image_sizes: tuple[tuple[int, int], ...] = ((256, 256),),
    shapekeys: bool = False,
) -> FakeObject:
    """A closed quad cube: 8 verts, 6 quads, 12 tris, skinned to ``armature``."""
    layers: dict[str, list[list[Uv]]] = {}
    if uv is not None:
        layers["UVMap"] = uv
    layers.update(extra_uv_layers or {})
    return FakeObject(
        kind="MESH",
        verts=8,
        faces=6,
        tris=12,
        uv_layers=layers,
        materials=list(materials),
        modifiers=[FakeModifier("Armature", "ARMATURE")] if armature else [],
        shapekeys=shapekeys,
        parent_armature=armature,
        image_sizes=list(image_sizes),
    )


def armature_object(bones: list[BoneInfo] | None = None) -> FakeObject:
    return FakeObject(
        kind="ARMATURE", bones=list(bones or three_bone_chain()), data_name="Armature"
    )


def scene_with_rigged_mesh(
    *,
    uv: bool = True,
    shapekeys: bool = False,
    image_sizes: tuple[tuple[int, int], ...] = ((256, 256),),
    materials: tuple[str, ...] = ("cube_mat",),
    extra_uv_layers: dict[str, list[list[Uv]]] | None = None,
) -> FakeScene:
    """``Armature`` + skinned ``Cube`` with a perfect atlas and one ``idle`` action."""
    cube = cube_object(
        uv=cube_uv_atlas() if uv else None,
        extra_uv_layers=extra_uv_layers,
        materials=materials,
        image_sizes=image_sizes,
        shapekeys=shapekeys,
    )
    return FakeScene(
        {"Armature": armature_object(), "Cube": cube},
        actions=[("Armature|Armature|idle", "OBJECT")],
    )


def scene_palette_uv() -> FakeScene:
    """Cube whose UVs are collapsed onto a 16x4 palette texture."""
    cube = cube_object(uv=cube_uv_palette(), materials=("palette_mat",), image_sizes=((16, 4),))
    return FakeScene(
        {"Armature": armature_object(), "Cube": cube},
        actions=[("Armature|Armature|idle", "OBJECT")],
    )


def scene_two_meshes_shapekeys() -> FakeScene:
    """``Body`` (no keys) and ``Face`` (shape keys), both skinned; plus the ``Key|`` action."""
    return FakeScene(
        {
            "Armature": armature_object(),
            "Body": cube_object(uv=cube_uv_atlas(), materials=("body_mat",)),
            "Face": cube_object(uv=cube_uv_atlas(), materials=("face_mat",), shapekeys=True),
        },
        actions=[("Armature|Armature|idle", "OBJECT"), ("Key|Armature|idle", "KEY")],
    )


def scene_no_armature() -> FakeScene:
    return FakeScene({"Cube": cube_object(armature=None, uv=cube_uv_atlas())})


def scene_two_armatures() -> FakeScene:
    return FakeScene(
        {
            "Armature": armature_object(),
            "Armature.001": armature_object(),
            "Cube": cube_object(uv=cube_uv_atlas()),
        }
    )


def scene_unskinned_extra() -> FakeScene:
    """Rigged cube plus an unskinned ``Prop`` mesh."""
    scene = scene_with_rigged_mesh()
    scene.objects["Prop"] = cube_object(armature=None, uv=cube_uv_atlas(), materials=("prop_mat",))
    return scene


# --------------------------------------------------------------------------- bake builders (WP-08)
def principled_inputs(
    *,
    basecolor_linked: bool = True,
    normal_linked: bool = True,
    roughness: float | None = 0.5,
    roughness_linked: bool = False,
    metallic: float | None = 0.0,
    metallic_linked: bool = False,
    occlusion_linked: bool = False,
    emission: tuple[float, float, float] | None = None,
    emission_linked: bool = False,
    emission_strength: float = 1.0,
) -> list[ShaderInput]:
    """Shader inputs as the glTF importer would expose them on a Principled material.

    ``None`` for ``roughness``/``metallic`` omits the socket entirely (not a Principled material);
    ``emission=None`` leaves ``Emission Color`` black.
    """
    inputs = [ShaderInput("Base Color", basecolor_linked, (0.8, 0.8, 0.8, 1.0))]
    if normal_linked:
        inputs.append(ShaderInput("Normal", True, ()))
    if roughness is not None:
        inputs.append(ShaderInput("Roughness", roughness_linked, (roughness,)))
    if metallic is not None:
        inputs.append(ShaderInput("Metallic", metallic_linked, (metallic,)))
    if occlusion_linked:
        inputs.append(ShaderInput("Occlusion", True, (1.0,)))
    colour = (*(emission or (0.0, 0.0, 0.0)), 1.0)
    inputs.append(ShaderInput("Emission Color", emission_linked, colour))
    inputs.append(ShaderInput("Emission Strength", False, (emission_strength,)))
    return inputs


def source_object(
    *,
    inputs: list[ShaderInput] | None = None,
    scale: float = 1.0,
    translation: Vec3 = (0.0, 0.0, 0.0),
    materials: tuple[str, ...] = ("tripo_mat",),
) -> FakeObject:
    """An unskinned provider mesh (added by ``import_glb``) with the cube's 8 synthetic vertices."""
    return FakeObject(
        kind="MESH",
        verts=8,
        faces=6,
        tris=12,
        uv_layers={"UVMap": cube_uv_atlas()},
        materials=list(materials),
        shader_inputs=list(principled_inputs() if inputs is None else inputs),
        scale=scale,
        translation=translation,
    )


def scene_for_bake(
    *,
    source: FakeObject | None = None,
    source_name: str = "result",
    targets: int = 1,
) -> FakeScene:
    """Prepared scene (``Armature`` + skinned cube(s) with one material) and a pending GLB source.

    The source object is only added when the job calls ``import_glb``.
    """
    scene = scene_with_rigged_mesh(materials=("cube_rigged_mat",))
    for index in range(1, targets):
        scene.objects[f"Cube.{index:03d}"] = cube_object(
            uv=cube_uv_atlas(), materials=("cube_rigged_mat",)
        )
    scene.glb_objects = {source_name: source if source is not None else source_object()}
    return scene
