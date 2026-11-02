"""``SceneApi``: the Blender-side port every job talks to (plan 09 §2.3, 03 §3).

``bpy_scene.BpyScene`` implements it with ``bpy``/``bmesh``; ``tests/fakes/fake_scene.py``
implements it over an in-memory scene graph. Jobs never import ``bpy`` themselves, so all job
logic is unit-testable under CPython. Object names are the handles (Blender object names are
unique per file).

Besides the ``prepare_model`` methods of plan 09 §2.3 the protocol declares the bake methods
(plan 11 §5.2, appendix B §5) and the material-binding/export methods (plan 12 §1.2, appendix B
§6) that WP-08 and WP-09 implement against. Standard library only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

Vec3 = tuple[float, float, float]
Uv = tuple[float, float]


@dataclass(frozen=True)
class BoneInfo:
    """Rest-pose data of one bone in armature space (appendix B §4).

    Attributes:
        name: Bone name (unique within the armature).
        parent: Parent bone name or ``None`` for a root bone.
        head: ``bone.head_local``.
        tail: ``bone.tail_local``.
        matrix_local: ``bone.matrix_local`` as a 4x4 row-major nested list.
    """

    name: str
    parent: str | None
    head: Vec3
    tail: Vec3
    matrix_local: list[list[float]]


@dataclass(frozen=True)
class ShaderInput:
    """One bake-relevant shader socket of a material (plan 11 §5.2 step 6, appendix B §5).

    Attributes:
        name: Socket name: a Principled BSDF input (``Base Color``, ``Metallic``, ``Roughness``,
            ``Normal``, ``Emission Color``, ``Emission Strength``) or the glTF settings group's
            ``Occlusion`` input.
        linked: True when a node link drives the socket (texture data); False for a constant.
        value: The socket's default value flattened to a tuple (scalar → 1 element, colour → 4);
            empty for sockets without a default (``Normal``).
    """

    name: str
    linked: bool
    value: tuple[float, ...] = ()


class SceneApi(Protocol):
    """Everything a job may do to the Blender scene. Object names identify objects."""

    # ------------------------------------------------------------------ scene lifecycle
    def reset(self) -> None:
        """Empty factory scene (``read_factory_settings(use_empty=True)``)."""
        ...

    def import_fbx(self, path: str, **kwargs: Any) -> None:  # noqa: ANN401 - operator kwargs
        """Import an FBX with ``fbx_settings.FBX_IMPORT_KWARGS``; imported objects are unhidden.

        Raises:
            RuntimeError: The importer reported an error (jobs map it to ``E_BJ_IMPORT_FAILED``).
        """
        ...

    def import_glb(self, path: str, **kwargs: Any) -> None:  # noqa: ANN401 - operator kwargs
        """Import a glTF/GLB (appendix B §2). Imported objects are unhidden.

        Raises:
            RuntimeError: The importer reported an error.
        """
        ...

    def open_blend(self, path: str) -> None:
        """Open a ``.blend`` file, replacing the current scene."""
        ...

    def save_blend(self, path: str) -> None:
        """Save the current file to ``path`` (``save_as_mainfile``), creating parent dirs."""
        ...

    def blender_version(self) -> tuple[int, int, int]:
        """``bpy.app.version``."""
        ...

    # ------------------------------------------------------------------ queries
    def armatures(self) -> list[str]:
        """Names of all ARMATURE objects, sorted."""
        ...

    def meshes(self) -> list[str]:
        """Names of all MESH objects, sorted."""
        ...

    def skinned_meshes(self, armature: str) -> list[str]:
        """Sorted names of meshes bound to ``armature`` (Armature modifier or armature parent)."""
        ...

    def armature_data_name(self, armature: str) -> str:
        """Name of the armature *data-block* behind object ``armature``."""
        ...

    def bones(self, armature: str) -> list[BoneInfo]:
        """Rest-pose bones of ``armature`` in Blender's collection order (not fingerprint order)."""
        ...

    def actions(self) -> list[str]:
        """Sorted names of object-level actions with import prefixes stripped.

        Blender names imported FBX takes ``<object>|<take>`` and the take itself is often already
        ``<object>|<action>`` (appendix B §8.1), so everything up to and including the last ``|`` is
        removed. Actions whose ``id_root`` is not ``OBJECT`` (e.g. the ``Key|…`` shape-key channel
        actions of appendix B §8.2) are excluded.
        """
        ...

    def mesh_counts(self, obj: str) -> tuple[int, int, int]:
        """``(vertices, faces, triangles)`` of the base mesh; triangles = ``sum(len(poly) - 2)``."""
        ...

    def has_shapekeys(self, obj: str) -> bool:
        """True when the mesh carries shape keys (which block ``apply_modifier``)."""
        ...

    def uv_layers(self, obj: str) -> list[str]:
        """UV layer names in slot order (the first one is the texturing layer)."""
        ...

    def uv_loops(self, obj: str, layer: str) -> list[Uv]:
        """Per-loop UVs of ``layer`` (``len == number of mesh loops``)."""
        ...

    def face_loop_ranges(self, obj: str) -> list[tuple[int, int]]:
        """Per-face half-open loop ranges ``(loop_start, loop_start + loop_total)``."""
        ...

    def material_slots(self, obj: str) -> list[str]:
        """Material names per slot (``""`` for an empty slot)."""
        ...

    def image_sizes(self, obj: str) -> list[tuple[int, int]]:
        """``(width, height)`` of every loadable image referenced by the object's materials."""
        ...

    def material_image_sizes(self, material: str) -> list[tuple[int, int]]:
        """``(width, height)`` of the images referenced by the named material's node tree."""
        ...

    def bbox_diagonal(self, objs: list[str]) -> float:
        """Diagonal of the world-space axis-aligned box around all ``objs`` (appendix B §3)."""
        ...

    def object_bbox(self, objs: list[str]) -> tuple[Vec3, Vec3]:
        """World-space ``(min, max)`` corners of the box around all ``objs``."""
        ...

    def world_vertex_positions(self, obj: str) -> list[Vec3]:
        """World-space base-mesh vertex positions (``matrix_world @ vertex.co``)."""
        ...

    # ------------------------------------------------------------------ mesh mutations
    def merge_material_slots(self, objs: list[str], name: str) -> None:
        """Keep one material named ``name`` on every object; all polygons use slot 0."""
        ...

    def keep_first_uv_layer(self, obj: str) -> None:
        """Remove every UV layer except the first (lightmap/extra UVs)."""
        ...

    def smart_unwrap(
        self, objs: list[str], angle_deg: float, island_margin: float, pack_margin: float
    ) -> None:
        """Smart UV Project + pack islands over all ``objs`` together (one shared UV space).

        Creates a UV layer when none exists.
        """
        ...

    def clear_shapekeys(self, obj: str) -> None:
        """Remove all shape keys (``shape_key_clear``), keeping the basis geometry."""
        ...

    def add_subsurf(  # noqa: PLR0917 - normative signature (plan 09 §2.3)
        self, obj: str, name: str, levels: int, kind: str, uv_smooth: str, limit_surface: bool
    ) -> None:
        """Append a Subdivision Surface modifier (``kind``: ``catmull`` | ``simple``)."""
        ...

    def move_modifier_to_top(self, obj: str, name: str) -> None:
        """Move modifier ``name`` to index 0 (above the Armature modifier; appendix B §3)."""
        ...

    def apply_modifier(self, obj: str, name: str) -> None:
        """Apply modifier ``name`` (object must be visible; shape keys must be gone)."""
        ...

    def shade_smooth(self, obj: str) -> None:
        """``use_smooth = True`` on every polygon."""
        ...

    # ------------------------------------------------------------------ export
    def export_glb(self, objs: list[str], path: str, **kwargs: Any) -> None:  # noqa: ANN401
        """Export exactly ``objs`` with ``fbx_settings.GLB_EXPORT_KWARGS``.

        Unknown kwargs are dropped by feature detection (appendix B §2). Armature deformation is
        bypassed for the export so the static mesh is the rest shape regardless of the current
        frame.

        Raises:
            RuntimeError: The exporter reported an error (``E_BJ_EXPORT_FAILED`` in jobs).
        """
        ...

    def export_fbx(self, objs: list[str], path: str, **kwargs: Any) -> None:  # noqa: ANN401
        """Export exactly ``objs`` (armature active) with ``fbx_settings.fbx_export_kwargs``.

        Raises:
            RuntimeError: The exporter reported an error.
        """
        ...

    def reset_pose(self, armature: str) -> None:
        """Unassign the active action and put every pose bone back to rest (appendix B §8.3).

        The FBX exporter writes armature node transforms from the *current pose*; only skin
        clusters carry the rest pose, so a clip file exported while posed re-imports with that
        pose as its rest pose. Jobs call this right before ``export_fbx``;
        ``bake_anim_use_all_actions`` still writes every action because the exporter assigns
        them one by one itself. Rest bones (``armature.data.bones``) are never touched, so the
        rig fingerprint is unchanged.
        """
        ...

    def clean_action_names(self) -> None:
        """Rename object actions to their import-prefix-stripped names (appendix B §8.1).

        The FBX exporter writes every take as ``<object>|<action>`` and the importer prefixes
        again, so without this each round trip grows the name (``Armature|Armature|idle`` →
        ``Armature|Armature|Armature|idle`` …) and Unity, which names clips after takes, sees
        the pile-up. Renaming ``Armature|Armature|idle`` → ``idle`` before export makes the
        delivered take ``Armature|idle``, exactly what the vendor file carried. An action whose
        clean name is already taken is left alone. Action names are not part of the rig
        fingerprint (``actions()`` is already stripped).
        """
        ...

    # ------------------------------------------------------------------ baking (WP-08)
    def set_render_engine_cycles(self, device: str, samples: int) -> None:
        """Cycles, ``device`` ``CPU``|``GPU``, fixed seed, no denoise, Standard view transform."""
        ...

    def configure_bake(
        self, margin_px: int, cage_extrusion: float, max_ray_distance: float
    ) -> None:
        """Selected-to-active bake settings (appendix B §5); distances in world units."""
        ...

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
        """Bake one map from ``sources`` onto the UVs of ``targets`` and save a PNG at ``out_path``.

        All targets share one atlas (prepare merged their materials and unwrapped them together),
        so one image is created and every target is baked into it in turn (the image is cleared
        before the first target only).

        Args:
            targets: Our prepared meshes (each made active in turn; must have UVs and a material).
            sources: Provider meshes (selected) whose surface is sampled.
            map_name: Image name (``basecolor`` …).
            bake_type: Cycles bake type (``NORMAL``, ``ROUGHNESS``, ``EMIT``, ``DIFFUSE``).
            resolution: Square image size in pixels.
            out_path: PNG destination.
            colorspace: ``sRGB`` or ``Non-Color``.
            emit_input: For ``EMIT`` bakes, the Principled/glTF input to route through an
                Emission shader on the sources (``Base Color``, ``Metallic``, ``Occlusion``);
                ``None`` for native bake types (``emission`` uses the material's own emission
                through native ``EMIT``). The rewiring is undone before the call returns.

        Raises:
            RuntimeError: ``bpy.ops.object.bake`` failed.
        """
        ...

    def shader_inputs(self, obj: str) -> list[ShaderInput]:
        """Bake-relevant sockets of every material on ``obj`` (empty when it has no material).

        Reports the Principled BSDF inputs ``Base Color``, ``Metallic``, ``Roughness``,
        ``Normal``, ``Emission Color``, ``Emission Strength`` and the ``Occlusion`` input of the
        glTF importer's settings group when present. Jobs decide from these which maps to bake.
        """
        ...

    def image_stats(self, path: str) -> tuple[float, float]:
        """``(mean, variance)`` of the RGB channels of the image file at ``path`` (0..1 floats).

        Used to detect empty bakes (``E_BJ_BAKE_EMPTY``: both ≈ 0 where the source had data).

        Raises:
            RuntimeError: The file cannot be loaded as an image.
        """
        ...

    def transform_objects(self, objs: list[str], scale: float, translation: Vec3) -> None:
        """``matrix_world = T @ S @ matrix_world`` for each object (source alignment)."""
        ...

    def hide_from_render(self, obj: str) -> None:
        """``hide_render = True`` (keeps the armature out of bakes)."""
        ...

    def set_pose_position(self, armature: str, rest: bool) -> None:
        """``armature.data.pose_position = "REST"`` (``rest=True``) or ``"POSE"``.

        The prepared meshes keep their Armature modifier and the imported action may pose them, so
        bakes run in rest position to match the static GLB the provider textured.
        """
        ...

    # ------------------------------------------------------------------ materials (WP-09)
    def bind_urp_material(
        self, objs: list[str], material_name: str, textures: dict[str, str]
    ) -> None:
        """Replace the materials of ``objs`` by one Principled material wired to ``textures``.

        ``textures`` maps ``basecolor`` | ``normal`` | ``metallic_smoothness`` | ``emission`` to
        PNG paths (appendix B §6); missing keys leave the corresponding input unlinked. Image
        paths are stored relative to the blend file.
        """
        ...
