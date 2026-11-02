# Appendix B — Blender-Side Implementation Notes (bpy 4.2 / 4.5)

Reference material for WP-03/04/08/09. Everything here is wrapped by `BpyScene`; jobs never call `bpy`
directly.

## 1. Headless invocation

```
blender -b --factory-startup --python-exit-code 2 --python <src>/tinyworld_pipeline/blender_jobs/entry.py -- --job job.json --result result.json
```

- `-b` background; `--factory-startup` ignores user prefs/add-ons (the FBX and glTF I/O add-ons are
  bundled and enabled by default in factory settings — `probe` asserts `bpy.ops.import_scene.fbx` and
  `bpy.ops.import_scene.gltf` exist).
- `--python-exit-code 2` makes Blender exit 2 when the script raises.
- Arguments after `--` are in `sys.argv[sys.argv.index("--")+1:]`.
- `entry.py` does `sys.path.insert(0, os.environ["GP_SRC_DIR"])` before importing
  `tinyworld_pipeline.blender_jobs.*`; the package `__init__` must not import third-party modules.
- Reset the scene explicitly at job start: `bpy.ops.wm.read_factory_settings(use_empty=True)`.

## 2. Import / export operators

- FBX import: `bpy.ops.import_scene.fbx(filepath=..., **FBX_IMPORT_KWARGS)` (kwargs in `09` §2.6).
  Blender creates one `Action` per animation take (`bpy.data.actions`), assigned to the armature object.
  Multiple takes become multiple actions; only the last stays assigned — export uses
  `bake_anim_use_all_actions=True` so all are written.
- FBX export: `bpy.ops.export_scene.fbx(filepath=..., **FBX_EXPORT_KWARGS)` (`12` §1.2). Select the
  armature + meshes first (`obj.select_set(True)`, `view_layer.objects.active = armature`).
- glTF import: `bpy.ops.import_scene.gltf(filepath=..., merge_vertices=False, import_shading="NORMALS",
  bone_heuristic="BLENDER")`. The importer creates one object per mesh; materials get Principled nodes and
  a `glTF Material Output` node group for occlusion.
- glTF export: `bpy.ops.export_scene.gltf(filepath=..., **GLB_EXPORT_KWARGS)` (`09` §2.5). Property
  names changed across 4.x (`export_colors` was removed in 4.2 in favour of `export_vertex_color`):
  `BpyScene.export_glb` builds kwargs by checking `bpy.ops.export_scene.gltf.get_rna_type().properties`
  and dropping unknown keys (log at DEBUG). This is the one place feature-detection is allowed.
- Blender 5.x: operator names may change (new C++ FBX importer). `probe` reports availability; runner
  refuses unsupported versions unless configured.

## 3. Mesh operations

- **Modifier order matters.** The Subdivision modifier must be above the Armature modifier so the
  armature deforms the subdivided mesh *after* weights are interpolated:
  `mod = obj.modifiers.new("GP_Subdiv", "SUBSURF"); mod.subdivision_type = "CATMULL_CLARK"|"SIMPLE";
  mod.levels = mod.render_levels = n; mod.uv_smooth = "PRESERVE_BOUNDARIES"; mod.use_limit_surface = False`
  then `bpy.ops.object.modifier_move_to_index(modifier=mod.name, index=0)` (4.x API) and
  `bpy.ops.object.modifier_apply(modifier=mod.name)` with `obj` active and in OBJECT mode.
- **Shape keys block modifier application** (`RuntimeError: Modifier cannot be applied to a mesh with
  shape keys`). Check `obj.data.shape_keys` first; `obj.shape_key_clear()` removes them (policy `drop`).
- Vertex groups (skin weights) survive subdivision: Blender interpolates `MDeformVert` weights per new
  vertex. Verify in integration tests by checking that every vertex has ≥ 1 group with weight > 0 after
  apply (`sum(len(v.groups) == 0 for v in me.vertices) == 0`).
- **Shade smooth** in 4.1+: `use_auto_smooth` no longer exists. Set `poly.use_smooth = True` for all
  polygons (`me.polygons.foreach_set("use_smooth", [True]*len(me.polygons))`). If a "Smooth by Angle"
  modifier exists on the vendor mesh, apply or remove it before subdividing (it is a geometry-nodes
  modifier; applying it bakes custom normals, which are fine for FBX export).
- Triangle count: `sum(len(p.vertices) - 2 for p in me.polygons)`.
- Material slot merge: pick the first slot's material, rename to `<model_id>_mat`, set every polygon's
  `material_index = 0`, remove the other slots (`bpy.ops.object.material_slot_remove` with active index
  loop, or `obj.data.materials.pop(index=i)` from the end).
- UV layers: `me.uv_layers`; keep `me.uv_layers[0]`, remove others via `me.uv_layers.remove(layer)`.
  Per-loop UVs: `layer.data[i].uv` (or `layer.uv` attribute in 4.x: `me.attributes[layer.name].data`).
  Loops per face: `poly.loop_start`, `poly.loop_total`.
- Smart UV Project must run in EDIT mode with faces selected; for several objects, select all of them
  and enter multi-object edit mode so islands are packed into one shared space:
  `bpy.ops.object.mode_set(mode="EDIT"); bpy.ops.mesh.select_all(action="SELECT");
  bpy.ops.uv.smart_project(angle_limit=math.radians(66), island_margin=0.02, area_weight=0.0,
  correct_aspect=True, scale_to_bounds=False); bpy.ops.uv.pack_islands(rotate=True, margin=0.01);
  bpy.ops.object.mode_set(mode="OBJECT")`. Operators need a valid context in background mode; use
  `with bpy.context.temp_override(active_object=obj, selected_objects=objs, selected_editable_objects=objs)`.
- Bounding box diagonal: from `obj.bound_box` corners transformed by `obj.matrix_world`, min/max over
  all objects.

## 4. Armature / rig fingerprint

- Bones: `arm.data.bones` (rest pose). `bone.head_local`, `bone.tail_local` (armature space),
  `bone.matrix_local` (4×4 rest pose in armature space), `bone.parent`.
- Depth-first, name-sorted traversal from `[b for b in bones if b.parent is None]`.
- Actions: `[a.name for a in bpy.data.actions]`; frame ranges `a.frame_range`.
- Never enter POSE mode or alter `arm.matrix_world`, `arm.data.transform(...)`, or bone roll.

## 5. Baking (Cycles) — selected-to-active

```python
scene.render.engine = "CYCLES"
scene.cycles.device = "CPU"   # or "GPU" if GP_BLENDER_CYCLES_DEVICE=GPU and a device is available
scene.cycles.samples = cfg.samples ; scene.cycles.use_denoising = False
scene.view_settings.view_transform = "Standard"; scene.view_settings.look = "None"
scene.display_settings.display_device = "sRGB"
bake = scene.render.bake
bake.use_selected_to_active = True; bake.cage_extrusion = ext; bake.max_ray_distance = dist
bake.margin = cfg.margin_px; bake.margin_type = "EXTEND"; bake.use_clear = True
bake.use_pass_direct = False; bake.use_pass_indirect = False; bake.use_pass_color = True   # DIFFUSE colour only
# per map:
img = bpy.data.images.new(name, res, res, alpha=False, float_buffer=False)
img.colorspace_settings.name = "sRGB" if map in {"basecolor","emission"} else "Non-Color"
node = target_mat.node_tree.nodes.new("ShaderNodeTexImage"); node.image = img; target_mat.node_tree.nodes.active = node
# select source objects, make target active:
bpy.ops.object.bake(type=BAKE_TYPE, pass_filter={"COLOR"} if DIFFUSE else set(), use_selected_to_active=True,
                    margin=cfg.margin_px, use_clear=True, target="IMAGE_TEXTURES")
img.filepath_raw = str(out_path); img.file_format = "PNG"; img.save()
```

- `NORMAL` bake with `normal_space="TANGENT"` from a source whose material has a Normal Map node
  captures the source's normal-map detail onto the target's tangent space. Target needs UVs (it has) and
  the exporter's tangents are computed by Unity (`use_tspace=False` on export is fine).
- `EMIT` trick: for metallic/occlusion/emission, temporarily rewire the source material — connect the
  relevant texture (or the Principled input's link) to an `Emission` shader → Material Output; bake type
  `EMIT`; restore links afterwards. Use 1 sample.
- glTF importer stores occlusion in a `glTF Material Output`/`glTF Settings` node group input named
  `Occlusion`; locate by node type `ShaderNodeGroup` and group name startswith `glTF`.
- `ROUGHNESS` bake type exists natively; `METALLIC` does not (hence EMIT).
- Alignment: if Tripo normalised the mesh, scale/translate the **source** objects so their bbox matches
  the target's (`obj.matrix_world = T @ S @ obj.matrix_world`), never the target.
- Hide the armature from render (`arm.hide_render = True`) and make sure only source objects are
  selected besides the active target.

## 6. Material binding for export (`export_rigged`)

Principled BSDF: `Base Color` ← Image (sRGB); `Normal` ← `ShaderNodeNormalMap` ← Image (Non-Color);
`Metallic` ← `ShaderNodeSeparateColor`.Red of metallic_smoothness; `Roughness` ← `ShaderNodeInvert`
of the image's Alpha; `Emission Color` ← Image (sRGB), `Emission Strength` 1.0 when present.
Image nodes reference files in `work/textures/urp/` with relative paths (`path_mode="RELATIVE"`), so the
`.blend` stays portable inside the model folder.

## 7. Gotchas collected

- Background mode has no window; any operator that needs `region`/`area` must be called with
  `temp_override`. Mesh/object/uv operators listed above work with the overrides shown.
- `bpy.ops.object.modifier_apply` requires the object to be active **and** visible/selectable in the
  view layer; unhide (`hide_set(False)`, `hide_viewport=False`) imported objects first.
- Imported FBX objects may arrive with non-identity object transforms (scale 0.01 + rotation). Do not
  "apply transforms" on the armature; the exporter handles it. For the static GLB, export objects as-is;
  positions in the GLB are world-space anyway.
- Images saved via `img.save()` need `filepath_raw` set and `file_format`; for 16-bit normal maps
  use `scene.render.image_settings.color_depth = "16"` and `img.save_render(path, scene=scene)`.
- `bpy.data.images.new(..., float_buffer=True)` + Non-Color is the safest for data maps; convert on save.
- Deterministic results: bakes are deterministic for `EMIT`/`DIFFUSE` colour passes; `cycles.seed` fixed
  to 0 anyway.
- Keep Blender's console quiet: `bpy.app.debug_value = 0`; capture stdout to the log file via the runner.

## 8. Observed on Blender 4.5.12 while generating the test fixtures (2026-11-02; WP-16)

These are facts, verified by `tests/integration/blender/test_fixtures_load.py`, that WP-04 and WP-09
MUST account for:

1. **Imported action names are double-prefixed.** The exporter writes the take as
   `<armature_object>|<action>`; the importer then names the action `<object>|<take>`, so a source
   action `idle` on object `Armature` comes back as `Armature|Armature|idle`. `rig_fingerprint.actions`
   will contain these names; WP-09's clip naming MUST strip everything up to and including the last
   `|` before deriving `<clip>` (and the RFC's "source actions ⊆ output actions" check compares the
   stripped names).
2. **Shape-keyed meshes produce an extra `Key|…` action on import** (the baked blend-shape channel),
   e.g. `Key|Armature|idle`. Action lists MUST ignore actions whose `id_root != "OBJECT"` (or whose
   name starts with `Key|`).
3. **Armature node transforms are written from the current pose, not the rest pose.** Only skin
   clusters carry the rest pose. For an armature with no skinned mesh (companion clip files) the
   importer takes the rest pose from the posed nodes, so a clip exported while posed re-imports with a
   different rest pose. `roundtrip_animation` MUST set `arm.animation_data.action = None` (and reset
   pose transforms) before export; the RFC's animation-mode leniency on `matrix_local` (12 §1.4) exists
   because vendor clip files may already carry this artefact.
4. **The 4.5.12 FBX exporter prints a debug tuple to stdout** (`(bpy.data.armatures['Armature'], 'POSE')`)
   for every skinned export. Never parse Blender stdout; results go through `result.json` only.
5. **Byte-deterministic FBX export is impossible** (`CreationTimeStamp`, and element UUIDs from
   Python's salted `hash()`); with `PYTHONHASHSEED=0` and `--python-use-system-env` only the
   timestamp bytes differ. Fingerprints MUST never hash FBX *outputs* of the pipeline for caching;
   hashing vendor *inputs* is fine. (`export` stage fingerprint hashes inputs only — unchanged.)
   `path_mode="RELATIVE"` also embeds the absolute texture path alongside the relative one.
6. **Slotted actions (4.4+)** are created automatically by `pose_bone.keyframe_insert`; an unassigned
   action with `use_fake_user` is still exported by `bake_anim_use_all_actions=True`.
7. Every kwarg in `FBX_IMPORT_KWARGS` / `FBX_EXPORT_KWARGS` is accepted by 4.5.12 unchanged.
9. **Take names grow on every round trip.** `prepared.blend` already holds `Armature|Armature|idle`; a plain
   export re-imports as `Armature|Armature|Armature|Armature|idle`. `export_rigged` and `roundtrip_animation`
   therefore rename OBJECT actions to their stripped names (`SceneApi.clean_action_names`) before export,
   so delivered takes read `Armature|idle` exactly like the vendor file (D-17). Unity names clips after
   takes — Spike S2 decides whether the postprocessor renames them to the manifest `clip_name`.
10. FBX material texture slots: only images wired directly into Principled inputs are exported
    (base colour — also duplicated as the transparency texture — and the normal map). Metallic/roughness
    behind SeparateColor/Invert nodes are not exported as texture slots; irrelevant because Unity builds
    materials from `manifest.json`.
8. `fake-bpy-module-4.5` stubs: `bpy.context.scene`/`view_layer`, `Object.pose`, `ShaderNode.outputs`,
   `Image.colorspace_settings` are `Optional` and need narrowing; `mathutils.Vector` is not a
   `Sequence[float]` (use `.to_tuple()`); `Modifier.object` and `ShapeKeyPoint.co` need
   `# type: ignore[attr-defined]` with a reason.
