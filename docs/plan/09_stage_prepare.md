# 09 — Stages `discover` and `prepare`

## 1. Stage `discover` (orchestrator only)

**Purpose:** turn a model folder into a `ResolvedModel` and establish the baseline state.

Steps (`pipeline/stages/discover.py`):

1. Parse `model.md` → `ModelDocument` (`library/frontmatter_io.py`).
2. Parse `style/style.md` → `StyleDocument`.
3. Resolve config (`config/layering.py`).
4. Resolve `primary_fbx` (explicit or the single top-level `.fbx`) and `animation_fbx` (globs, sorted).
5. Compose prompt (`texturing/prompt_builder.py`):
   `template.format_map(sections ∪ {"model_prompt": body["prompt"], "display_name", "tags"})`,
   collapse runs of blank lines, strip, enforce length rule, compute sha.
6. Validate (all rules in `05_…` §8). Any error → `ValidationError` (stage fails, nothing written).
7. Load or create `ModelState`; if `pipeline_version` major differs → mark all stages pending with
   `extra.invalidated_by="pipeline_version"`.

Fingerprint: sha256 of (`model.md` bytes, `library.toml` bytes, `style.md` bytes, env/CLI override
dump, `pipeline_version`). Outputs: none on disk besides state. `discover` is cheap and always re-runs
when its fingerprint changes; downstream stages have their own fingerprints so a comment edit in
`model.md` does **not** trigger Blender work.

## 2. Stage `prepare` (orchestrator + Blender job `prepare_model`)

**Purpose:** produce `work/prepared.blend` (rigged, subdivided, unwrapped) and `work/static.glb`
(unrigged copy for Tripo), plus `work/prepare_report.json` with the source rig fingerprint.

### 2.1 Orchestrator side (`pipeline/stages/prepare.py`)

```python
class PrepareStage:  # NORMATIVE
    name = "prepare"
    def input_fingerprint(self, m, ctx):
        return sha_many(file_sha(m.primary_fbx), cfg_dump(m.config.geometry), cfg_dump(m.config.uv),
                        m.config.export.fbx_scale_option, BLENDER_JOBS_CODE_SHA)
    def run(self, m, ctx):
        params = PrepareModelParams(...)
        result = ctx.blender.run(JobRequest(job="prepare_model", params=params.model_dump(mode="json"), ...))
        data = PrepareModelData.model_validate(result.data)          # raises on protocol drift
        ctx.verify_file_sha(m.paths.work / "static.glb", data.static_glb_sha256)
        write_json(m.paths.work / "prepare_report.json", data)
        return StageOutcome(status="succeeded", outputs=["work/prepared.blend","work/static.glb","work/prepare_report.json"],
                            extra={"triangles_after": data.mesh.triangles_after, "uv_decision": data.uv.decision})
```

`BLENDER_JOBS_CODE_SHA` is the sha over all files in `blender_jobs/` computed lazily on first use and
cached (`19` §2), so changing
Blender-side code invalidates `prepare` (and downstream) automatically.

### 2.2 Blender job `prepare_model` — algorithm (normative order)

```
1  import_fbx(source_fbx, FBX_IMPORT_KWARGS)                      # fbx_settings.py
2  arms = armatures(); meshes = skinned_meshes(arms)
   assert len(arms) == 1                     → E_BJ_NO_ARMATURE / E_BJ_MULTI_ARMATURE
   assert len(meshes) >= 1                   → E_BJ_NO_SKINNED_MESH
   warn on unskinned meshes (exported as-is, not subdivided unless `geometry.include_unskinned`)
3  rig = rig_fingerprint(arm)                                     # BEFORE any mesh edits
4  if geometry.merge_material_slots: merge_material_slots(meshes) # one material "<model_id>_mat"
5  if geometry.keep_only_one_uv_layer: keep_first_uv_layer(meshes)  # remove lightmap/extra UVs
6  uv_stats = analyse_uv(meshes, palette_hint_textures, uv_cfg)
   decision = decide_uv(uv_stats, uv_cfg)                         # pure (uv_heuristics.py)
   if decision in {"unwrapped","created"}: smart_unwrap(meshes, uv_cfg)  # on the BASE mesh
7  for mesh in meshes:
       if has_shapekeys(mesh):
           policy fail → E_BJ_SHAPEKEYS (ok=False, data has mesh names)
           policy drop → clear_shapekeys(mesh); warn
           policy skip → continue (mesh not subdivided; warn)
       if geometry.subdivision_levels > 0:
           add_subsurf(mesh, levels, type, uv_smooth="PRESERVE_BOUNDARIES", use_limit_surface=False)
           move_modifier_to_top(mesh, "GP_Subdiv")                 # ABOVE Armature modifier
           apply_modifier(mesh, "GP_Subdiv")
       if geometry.shade_smooth: shade_smooth(mesh)                # polygons.use_smooth = True
8  mesh_stats = collect_mesh_stats(meshes)
9  save_blend(out_blend)                                          # rigged intermediate
10 export_static_glb(meshes, out_static_glb)                      # see 2.5
11 static fingerprint: triangles, unique positions (computed from exported GLB by re-reading? No —
   computed from Blender mesh data after triangulation; orchestrator recomputes from GLB in verify
   and cross-checks)
12 write result.json with PrepareModelData
```

Steps 4–7 operate on **evaluated base meshes**, never on the armature. The armature, its bones, its
rest pose, its object transform and all actions are untouched; this invariant is asserted by step 12
re-computing the rig fingerprint and comparing it to step 3 (`E_BJ_RIG_MUTATED` if different).

### 2.3 `SceneApi` (Blender-side port) — normative methods used by all jobs

```python
class SceneApi(Protocol):
    # scene lifecycle
    def reset(self) -> None
    def import_fbx(self, path: str, **kwargs) -> None
    def import_glb(self, path: str, **kwargs) -> None
    def open_blend(self, path: str) -> None
    def save_blend(self, path: str) -> None
    def blender_version(self) -> tuple[int, int, int]
    # queries
    def armatures(self) -> list[str]
    def meshes(self) -> list[str]
    def skinned_meshes(self, armature: str) -> list[str]
    def bones(self, armature: str) -> list[BoneInfo]          # name, parent, head, tail, matrix_local
    def actions(self) -> list[str]
    def mesh_counts(self, obj: str) -> tuple[int, int, int]   # verts, faces, tris
    def has_shapekeys(self, obj: str) -> bool
    def uv_layers(self, obj: str) -> list[str]
    def uv_loops(self, obj: str, layer: str) -> list[tuple[float, float]]   # per-loop UVs
    def face_loop_ranges(self, obj: str) -> list[tuple[int, int]]
    def material_slots(self, obj: str) -> list[str]
    def image_sizes(self, obj: str) -> list[tuple[int, int]]  # textures referenced by materials
    def bbox_diagonal(self, objs: list[str]) -> float
    # mutations
    def merge_material_slots(self, objs: list[str], name: str) -> None
    def keep_first_uv_layer(self, obj: str) -> None
    def smart_unwrap(self, objs: list[str], angle_deg: float, island_margin: float, pack_margin: float) -> None
    def clear_shapekeys(self, obj: str) -> None
    def add_subsurf(self, obj: str, name: str, levels: int, kind: str, uv_smooth: str, limit_surface: bool) -> None
    def move_modifier_to_top(self, obj: str, name: str) -> None
    def apply_modifier(self, obj: str, name: str) -> None
    def shade_smooth(self, obj: str) -> None
    def export_glb(self, objs: list[str], path: str, **kwargs) -> None
    def export_fbx(self, objs: list[str], path: str, **kwargs) -> None
    # baking (08 §bake) and material binding (12) add more methods; see those docs
```

`BpyScene` implements this with `bpy`/`bmesh`; `tests/fakes/fake_scene.py` implements it over a tiny
in-memory scene graph (objects with counts, uv arrays, modifiers list, shapekey flags, materials). All
job logic is tested against the fake; `BpyScene` is tested by the Blender integration suite.

### 2.4 UV heuristics (`blender_jobs/uv_heuristics.py`, pure)

```python
@dataclass(frozen=True)
class UvAnalysis:
    has_uv: bool
    overlap_ratio: float          # fraction of total UV face area that is covered ≥2×
    island_count: int
    source_texture_max_px: int | None
    uv_bbox_coverage: float       # area of UV bbox hull relative to 1.0

def analyse(uv_loops_per_face: list[list[tuple[float,float]]], texture_sizes: list[tuple[int,int]]) -> UvAnalysis
def decide(a: UvAnalysis, cfg: UvCfg) -> tuple[Literal["kept","unwrapped","created"], str]:
    if cfg.mode == "keep":   return "kept", "mode=keep"           # even without UVs → created? no: E_BJ_NO_UV_KEEP
    if cfg.mode == "unwrap": return ("unwrapped" if a.has_uv else "created"), "mode=unwrap"
    # auto
    if not a.has_uv:                                   return "created",   "no uv layer"
    if a.source_texture_max_px is not None and a.source_texture_max_px <= cfg.palette_texture_max_px:
                                                       return "unwrapped", f"palette texture {px}px"
    if a.overlap_ratio > cfg.overlap_ratio_threshold:  return "unwrapped", f"overlap {ratio:.2f}"
    return "kept", "uvs usable"
```

Overlap ratio: rasterize UV triangles into a 512×512 coverage grid and compute
`cells_covered_≥2 / cells_covered_≥1`. Blender bundles numpy, so `uv_heuristics.py` MAY import numpy
(it is the one third-party module allowed Blender-side besides `bpy`/`bmesh`/`mathutils`); it is also
available in the orchestrator venv, so the module is unit-testable unchanged. Deterministic and unit-testable with synthetic UV sets
(fully overlapping quads → 1.0; a perfect atlas → 0.0).

Smart unwrap: edit mode, select all faces,
`bpy.ops.uv.smart_project(angle_limit=radians(angle), island_margin=island_margin, area_weight=0.0,
correct_aspect=True, scale_to_bounds=False)`, then `bpy.ops.uv.pack_islands(rotate=True,
margin=pack_margin)`, object mode. Multiple mesh objects are unwrapped **together** (multi-object edit
mode) so they share one UV space.

### 2.5 Static GLB export settings

```python
GLB_EXPORT_KWARGS = dict(   # NORMATIVE; glTF 2.0 exporter
    export_format="GLB", use_selection=True,
    export_apply=True,                 # apply modifiers (none remain; safe)
    export_texcoords=True, export_normals=True, export_tangents=False,
    export_materials="PLACEHOLDER",    # no textures; single placeholder material keeps primitive grouping
    export_colors=False, export_attributes=False,
    export_skins=False, export_animations=False, export_morph=False,   # UNRIGGED
    export_yup=True, export_image_format="NONE",
)
```

The exporter triangulates; the static GLB therefore has `triangle_count` equal to our mesh's
triangulated count. Blender-side we compute `static_glb_triangles` from the mesh (`sum(len(poly)-2)`)
and `unique_positions` from quantized vertex coordinates — the orchestrator's `verify` re-derives both
from the GLB and MUST agree (`E_VERIFY_SELF_MISMATCH` otherwise — guards against exporter surprises).

### 2.6 FBX import settings (`blender_jobs/fbx_settings.py`)

```python
FBX_IMPORT_KWARGS = dict(   # NORMATIVE
    use_anim=True, anim_offset=1.0,
    ignore_leaf_bones=False,            # vendor files should not have them; do not guess
    force_connect_children=False,
    automatic_bone_orientation=False,   # preserve vendor bone axes exactly
    primary_bone_axis="Y", secondary_bone_axis="X",
    use_custom_normals=True, use_image_search=True,
    use_prepost_rot=True, global_scale=1.0, bake_space_transform=False,
    axis_forward="-Z", axis_up="Y",
)
```

These are the only import kwargs in the codebase; `roundtrip_animation` uses the same dict, so
companion clips and the model share one interpretation of the vendor skeleton.

### 2.7 Rig fingerprint (`blender_jobs/rig_fingerprint.py`)

- Bones in depth-first order from root bones sorted by name, children sorted by name.
- `transform_path` = `"/".join([armature_object_name, *ancestors, bone])` — mirrors Unity's path for a
  model whose root GameObject is the FBX root. (Unity strips the FBX root node name; the armature object
  becomes a child; verified in Spike S2.)
- `fingerprint_sha256` = sha256 over `name|parent|` + `matrix_local` elements rounded to 1e-5.
- Actions: `bpy.data.actions` names (Blender creates one per FBX take, named after the take).

### 2.8 Errors (`ok=False` results)

| Code | Meaning | Orchestrator reaction |
|---|---|---|
| `E_BJ_IMPORT_FAILED` | FBX importer raised | fail stage; include importer message |
| `E_BJ_NO_ARMATURE` / `E_BJ_MULTI_ARMATURE` | rig assumptions violated | fail; suggest `source.primary` |
| `E_BJ_NO_SKINNED_MESH` | nothing bound to the armature | fail |
| `E_BJ_SHAPEKEYS` | policy `fail` and shape keys exist | fail; message lists meshes and the `geometry.shapekeys: drop` fix |
| `E_BJ_NO_UV_KEEP` | `uv.mode=keep` but no UV layer | fail |
| `E_BJ_UNWRAP_FAILED` | smart project produced zero UV area | fail |
| `E_BJ_FACE_LIMIT` | projected triangle count after subdivision > `geometry.max_triangles` (checked **before** applying: `tris × 4^levels`) | fail; message suggests lower `subdivision_levels` |
| `E_BJ_RIG_MUTATED` | fingerprint changed during prepare (bug guard) | fail hard |
| `E_BJ_EXPORT_FAILED` | GLB export raised | fail |
| `E_BJ_UNKNOWN_JOB` / `E_BJ_BAD_PARAMS` / `E_BJ_PROTOCOL` / `E_BJ_UNHANDLED` | `entry.py` envelope errors: unknown job name; job.json fails dataclass validation; `protocol_version` mismatch; uncaught exception (traceback attached, exit 2) | fail; `E_PROTOCOL`-class handling in the runner |

### 2.9 Extension seam: `GeometryStrategy`

```python
class GeometryStrategy(Protocol):          # blender_jobs/jobs/prepare_model.py
    def apply(self, scene: SceneApi, meshes: list[str], params: PrepareModelParams) -> list[str]  # warnings
class SubdivideStrategy: ...               # v1
# future: RemeshTransferStrategy — import external high-poly, Data Transfer weights, rebind
```

Selection by `params.geometry.strategy` (v1 only accepts `"subdivide"`; the field exists with that
default so adding `remesh_transfer` later is config-only).

### 2.10 Tests (summary; details in `15_…`)

- `uv_heuristics`: property tests (overlap in [0,1], perfect atlas → 0, full overlap → 1), decision
  table tests for every branch of `decide`.
- `prepare_model` with `FakeScene`: single mesh happy path; multi-mesh; palette texture → unwrap; shape
  keys × 3 policies; levels 0 (no modifier); `E_BJ_*` paths; rig-mutation guard.
- Orchestrator `PrepareStage`: fingerprint stability; protocol drift (bad `data`) → error; sha
  cross-check failure path.
- Integration (`@blender`): run on `rigged_cube.fbx` and `palette_cube.fbx`; assert counts, GLB exists,
  blend reopens, fingerprint bone list equals fixture's known bones.
