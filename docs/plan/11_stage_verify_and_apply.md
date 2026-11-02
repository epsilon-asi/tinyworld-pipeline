# 11 — Stages `verify` and `apply`

## 1. GLB loading (`gltf/loader.py`)

```python
@dataclass(frozen=True)
class GlbMesh:
    positions: np.ndarray        # (N,3) float32, world-space (node transforms applied)
    uvs: np.ndarray | None       # (N,2) float32, TEXCOORD_0
    indices: np.ndarray          # (M,3) uint32 triangles
    material_index: int | None
@dataclass(frozen=True)
class GlbDocument:
    meshes: list[GlbMesh]        # one per primitive, all nodes traversed
    materials: list[GlbMaterial] # pbrMetallicRoughness + normal/occlusion/emissive texture refs + factors
    images: list[GlbImage]       # bytes + mime + name
def load_glb(path: Path) -> GlbDocument   # NORMATIVE
```

- Uses `pygltflib.GLTF2.load_binary`; accessors decoded with numpy by componentType/type; sparse
  accessors and Draco → `UnsupportedGlbError` (Tripo output is plain; `compress` is never requested).
- Node hierarchy traversed with accumulated 4×4 transforms so positions are comparable regardless of
  how the exporter nested nodes. Y-up assumed on both sides (both our exporter and Tripo emit Y-up glTF).
- Non-triangle primitive modes → error (we never produce them; Tripo does not either).

## 2. Surface fingerprint (`gltf/fingerprint.py`, pure)

```python
def surface_fingerprint(doc: GlbDocument, *, q_pos: float, q_uv: float) -> SurfaceFingerprint  # NORMATIVE
```

Algorithm:

1. Concatenate all primitives' positions; `P = round(positions / q_pos).astype(int64)`.
2. `unique_position_count = len(np.unique(P, axis=0))`.
3. `position_set_sha256 = sha256(np.unique(P, axis=0).tobytes())` (sorted lexicographically by `np.unique`).
4. If every primitive has UVs: rows `R = concat(P, round(uv / q_uv))`, deduplicate, sort, hash →
   `surface_sha256`; else `has_uv=False`, `surface_sha256 = sha256(b"")`.
5. `triangle_count = sum(len(indices))`.

Why order-independent: Tripo may merge/split primitives and reorder vertices; glTF exporters split
vertices on UV seams differently. Sets of quantized rows are invariant to all of that. Why quantize at
1e-4: float32 round-trips through two exporters typically differ by ≤ 1e-6 in unit-scale models; 1e-4
absorbs that while still distinguishing any real re-unwrap (UV changes are ≫ 1e-4).

Edge: Tripo **does** normalise uploaded meshes (documented: task output carries `normalize_scale` and,
when available, `normalize_translation`; `normalized = original × normalize_scale`). `verify` therefore
always maps `theirs` back into our frame before fingerprinting: first using the documented values from
`work/tripo/task.json` (`theirs = (theirs − normalize_translation) / normalize_scale`), and if those are
absent, falling back to a bounding-box fit (pure uniform scale + translation, accepted when the fitted
boxes agree within 1e-3 relative). The transform used is recorded in `VerifyReport.reason`
(`"; normalised scale=… source=task|bbox"`). Bake route still works either way because the bake job also
aligns bounding boxes (see §5.2).

## 3. Stage `verify` (`pipeline/stages/verify.py`)

```
ours   = surface_fingerprint(load_glb(work/static.glb))
theirs = surface_fingerprint(load_glb(work/tripo/result.glb))
self-check: ours.triangle_count == prepare_report.static_glb_triangles and
            ours.unique_position_count == prepare_report.static_glb_unique_positions   → else E_VERIFY_SELF_MISMATCH
report = decide_route(ours, theirs, cfg.verify)              # 06 §6
write work/verify_report.json
```

Fingerprint of the stage: shas of both GLBs + `cfg.verify`. Outputs: `work/verify_report.json`.

## 4. Stage `apply` (`pipeline/stages/apply.py`)

```
route = verify_report.route
if route == "extract": raw = extract_textures(load_glb(result.glb), work/textures/raw)     # §5.1
else:                  raw = run blender job bake_textures → work/textures/raw             # §5.2
urp = pack_urp(raw, work/textures/urp, model_id)                                           # §6
write work/apply_report.json {route, raw_maps, urp_maps, factors}
```

Fingerprint: `route`, sha(result.glb), and for the bake route only: `cfg.bake`, sha(prepared.blend),
`blender_jobs_code_sha()`. Outputs: `work/textures/urp/*.png`, `work/apply_report.json`.

### 5.1 Extract (`gltf/extract.py`)

- Find materials actually referenced by primitives. If more than one **distinct** texture set is
  referenced, `extract_textures` raises `MultiMaterialResultError`; the `apply` stage catches it and
  falls back to the bake route automatically (log `W_EXTRACT_FALLBACK_BAKE`, `apply_report.
  fallback_to_bake = true`) rather than failing, because the bake collapses everything onto our single
  material. There is no config key for multi-material extraction in v1.
- Images are written as-is (PNG or JPEG bytes → re-encoded to PNG via Pillow if JPEG, to avoid lossy
  re-compression later) into `raw/`: `basecolor.png`, `metallic_roughness.png`, `normal.png`,
  `occlusion.png`, `emissive.png` — whichever exist.
- Factors captured into `RawTextureSet.factors`: `baseColorFactor` (RGBA), `metallicFactor`,
  `roughnessFactor`, `emissiveFactor` (RGB), `normalTexture.scale`, `occlusionTexture.strength`,
  `KHR_materials_emissive_strength` if present.
- `KHR_texture_transform` present → `UnsupportedGlbError` (never seen from Tripo; fail loudly).

```python
@dataclass(frozen=True)
class RawTextureSet:
    basecolor: Path | None; metallic_roughness: Path | None; normal: Path | None
    occlusion: Path | None; emissive: Path | None
    factors: TextureFactors
    source: Literal["extract", "bake"]
```

### 5.2 Bake (Blender job `bake_textures`)

Inputs: `prepared.blend`, `result.glb`, `BakeSection`, `bbox_diagonal`. Steps (normative):

```
1  open_blend(prepared.blend); target_meshes = skinned meshes (our mesh)
2  import_glb(result.glb, merge_vertices=False, import_shading="NORMALS") → source objects
3  align: compute bbox of source vs target; if uniform-scale/translation mismatch > 1e-3 relative,
   transform source objects to match target bbox (never touch target)
4  hide armature from render/bake; ensure target has exactly one material (prepared guarantees it)
5  scene.render.engine = "CYCLES"; device from env GP_BLENDER_CYCLES_DEVICE (default CPU)
   view_settings.view_transform = "Standard"; display_device "sRGB"; look "None"
   bake settings: use_selected_to_active=True, cage_extrusion = ratio*bbox_diag,
   max_ray_distance = ratio*bbox_diag, margin = margin_px, margin_type="EXTEND", use_clear=True
6  for each map in cfg.maps (skip if the source material lacks the input):
     create image "<map>" res×res (float=False; normal: 16-bit PNG optional), colorspace:
       basecolor/emission → sRGB ; others → Non-Color
     add Image Texture node on target material, set active
     configure source material + bake type:
       basecolor : type EMIT  with Base Color rewired to an Emission shader (ADR-0004: DIFFUSE/COLOR
                   is scaled by (1 − metallic) and loses ~4 % dielectric energy in Blender 4.x; EMIT is exact)
       normal    : type NORMAL, normal_space TANGENT        # captures source normal-map detail
       roughness : type ROUGHNESS
       metallic  : type EMIT  with source Principled.Metallic input rewired to an Emission shader
       occlusion : type EMIT  with occlusion texture (glTF importer puts it in a "glTF Material Output"
                   group) rewired to Emission; if absent → skip map
       emission  : type EMIT  native, no rewiring (keeps the importer's Emission Strength from
                   KHR_materials_emissive_strength; values > 1 clip in 8-bit PNG)
     select sources, make target active, bpy.ops.object.bake(type=..., use_selected_to_active=True, ...)
     image.filepath_raw = out_dir/<map>.png; file_format PNG; save
     restore source node tree (undo rewiring) before next map
7  write BakeTexturesData (maps, seconds); do NOT save prepared.blend (bake nodes are throwaway)
```

Samples: `cfg.bake.samples` for `basecolor`/`normal`/`roughness`; `EMIT` bakes use 1 sample (no noise
possible). Denoising off. Output of `metallic` + `roughness` is merged into `metallic_roughness.png`
by the orchestrator (`roughness → G`, `metallic → B`) so both routes yield the same `RawTextureSet`
shape before packing.

Errors: `E_BJ_BAKE_NO_SOURCE_MATERIAL`, `E_BJ_BAKE_FAILED(map)`, `E_BJ_BAKE_EMPTY(map)` (all-black
result where source had data — detected by variance ≈ 0).

## 6. URP packing (`textures/urp_pack.py`, pure numpy/Pillow)

Inputs: `RawTextureSet`. Output: `UrpTextureSet` (paths) written to `work/textures/urp/`.

| URP file | Channels | Formula (all in 0..1 floats; integer output 8-bit) |
|---|---|---|
| `<id>_basecolor.png` (sRGB) | RGBA | `rgb = base.rgb * baseColorFactor.rgb`; `a = base.a * baseColorFactor.a` (alpha kept; URP opaque ignores it) |
| `<id>_normal.png` (linear, tagged NormalMap) | RGB | copy. glTF normal maps are OpenGL +Y; Unity expects +Y → **no green flip**. `normalTexture.scale ≠ 1` → recorded in manifest `_BumpScale` |
| `<id>_metallic_smoothness.png` (linear) | RGBA | `R = mr.b * metallicFactor`; `G = 0`; `B = 0`; `A = 1 − mr.g * roughnessFactor` |
| `<id>_occlusion.png` (linear) | RGBA | `G = occ.r` (URP samples G); `R = B = occ.r`; `A = 1`. Strength → `_OcclusionStrength` |
| `<id>_emission.png` (sRGB) | RGB | copy; `emissiveFactor` and emissive strength → `_EmissionColor = factor × strength` |

Missing inputs → missing outputs; the manifest only lists textures that exist, and material keywords
are derived from presence (`_NORMALMAP`, `_METALLICSPECGLOSSMAP`, `_OCCLUSIONMAP`, `_EMISSION`).
If `metallic_roughness` is missing, no metallic map is written and the manifest sets `_Metallic = 0`,
`_Smoothness = 0.5`.

Resolution: outputs keep input resolution; if maps differ in size, all are resampled (Lanczos) to the
largest. Hypothesis property tests: packing is idempotent on identity factors, channel ranges preserved,
`A = 1 − G` exactly for 8-bit inputs (`255 − g`).

## 7. Reports

`work/apply_report.json`:

```json
{"route":"extract","raw":{"basecolor":"work/textures/raw/basecolor.png", …},
 "urp":{"basecolor":"work/textures/urp/ant_worker_basecolor.png", …},
 "factors":{"base_color":[1,1,1,1],"metallic":1.0,"roughness":1.0,"emissive":[0,0,0],"normal_scale":1.0,"occlusion_strength":1.0},
 "resolution":[2048,2048],"fallback_to_bake":false}
```

## 8. Tests

- `loader`: synthetic GLBs from `tests/fakes/glb_builder.py` (quad, cube, two-primitive, nested nodes
  with transforms, missing UVs, JPEG + PNG images, sparse accessor → error).
- `fingerprint`: invariance under vertex permutation, primitive split/merge, seam re-splitting;
  sensitivity to a single UV change > quantum; scale-normalisation path.
- `decide_route`: full decision table incl. forced routes and `allow_geometry_change`.
- `extract`: all maps present / subsets / multi-material fallback / `KHR_texture_transform` error.
- `bake_textures` with `FakeScene`: map loop, skip-when-absent, rewiring restored, alignment branch,
  errors. Integration (`@blender`): bake a textured cube onto itself → output mean colour within
  tolerance of input.
- `urp_pack`: formulas per channel (exact integers), resampling branch, missing maps, factors.
