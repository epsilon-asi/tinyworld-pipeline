# 18 — Risks, Open Questions, Decisions Log, Spikes

Severity: **blocking** (must be resolved before the dependent WP merges), **high**, **medium**, **low**.
Each risk names its mitigation already built into the design and the spike or decision that closes it.

| ID | Risk / unknown | Sev | Built-in mitigation | Closed by |
|---|---|---|---|---|
| R-01 | Tripo may re-unwrap or remesh uploaded meshes; docs do not state UV/topology preservation. **Confirmed 2026-11-02:** uploaded meshes are rescaled/translated (`output.normalize_scale`/`normalize_translation`), which `verify` now undoes exactly (11 §2) | blocking for WP-08 scope sizing (not for correctness) | `verify` fingerprints + `bake` fallback route; pipeline is correct either way | S1 (WP-14a). If UVs are preserved, `bake` becomes a rarely used fallback; if not, bake is the main path and its performance matters more |
| R-02 | `image`/`images` guidance may be rejected for uploaded (`file_token`) models (third-party docs say only text is accepted for imported models) | high | `text+style` default; `W_GUIDANCE_UNVERIFIED` warning | S1 |
| R-03 | Tripo result GLB structure (JPEG vs PNG textures, multiple primitives, KHR extensions, emissive strength) | medium | loader supports PNG/JPEG, multi-primitive, factors; fails loudly on unsupported extensions | S1 fixtures |
| R-04 | Blender FBX round-trip may alter bone orientation/scale so vendor clips stored in separate files no longer match | high | `automatic_bone_orientation=False`, `add_leaf_bones=False`, companion clips round-tripped with the same settings, RFC compares fingerprints | S2 (WP-14b) |
| R-05 | Unity transform paths may differ from our `transform_path` (armature object node presence/naming) | medium | RFC compares our own before/after only; path format verified in Unity | S2 |
| R-06 | `FBX_SCALE_ALL` vs `FBX_SCALE_NONE` behaviour in Unity 6 (×100 armature scale, root motion scale) | medium | config option; default decided by spike | S2 |
| R-07 | Blender 5.x API changes (FBX/glTF operator names, modifier API) | medium | version gate; `BpyScene` adapter isolates operator calls; `allow_unsupported` flag | WP-03 probe; re-test when 5.x LTS exists |
| R-08 | Subdivision on non-manifold / n-gon heavy vendor meshes may produce artefacts or explode face counts | medium | `simple` type option; face-count cap check (`E_BJ_FACE_LIMIT` at 500 k tris); per-model `levels` | WP-04 integration on real packs (WP-17) |
| R-09 | Smart UV Project on organic meshes yields many islands → visible seams after AI texturing/baking | medium | unwrap **before** subdivision; margins; bake `EXTEND` margin; option to keep vendor UVs when usable | WP-17 visual review |
| R-10 | Shape keys (blinks) lost with subdivision | low (accepted) | explicit policy; levels=0 preserves them | decision D-03 |
| R-11 | Tripo pricing/credit table changes | low | real `credits_consumed` recorded; estimate table configurable; defaults set from the official table on 2026-11-02 (fast 10 / standard 10 / detailed 20 / extreme 30) | ops |
| R-19 | Windows as a `gp` host is deferred; three touch points (executable discovery, process-tree kill, fake Blender exe launcher) are POSIX-only in v1 | low | `pathlib` + forward-slash JSON paths everywhere; platform branches isolated in `blender/locate.py`, `blender/runner.py`, `tests/fakes/fake_blender_exe.py`; Linux CI keeps the code honest | ADR-0002; follow-up WP when a Windows host is needed |
| R-20 | Tripo output URLs expire ~5 min after retrieval | low | download immediately after `success`; resume re-fetches task detail for a fresh URL (10 §3) | WP-05 tests |
| R-12 | Licensing: Tripo free tier restricts commercial use; free packs' licences vary | high (legal) | `source.license` recorded in manifest provenance; `W_LICENSE_MISSING`; runbook says use paid tier | user decision |
| R-13 | Normal-map convention mismatch (OpenGL vs DirectX) | low | glTF and Unity both use +Y (OpenGL); no flip; verified visually in WP-17 | WP-17 |
| R-14 | Baked `EMIT`-trick maps for metallic/occlusion pick up colour management transforms | low | `view_transform=Standard`, Non-Color images, 1 sample | WP-08 integration test asserts exact values |
| R-15 | Windows path length / Unicode in model names | low | `model_id` regex; outputs ≤ 100 chars deep; long-path aware `pathlib` | WP-02 tests |
| R-16 | Concurrent runs or Unity touching `output/` during sync | low | run lock; atomic folder swap; `.meta` preservation | WP-12 tests |
| R-17 | Tripo task timeouts for large meshes (15 min default) | low | `poll_timeout_s` configurable; resume keeps paying only once | ops |
| R-18 | Ambiguous `create_texture_task` failures (POST timed out after server accepted) → double billing | medium | no blind retry; `TripoAmbiguousCreateError` + `gp tripo task` / batch query guidance | WP-05 |

## Decisions log

| ID | Decision | Rationale | Date |
|---|---|---|---|
| D-01 | Orchestrator on its own Python; Blender via subprocess + JSON job protocol | Avoid ABI coupling; testability; any Blender install works | 2026-11-02 |
| D-02 | Only the **static, unrigged** mesh is ever sent to Tripo | Animations cannot be damaged by the provider | 2026-11-02 |
| D-03 | Shape keys: default policy `fail`, opt-in `drop`/`skip`; preserving them is out of v1 | Explicit over silent loss; low demand for face shapes on creatures | 2026-11-02 |
| D-04 | Unwrap before subdivision when unwrapping is needed | Fewer, larger islands; subdivision interpolates UVs smoothly | 2026-11-02 |
| D-05 | Verification uses order-independent quantized surface fingerprints | Robust to vertex reordering and primitive merges by either exporter | 2026-11-02 |
| D-06 | Unity integration via `manifest.json` + Editor `AssetPostprocessor`; no hand-written `.mat` YAML | Stable across Unity versions; GUID-safe | 2026-11-02 |
| D-07 | `manifest.json` is JsonUtility-compatible (arrays, no nulls) | Zero C# dependencies | 2026-11-02 |
| D-08 | Companion animation FBX files are round-tripped through the same Blender settings | Guarantees identical skeleton interpretation across files | 2026-11-02 |
| D-09 | Coverage gate 80 % line+branch+function; `bpy_scene.py`/`entry.py` excluded from unit coverage, covered by integration | Thin adapters cannot be meaningfully unit-tested without Blender | 2026-11-02 |
| D-10 | `text+style` is the default guidance mode | Works for uploaded meshes regardless of R-02 outcome | 2026-11-02 |
| D-11 | Future generative pipeline fixed at folder/contract level only | Avoid squatting names and rework; keep v1 small | 2026-11-02 |
| D-12 | macOS (Apple Silicon) is the primary and only supported `gp` host in v1; Linux for CI; Windows host deferred. Unity import happens on any machine from the output files | Build and maintain on one platform; outputs are plain files | 2026-11-02 |
| D-13 | Repository is public (`github.com/epsilon-asi/grounded-pipeline`); the asset library tree (`library/`) is gitignored wholesale. Only the synthetic `tests/fixtures/library_minimal` is tracked | Open-source tooling, private assets and licences | 2026-11-02 |
| D-14 | Tripo contract corrections from the official docs: host `openapi.tripo3d.ai`; `style_image` is an object; `images` is a view-key array; statuses add `expired`/`unknown`; URLs expire; credit table 10/10/20/30 | Plan 1.0 was written against third-party summaries | 2026-11-02 |
| D-15 | Per-model override models are explicit hand-written classes (`GeometryOverrides` …) checked against `make_partial` by a parity test | mypy `--strict` compatibility (ADR-0003) | 2026-11-02 |
| D-16 | Base colour is baked with the EMIT trick, emission natively; `bake_map` bakes all target meshes into one image | Blender 4.x `DIFFUSE/COLOR` is scaled by (1 − metallic) and loses ~4 % (measured 13/255 vs 0/255) (ADR-0004) | 2026-11-02 |
| D-17 | Before any FBX export, OBJECT actions are renamed to their prefix-stripped names and the armature is put in rest pose (`reset_pose`) | Blender's exporter/importer double-prefix take names on every round trip and write node transforms from the current pose (appendix B §8) | 2026-11-02 |
| D-18 | Shared value objects that two sibling adapter packages need (`RawTextureSet`, `TextureFactors`) live in `domain/textures.py` | Keeps the layer contract strict without structural-protocol duplication | 2026-11-02 |
| D-19 | `apply` bake-route fingerprint includes the Blender-side code sha, like `prepare` | A change to `bake_textures.py` must re-run cached bakes | 2026-11-02 |

## Spikes

### S1 — Tripo behaviour validation (WP-14a)
**Questions:** (1) Are vertex positions and UVs of an uploaded GLB preserved in the textured output?
(2) Which guidance modes are accepted for `file_token` inputs? (3) Output GLB structure and texture
formats; presence of `KHR_materials_emissive_strength`, `KHR_texture_transform`, Draco. (4) Credits per
quality level with/without style image. (5) Typical task durations for 5 k / 50 k tri meshes.
**Method:** `rigged_cube` static export and one real free model; `tripo_live` test + manual inspection
in Blender; record fixtures.
**Outputs:** findings doc; update R-01..R-03, `appendix_A`, `tripo/budget.py` table, fixture JSONs.

### S2 — Blender FBX round-trip and Unity import (WP-14b)
**Questions:** (1) Do clips from separate vendor FBX files play on the round-tripped model? (2) Does
`FBX_SCALE_ALL` remove the ×100 scale without breaking root motion? (3) Exact Unity transform path
format vs our `transform_path`. (4) Any bone orientation drift after round-trip (compare fingerprints)?
(5) Does `armature_nodetype="NULL"` vs `"ROOT"` change the Unity hierarchy?
**Method:** three free packs; scripted Blender round-trip using the normative kwargs; Unity 6 URP test
project; screenshots.
**Outputs:** findings doc; decisions into R-04..R-06; ADR if kwargs change.

## Changelog

- **1.2 (2026-11-02)** — implementation pass (WP-00…WP-13 built). D-16…D-19, ADR-0004; appendix B §8 Blender
  4.5.12 facts; `.importlinter` layers = cli > pipeline > library > adapters > tripo > config > domain and
  `exclude_type_checking_imports`; `E_BJ_PROTOCOL`, `E_STYLE_BASE_PROMPT_MISSING`, `W_UNKNOWN_ENV`,
  `W_EXTRACT_FALLBACK_BAKE`, `W_HANDOFF_NO_TEXTURES`, `W_PROMPT_LONG` codes added where documented.
- **1.1 (2026-11-02)** — review pass before WP-00. Tripo API facts verified against
  `developers.tripo3d.ai` (D-14; `05`, `06`, `10`, `11`, appendix A updated). macOS-primary host (D-12,
  ADR-0002; `01`, `04`, `07`, `16`). Public repo + ignored library tree (D-13; `04` §6). Explicit override
  classes (D-15, ADR-0003; `06` §2–3). Tooling fixes in `04` §2: coverage flags moved out of `addopts`,
  removed ruff `ANN101/102`, stubs `fake-bpy-module-4.5`. `BLENDER_JOBS_CODE_SHA` is lazy everywhere
  (`09`, `13`). numpy allowed Blender-side (`09` §2.4). Blender 4.5 LTS is primary (`01`, `04` §5).

## Open questions for the product owner (non-blocking; defaults apply until answered)

1. Should `output/` be committed to the repo (reproducible but large) or treated as a build artefact? (Default: ignored.)
2. Target texture resolution per creature tier (e.g. 1 k for ambient bugs, 2 k for enemies)? (Default: 2 k everywhere; per-model override exists.)
3. Is Humanoid rig support needed for any asset (NPC packs)? (Default: no.)
4. Should the pipeline also produce a `.blend` library file for artists? (Default: `prepared.blend` is kept per model.)
