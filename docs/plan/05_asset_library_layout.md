# 05 — Asset Library Layout (on-disk contract)

The asset library is the single input/output surface of the pipeline. Its layout is a **contract**:
the CLI, the stages, the Unity postprocessor and humans all rely on it. `gp init` scaffolds it;
`gp new-model` scaffolds a model folder; `gp validate` enforces every rule below.

## 1. Tree

```
library/
├── library.toml                 # library-wide configuration (defaults for every model)
├── .pipeline/                   # library-level state
│   ├── run.lock                 # prevents concurrent `gp run` on this library
│   └── runs/<run_id>/           # run.log (JSONL), report.json, console.txt
├── style/                       # STYLE SOURCE — shared across all models
│   ├── style.md                 # base prompt(s) + style settings (front matter)
│   └── references/              # style images: the look we want everything to share
│       ├── primary.png          # REQUIRED if any model uses prompt_mode text+style
│       └── *.png|jpg
├── models/
│   └── <model_id>/              # ONE FOLDER = ONE MODEL
│       ├── model.md             # REQUIRED. settings (front matter) + prompt (body)
│       ├── source/              # REQUIRED. vendor files, never modified by the pipeline
│       │   ├── <primary>.fbx    # the rigged (and usually animated) model
│       │   ├── animations/      # OPTIONAL. companion clips sharing the skeleton
│       │   │   └── *.fbx
│       │   ├── textures/        # OPTIONAL. vendor textures (reference / palette detection)
│       │   └── LICENSE*         # OPTIONAL but recommended: the pack's licence text
│       ├── references/          # OPTIONAL. per-model guidance images for texturing
│       │   ├── front.png left.png back.png right.png   # multi-view guidance (if provider accepts)
│       │   ├── style.png        # overrides library/style/references/primary.png for this model
│       │   └── *.png|jpg        # any extra, listed explicitly in model.md to be used
│       ├── work/                # pipeline intermediates (disposable, gitignored)
│       ├── output/              # deliverables (see 12_…)
│       └── .pipeline/           # per-model state + logs
│           ├── state.json
│           ├── state.lock
│           └── logs/<stage>-<run_id>.log
└── generative/                  # RESERVED for the future pipeline (17_…). Empty in v1.
```

## 2. Naming rules

- `model_id` MUST match `^[a-z0-9][a-z0-9_]{1,63}$`. It is used as FBX name, texture prefix, material
  name, Unity folder name and log key. `gp validate` fails on violations.
- Exactly one primary source FBX unless `source.primary` names it explicitly. If `source/` contains
  several `.fbx` at top level and `source.primary` is unset → validation error `E_SOURCE_AMBIGUOUS`.
- Companion animations: every `source/animations/*.fbx` is used unless `source.animations` lists a
  subset (globs allowed). Each MUST contain an armature whose bone names are a superset-compatible match
  of the primary (checked at `export`, see `12_…` §4).
- Reference images MUST be PNG or JPEG, ≤ 20 MB (Tripo upload limit), minimum 256×256 px.
- Output file names are derived from `model_id`:
  `<model_id>.fbx`, `textures/<model_id>_basecolor.png`, `textures/<model_id>_normal.png`,
  `textures/<model_id>_metallic_smoothness.png`, `textures/<model_id>_occlusion.png`,
  `textures/<model_id>_emission.png`, `animations/<model_id>@<clip>.fbx`, `manifest.json`.
  The `@clip` convention lets Unity's importer show clips grouped under the model.

## 3. `library.toml` — full reference with defaults

```toml
schema_version = 1

[library]
name = "tinyworld-creatures"
unity_import_dir = ""              # e.g. "D:/Game/Assets/Art/Creatures" ; empty = no sync
concurrency = 2                    # models processed in parallel
blender_parallelism = 1            # simultaneous Blender processes

[blender]
executable = ""                    # empty = auto-detect (GP_BLENDER_EXE env wins)
allow_unsupported = false          # permit Blender versions outside 4.2/4.5 LTS
job_timeout_s = 1800
keep_blend_files = true            # keep work/prepared.blend after success

[geometry]
strategy = "subdivide"             # v1: only "subdivide" (seam for a future "remesh_transfer")
subdivision_levels = 1             # 0..3 ; 0 = geometry untouched (texture-only run)
subdivision_type = "catmull"       # "catmull" | "simple"
shade_smooth = true
shapekeys = "fail"                 # "fail" | "drop" | "skip" (skip = leave that mesh unsubdivided)
merge_material_slots = true        # single material slot → single texture set
keep_only_one_uv_layer = true
include_unskinned = true           # export meshes not bound to the armature as-is (never subdivided)
max_triangles = 500000             # abort prepare if the subdivided result would exceed this

[uv]
mode = "auto"                      # "auto" | "keep" | "unwrap"
palette_texture_max_px = 128       # source texture ≤ this → treated as palette → unwrap
overlap_ratio_threshold = 0.20     # UV overlap above this → unwrap
smart_project_angle_deg = 66.0
island_margin = 0.02
pack_margin = 0.01

[texture]
provider = "tripo"
model_version = "v3.5-20260815"
quality = "standard"               # "fast" | "standard" | "detailed" | "extreme"
pbr = true
bake = true                        # Tripo-side bake flag (material effects → base textures)
delight = true
alignment = "original_image"       # "original_image" | "geometry" (only matters with image guidance)
prompt_mode = "text+style"         # "text" | "text+style" | "image" | "images"
seed = 1337                        # texture_seed; per-model override allowed
style_image = "style/references/primary.png"   # relative to library root

[verify]
route = "auto"                     # "auto" | "extract" | "bake"
position_quantum = 1e-4            # model units
uv_quantum = 1e-4
allow_geometry_change = false      # if Tripo remeshed: false = fail, true = bake anyway

[bake]
resolution = 2048
samples = 16
margin_px = 16
cage_extrusion_ratio = 0.01        # × bounding-box diagonal
max_ray_distance_ratio = 0.05
maps = ["basecolor", "normal", "metallic", "roughness", "occlusion", "emission"]

[export]
fbx_scale_option = "FBX_SCALE_ALL"
roundtrip_companion_animations = true
embed_textures = false

[unity]
avatar_mode = "copy_from_source"   # "copy_from_source" | "create_from_model"
source_asset_root = ""             # Unity path of the vendor import, e.g. "Assets/ThirdParty/Quaternius"
shader = "Universal Render Pipeline/Lit"

[tripo]
base_url = "https://openapi.tripo3d.ai/v3"
poll_interval_s = 2.0
poll_timeout_s = 900
max_concurrent_tasks = 4
min_request_interval_s = 0.25
max_credits_per_run = 0            # 0 = unlimited
use_presign_above_mb = 60

[tripo.credit_table]               # estimates only (official table 2026-11-02); real credits_consumed is recorded per task
fast = 10
standard = 10
detailed = 20
extreme = 30
style_image = 0

[logging]
level = "INFO"
json_console = false
```

Every `[section].key` here can be overridden per model in `model.md` front matter under the same
section name (see §5), and by environment variables `GP_<SECTION>__<KEY>` (double underscore), and by
CLI flags. Precedence is defined in `07_config_and_secrets.md`.

## 4. `style/style.md`

```markdown
---
schema_version: 1
name: tinyworld
primary_reference: references/primary.png
negative_terms: ["outline", "cel shading", "pixel art", "photorealistic"]   # kept for future providers
---

## base_prompt
Stylized hand-painted game creature, smooth soft shading, saturated vibrant colours,
subtle painted gradients, clean readable shapes, no outlines, PBR, slightly glossy chitin,
backyard-scale insect realism with friendly proportions.

## material_notes
Chitin: glossy with soft specular. Fur: matte, painted strands. Eyes: wet, dark, high gloss.

## palette
#2E7D32 leaf green, #F9A825 pollen yellow, #6D4C41 soil brown, #1565C0 sky blue accents
```

Rules: `## base_prompt` is REQUIRED. Other sections are optional and are appended to the composed
prompt only if the model's `prompt.include_sections` lists them. Headings are matched
case-insensitively; section bodies are stripped of leading/trailing whitespace.

## 5. `models/<model_id>/model.md`

```markdown
---
schema_version: 1
display_name: Worker Ant
source:
  primary: source/Ant.fbx
  animations: ["source/animations/*.fbx"]
  license: "CC0 — Quaternius"
geometry:
  subdivision_levels: 1
  shapekeys: drop
uv:
  mode: auto
texture:
  seed: 4242
  quality: detailed
  prompt_mode: text+style
  guidance_images:                # only used when prompt_mode is image/images
    front: references/front.png
    left:  references/left.png
    back:  references/back.png
    right: references/right.png
prompt:
  include_sections: [base_prompt, material_notes]
  template: "{base_prompt}\n\n{model_prompt}\n\n{material_notes}"
unity:
  source_asset_path: "Assets/ThirdParty/Quaternius/Ant.fbx"   # for avatar copy
tags: [insect, enemy, tier1]
---

## prompt
A worker ant: matte charcoal-black head and thorax, warm red-brown abdomen with soft painted
segment highlights, pale amber mandibles, large glossy dark eyes, thin articulated legs with
subtle lighter joints.

## notes
Vendor pack uses a 16x4 palette texture; expect auto-unwrap. Eyes have a blink shape key — dropped.
```

Rules:

- Front matter is YAML (parsed by `python-frontmatter`); the body is Markdown split on `## ` headings.
- `## prompt` is REQUIRED when `texture.prompt_mode` includes text. Composed prompt length MUST be
  ≤ 1 000 characters after templating (`E_PROMPT_TOO_LONG`); the builder warns above 600.
- Unknown front-matter keys → `E_UNKNOWN_KEY` (pydantic `extra="forbid"`). Typos must not pass silently.
- All paths are relative to the model folder except `texture.style_image`, which is relative to the
  library root (or absolute).
- `prompt.template` placeholders: any style section name, `{model_prompt}` (the `## prompt` body),
  `{display_name}`, `{tags}`. Missing placeholders are an error at validation time.

## 6. `work/`, `output/`, `.pipeline/` contents

| Path | Producer | Consumer |
|---|---|---|
| `work/prepared.blend` | prepare | apply (bake), export |
| `work/static.glb` | prepare | texture, verify |
| `work/prepare_report.json` | prepare | verify, export (source rig fingerprint), handoff (provenance) |
| `work/tripo/request.json` | texture | debugging, cost audit |
| `work/tripo/task.json` | texture | resume polling, handoff (provenance, credits) |
| `work/tripo/result.glb` | texture | verify, apply |
| `work/tripo/preview.png` | texture | humans (Tripo's render, if provided) |
| `work/verify_report.json` | verify | apply, handoff |
| `work/textures/raw/*.png` | apply | apply (pack step) |
| `work/textures/urp/*.png` | apply | export, handoff |
| `work/apply_report.json` | apply | handoff (provenance: route, factors) |
| `work/export/<id>.fbx`, `animations/*.fbx` | export | handoff |
| `work/export/rig_fingerprint_out.json`, `rig_fidelity_report.json` | export | handoff |
| `output/<id>/…` | handoff | Unity, humans |
| `.pipeline/state.json` | every stage (via StateStore) | runner, `gp status` |

`gp clean --work` deletes `work/`; `gp clean --output` deletes `output/`; `gp clean --state` deletes
`.pipeline/state.json` (forcing a full re-run). Nothing under `source/`, `references/`, or `model.md`
is ever written by the pipeline.

## 7. Reserved: `library/generative/` (future)

```
generative/
├── concepts/<batch_id>/{batch.md, prompts/*.md, images/*.png, review.json}
├── items/<item_id>/{item.md, concept.png, multiview/{front,left,back,right}.png,
│                     mesh/*.glb, rig/*.glb, work/, output/, .pipeline/}
```

Fixed now so that v1 code never squats on these names. Specified in `17_future_generative_pipeline.md`.

## 8. Validation rules (`gp validate`) — error codes

| Code | Condition |
|---|---|
| `E_MODEL_ID` | folder name violates the regex |
| `E_MODEL_MD_MISSING` | no `model.md` |
| `E_FRONTMATTER` | YAML parse error or pydantic validation error (message includes path + key) |
| `E_UNKNOWN_KEY` | unknown front-matter key |
| `E_SOURCE_MISSING` | `source.primary` not found or no `.fbx` in `source/` |
| `E_SOURCE_AMBIGUOUS` | multiple top-level `.fbx` and no `source.primary` |
| `E_ANIM_GLOB_EMPTY` | `source.animations` pattern matches nothing (warning if list is default) |
| `E_PROMPT_MISSING` | text mode without `## prompt` |
| `E_PROMPT_TOO_LONG` | composed prompt > 1 000 chars |
| `E_TEMPLATE_PLACEHOLDER` | unknown `{placeholder}` in template |
| `E_STYLE_IMAGE_MISSING` | `text+style` mode and style image not found |
| `E_STYLE_BASE_PROMPT_MISSING` | `style/style.md` has no non-empty `## base_prompt` section |
| `E_GUIDANCE_IMAGES` | `images` mode without `front` or fewer than two views; or file missing / too large |
| `E_REF_IMAGE_FORMAT` | reference image not PNG/JPEG or < 256 px |
| `W_LICENSE_MISSING` | no `source.license` and no `source/LICENSE*` file (warning) |
| `E_LIBRARY_TOML` | library.toml invalid |
| `E_UNITY_DIR` | `unity_import_dir` set but not an existing directory |
| `E_PATH_AMBIGUOUS` | a relative path resolves both under the model folder and the library root |
| `W_GUIDANCE_UNVERIFIED` | `image`/`images` prompt mode used before Spike S1 confirmed support (warning) |
