# Grounded Pipeline Importer (`com.studio.grounded-pipeline`)

Editor-only Unity package that turns a `grounded_pipeline` output folder into a ready-to-use asset:
Generic rig with a copied (or created) Avatar, URP/Lit material built from the manifest, correct
texture import settings, and companion `<id>@<clip>.fbx` clips sharing the model's Avatar.

Requires Unity 6000.x with the Universal Render Pipeline (`com.unity.render-pipelines.universal`
17.x). Nothing in this package ships to players.

## Install (from disk)

1. Copy or check out this repository on the Unity PC (or point at a synced folder).
2. In Unity: **Window > Package Manager > + > Add package from disk...** and choose
   `unity/com.studio.grounded-pipeline/package.json`.
3. The Console should stay silent. If it reports a compile error inside `UrpMaterialBuilder.cs`
   around `BaseShaderGUI.SetMaterialKeywords`, delete the `GP_URP` entry from
   `Editor/GroundedPipeline.Editor.asmdef` (`versionDefines`): the explicit keywords the builder
   enables are sufficient on their own.

## Using it

Copy a pipeline output folder as-is into the project, for example
`output/ant_worker/` to `Assets/Art/Creatures/ant_worker/`:

```
ant_worker/
├── ant_worker.fbx
├── manifest.json
├── textures/ant_worker_basecolor.png, ant_worker_normal.png, ant_worker_metallic_smoothness.png, ...
└── animations/ant_worker@idle.fbx, ant_worker@walk.fbx
```

`gp unity sync` does the same copy into `[library].unity_import_dir` while keeping existing
`.meta` files, so GUIDs (and therefore scene and prefab references) survive re-runs.

### What the postprocessor does

`GroundedPipelinePostprocessor` (`AssetPostprocessor`, order 100) acts only on assets referenced
by a `manifest.json` in the same folder or the parent folder (`textures/`, `animations/`):

| Callback | Effect |
|---|---|
| `OnPreprocessTexture` | `kind == "normal"` -> Normal map type; otherwise `sRGBTexture = srgb`; mipmaps on; `alphaIsTransparency` off; compressed; `_MetallicGlossMap` keeps its alpha (`alphaSource = FromInput`). |
| `OnPreprocessModel` | Generic rig; `materialImportMode = None`; normals imported; blend shapes off; `bakeAxisConversion = false`; `useFileScale = true`. Registers the manifest as a source dependency and the textures as artifact dependencies. Avatar: `copy_from_source` copies from `rig.source_avatar_asset_path` when that asset has an Avatar, otherwise falls back to *Create From This Model* with a warning. Companion `<id>@<clip>.fbx` files copy the model FBX's Avatar. |
| `OnPostprocessModel` | Builds `Materials/<name>.mat` next to the FBX (updated in place, GUID kept) from `materials[]`: textures by slot, floats, colours, keywords; `_EMISSION` also sets `globalIlluminationFlags = RealtimeEmissive`. Assigns it to every renderer (or to the names listed in `renderers`). Logs `[GroundedPipeline] <id>.fbx configured (Generic, avatar: copy|create, materials: N)`. |
| `OnPostprocessAllAssets` | A changed `manifest.json` force-reimports the FBX files it references. |

All errors are reported as `[GroundedPipeline] <asset>: <message>` and never block unrelated
imports.

### Menu: Tools > Grounded Pipeline

- **Reimport All Pipeline Models** - force-reimport every FBX referenced by a manifest under `Assets/`.
- **Validate Manifests** - load every manifest, report schema problems and missing referenced files.
- **Open Log** - reveal the Editor log file.
- **Dump Clip Paths** - see Spike S2 below.

## Manual verification checklist (plan 12 §4.4)

1. Drop `output/<id>` into `Assets/Art/Creatures/`. The Console shows no `[GroundedPipeline]`
   errors (one `configured (...)` info line per model).
2. Select the FBX: the Rig tab shows **Generic** and the Avatar is *Copy From Other Avatar* (or
   *Create From This Model* when no source was available).
3. `Materials/<id>_mat.mat` exists next to the FBX with shader **Universal Render Pipeline/Lit**,
   all maps assigned, and the normal map slot populated with a texture imported as a normal map.
4. Drag the model into a scene; create an Animator Controller from the `@` clips (or use the
   vendor controller); idle/walk play without stretching or detached bones; 1 unit = 1 m compared
   with the vendor import side by side.
5. Textures: normal map import type = **Normal map**; metallic/occlusion maps have **sRGB (Color
   Texture)** off; basecolor/emission have it on.

Take screenshots of 2-5 and attach them to the WP-12 PR.

## Spike S2: Dump Clip Paths

Spike S2 must confirm the transform paths Unity binds clips to (expected form
`Armature/Root/Spine`, matching the rig fingerprint's `transform_path`):

1. Import a model and one companion clip as above.
2. In the Project window select the companion FBX (or expand it and select an `AnimationClip`).
3. **Tools > Grounded Pipeline > Dump Clip Paths**. The Console prints, per clip, the distinct
   `AnimationUtility.GetCurveBindings(clip)[i].path` values.
4. Compare with `work/prepare_report.json` -> `rig.bones[].transform_path`. Any difference (for
   example a missing `Armature/` prefix) goes into `docs/spikes/S2-fbx-roundtrip-findings.md` and
   decisions R-04..R-06.

## Tests

`Tests/Editor/ManifestLoadTests.cs` (EditMode, Unity Test Framework) checks that `JsonUtility`
round-trips the golden manifest (`tests/fixtures/manifest_golden.json`, embedded verbatim) and
that `ManifestLocator` resolves the model, `textures/` and `animations/` paths. Run them through
**Window > General > Test Runner > EditMode**. The Python suite (`tests/unit/unity`) keeps the C#
DTO field names in parity with the pydantic models and the embedded golden JSON in sync.
