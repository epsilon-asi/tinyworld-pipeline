# 12 — Stages `export` and `handoff`, Unity consumer

## 1. Stage `export` (orchestrator + Blender jobs `export_rigged`, `roundtrip_animation`)

**Purpose:** produce the Unity-bound FBX (rigged, subdivided, textured) and round-trip companion
animation files so every file shares identical skeleton naming/orientation; prove rig fidelity.

### 1.1 Orchestrator (`pipeline/stages/export.py`)

```
1 run export_rigged(prepared.blend, textures_dir=work/textures/urp, out=work/export/<id>.fbx)
2 for each companion fbx (if cfg.export.roundtrip_companion_animations):
      run roundtrip_animation(src, out=work/export/animations/<id>@<clip>.fbx, expected_bone_names)
3 rfc = rig_fidelity_check(prepare_report.rig, export_data.rig, tolerances)        # pure, 1.4
  for each roundtrip: rfc_anim = rig_fidelity_check(prepare_report.rig, data.rig, tolerances, mode="animation")
4 write work/export/rig_fidelity_report.json ; any "fail" → RigFidelityError (stage fails)
```

Clip naming: `<clip>` = companion file stem with the model name prefix stripped if present
(`Ant_Walk.fbx` → `walk`; `Walk.fbx` → `walk`), lower-cased, non-alnum → `_`. Collisions →
`E_ANIM_NAME_COLLISION`.

Fingerprint: shas of `prepared.blend`, every URP texture, every companion FBX, `cfg.export`,
`BLENDER_JOBS_CODE_SHA`.

### 1.2 Blender job `export_rigged`

```
1 open_blend(prepared.blend)
2 if textures_dir: build Principled BSDF material "<material_name>" on all skinned meshes:
     Base Color ← basecolor (sRGB) ; Normal ← Normal Map node ← normal (Non-Color)
     Metallic ← metallic_smoothness.R (Separate Color) ; Roughness ← 1 − metallic_smoothness.A (Invert)
     Emission ← emission (sRGB) if present
   (Unity ignores Blender node trees; this only gives sensible FBX material names/texture refs and lets a
    human open the .blend and see the result.)
3 select armature + skinned meshes (+ unskinned meshes if cfg) ; export_fbx(FBX_EXPORT_KWARGS)
4 rig = rig_fingerprint(arm)  → must equal prepared fingerprint (E_BJ_RIG_MUTATED guard)
5 result: rig, fbx sha, actions exported
```

```python
FBX_EXPORT_KWARGS = dict(   # NORMATIVE (blender_jobs/fbx_settings.py)
    use_selection=True, object_types={"ARMATURE", "MESH"},
    use_mesh_modifiers=True, mesh_smooth_type="FACE", use_tspace=False,
    add_leaf_bones=False, primary_bone_axis="Y", secondary_bone_axis="X",
    armature_nodetype="NULL", use_armature_deform_only=False,
    bake_anim=True, bake_anim_use_all_bones=True, bake_anim_use_nla_strips=False,
    bake_anim_use_all_actions=True, bake_anim_force_startend_keying=True,
    bake_anim_step=1.0, bake_anim_simplify_factor=0.0,
    apply_scale_options=<cfg.export.fbx_scale_option>, global_scale=1.0,
    apply_unit_scale=True, use_space_transform=True, bake_space_transform=False,
    axis_forward="-Z", axis_up="Y",
    path_mode="COPY" if cfg.export.embed_textures else "RELATIVE", embed_textures=cfg.export.embed_textures,
    use_custom_props=False, use_metadata=True,
)
```

Spike S2 validates this dict on three vendor packs in Unity (clips play, no ×100 scale, no leaf bones,
root motion intact). Changing it requires an ADR.

### 1.3 Blender job `roundtrip_animation`

```
1 reset; import_fbx(source_fbx, FBX_IMPORT_KWARGS)
2 arm = single armature (E_BJ_NO_ARMATURE); meshes present are KEPT if they exist (some vendors ship a
  dummy mesh) but NOT subdivided/unwrapped
3 rig = rig_fingerprint(arm); missing/extra bones vs expected_bone_names recorded
4 export_fbx(FBX_EXPORT_KWARGS with object_types={"ARMATURE","MESH"}) to out_fbx
5 result: rig, actions, missing_bones, extra_bones
```

### 1.4 Rig Fidelity Check (`pipeline/rig_fidelity.py`, pure)

```python
@dataclass(frozen=True)
class RfcTolerances:
    position_abs: float = 1e-4          # bone head/tail, model units
    matrix_abs: float = 1e-4            # per element of matrix_local
    allow_extra_actions: bool = True
def rig_fidelity_check(src: RigFingerprint, out: RigFingerprint, tol: RfcTolerances,
                       mode: Literal["model","animation"] = "model") -> RigFidelityReport
```

Checks (each produces `pass | warn | fail`):

| Check | model mode | animation mode |
|---|---|---|
| armature object name equal | fail | fail |
| bone name set equal | fail | missing bones → fail; extra bones → warn |
| parent map equal (for common bones) | fail | fail |
| depth-first order equal | fail | warn |
| transform paths equal | fail | fail for common bones |
| `matrix_local` within `matrix_abs` | fail | warn (clips may legitimately differ in rest pose only if vendor exported them so; flagged) |
| head/tail within `position_abs` | fail | warn |
| actions: source actions ⊆ output actions | fail | n/a |

Overall: any `fail` → `RigFidelityError`; `warn` → `rig_fidelity = "pass_with_warnings"` in manifest and
console warning with specifics (bone name + delta).

## 2. Stage `handoff` (`pipeline/stages/handoff.py`)

```
1 build Manifest (06 §10) from: ResolvedModel, prepare/verify/apply reports, export data, texture result
2 stage into output/<id>.tmp-<run_id>/: <id>.fbx, textures/<id>_*.png, animations/<id>@*.fbx, manifest.json
3 atomic swap: rename existing output/<id> → output/<id>.old-<run_id>; rename tmp → output/<id>; delete .old
4 if cfg.library.unity_import_dir: unity.sync.copy_model(output/<id>, unity_import_dir/<id>)  (same atomic swap;
  preserves existing *.meta files for files with the same relative path so Unity GUIDs survive)
5 StageOutcome outputs = every file under output/<id>
```

Material → renderer mapping: `renderers=[]` (all skinned meshes share one material in v1).
`root_bone` = first root bone name from the rig fingerprint.

## 3. Output layout (`output/<id>/`)

```
ant_worker/
├── ant_worker.fbx
├── manifest.json
├── textures/
│   ├── ant_worker_basecolor.png
│   ├── ant_worker_normal.png
│   ├── ant_worker_metallic_smoothness.png
│   ├── ant_worker_occlusion.png          (if present)
│   └── ant_worker_emission.png           (if present)
└── animations/
    ├── ant_worker@idle.fbx
    └── ant_worker@walk.fbx
```

## 4. Unity consumer — package `com.studio.grounded-pipeline` (Editor-only)

### 4.1 Files

| File | Responsibility |
|---|---|
| `PipelineManifest.cs` | `[Serializable]` DTOs mirroring 06 §10 exactly; `static PipelineManifest Load(string jsonPath)` via `JsonUtility.FromJson` |
| `ManifestLocator.cs` | For an asset path, find `manifest.json` in the same folder (or parent for `animations/`, `textures/`) whose `fbx`/`textures[].file`/`animations[].file` references that asset. Caches per import session. |
| `TextureImportRules.cs` | Given a `ManifestTexture`, configure `TextureImporter`: `kind=="normal"` → `TextureImporterType.NormalMap`; `sRGBTexture = srgb`; `mipmapEnabled = true`; `alphaIsTransparency=false`; `textureCompression = Compressed`; for `_MetallicGlossMap` keep alpha (`alphaSource = FromInput`). |
| `UrpMaterialBuilder.cs` | Create/update `Material` with `Shader.Find(manifest.materials[i].shader)`; set textures by slot; set floats/colors; enable keywords; set `_Smoothness=1`, `_SmoothnessTextureChannel=0` when metallic map present; emission: `EnableKeyword("_EMISSION")`, `globalIlluminationFlags = RealtimeEmissive`. Saves to `Materials/<name>.mat` beside the FBX (create or overwrite in place to keep GUID). |
| `GroundedPipelinePostprocessor.cs` | `AssetPostprocessor` (see 4.2). `GetPostprocessOrder() => 100`. |
| `PipelineMenu.cs` | `Tools/Grounded Pipeline/Reimport All Pipeline Models`, `…/Validate Manifests`, `…/Open Log`. |

### 4.2 Postprocessor behaviour

```
OnPreprocessAsset():
  if asset is manifest.json → nothing (TextAsset)
OnPreprocessTexture():
  m = ManifestLocator.ForTexture(assetPath); if null return
  TextureImportRules.Apply((TextureImporter)assetImporter, m.texture)
OnPreprocessModel():
  m = ManifestLocator.ForModel(assetPath); if null return
  context.DependsOnSourceAsset(manifestPath)                       // re-import when manifest changes
  foreach texture in m.materials[*].textures: context.DependsOnArtifact(texturePath)  // textures first
  var mi = (ModelImporter)assetImporter;
  mi.animationType = ModelImporterAnimationType.Generic;
  mi.materialImportMode = ModelImporterMaterialImportMode.None;    // we build materials ourselves
  mi.importNormals = ModelImporterNormals.Import; mi.importBlendShapes = false;
  mi.bakeAxisConversion = false; mi.useFileScale = true;
  if (isAnimationFile)                                             // "<id>@<clip>.fbx"
      mi.importAnimation = true; mi.importMaterials irrelevant; (clips are picked up by Unity's @ convention)
  avatar:
    if m.rig.avatar_mode == "copy_from_source" && !string.IsNullOrEmpty(m.rig.source_avatar_asset_path)
       && (avatar = AssetDatabase.LoadAssetAtPath<Avatar>(path)) != null
          mi.avatarSetup = ModelImporterAvatarSetup.CopyFromOther; mi.sourceAvatar = avatar;
          context.DependsOnArtifact(path)
    else  mi.avatarSetup = ModelImporterAvatarSetup.CreateFromThisModel;  (warn if copy was requested)
    for animation files: avatarSetup = CopyFromOther with sourceAvatar = the model FBX's avatar
       (DependsOnArtifact(model fbx path)) so clips share the model's Avatar.
OnPostprocessModel(GameObject go):
  m = …; if null return
  foreach mat in m.materials:
     material = UrpMaterialBuilder.BuildOrUpdate(mat, folder)
     foreach renderer in go.GetComponentsInChildren<Renderer>():
        if mat.renderers.Length == 0 || mat.renderers.Contains(renderer.name): renderer.sharedMaterial = material
OnPostprocessAllAssets(...):
  if any manifest.json imported/changed → AssetDatabase.ImportAsset(fbx referenced, ForceUpdate)  // manifest edits propagate
```

Error handling: all exceptions caught per asset and surfaced via `Debug.LogError($"[GroundedPipeline]
{assetPath}: {message}")` — a bad manifest must never break unrelated imports.

### 4.3 `package.json`

```json
{ "name": "com.studio.grounded-pipeline", "version": "0.1.0", "displayName": "Grounded Pipeline Importer",
  "unity": "6000.0", "dependencies": { "com.unity.render-pipelines.universal": "17.0.0" } }
```

### 4.4 Verification checklist (manual until the EditMode tests exist)

1. Drop `output/<id>` into `Assets/Art/Creatures/`. Confirm console shows no `[GroundedPipeline]` errors.
2. Select FBX → Rig tab shows Generic + Avatar copied (or created).
3. Materials folder contains `<id>_mat.mat` with URP/Lit, all maps assigned, normal map flagged.
4. Drag model into a scene; Animator with the vendor controller (or a quick controller from the
   `@` clips) plays idle/walk without stretching or detached bones.
5. Textures: normal map import type = Normal map; metallic/occlusion sRGB off.

EditMode tests (WP-12 stretch): import `tests/fixtures/library_minimal` output into a temp Unity
project via `AssetDatabase.ImportAsset` and assert importer settings and material slots.

## 5. Tests (Python side)

- `rig_fidelity_check`: every check × pass/warn/fail; tolerances boundaries; animation-mode leniency.
- `export_rigged`/`roundtrip_animation` with `FakeScene`: material binding branches (with/without maps),
  kwargs passed exactly (`FBX_EXPORT_KWARGS` snapshot test), mutation guard.
- `handoff`: manifest construction from reports (golden file `tests/fixtures/manifest_golden.json`),
  atomic swap incl. failure mid-copy leaves previous output intact, `.meta` preservation in Unity sync,
  JsonUtility compatibility lint (no `null`, no dicts) over the produced JSON.
- C# parity test (06 §10).
