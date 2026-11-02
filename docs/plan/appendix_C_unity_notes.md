# Appendix C — Unity 6 / URP Import Notes

Reference for WP-12 and Spike S2.

## 1. Rig import

- `ModelImporter.animationType = ModelImporterAnimationType.Generic`.
- Avatar: `avatarSetup = ModelImporterAvatarSetup.CopyFromOther` + `sourceAvatar = <Avatar>` copies the
  skeleton description (incl. root node selection used for root motion) from the vendor import. Falls back
  to `CreateFromThisModel` when the source asset isn't in the project. Generic clips bind by transform
  path, so either works as long as bone names/hierarchy match — copying preserves Avatar-level settings.
- Animation files named `<model>@<clip>.fbx` in the same folder are auto-associated with `<model>.fbx`
  in the Project window; each should also get `avatarSetup = CopyFromOther` with the model's Avatar.
- Root motion: configure per clip in the Animation tab (not touched by the postprocessor in v1).
- Scale: with `FBX_SCALE_ALL`, Unity's File Scale shows 1 and bones have unit scale; with
  `FBX_SCALE_NONE`, armature objects import at scale 100 with File Scale 0.01 (visually identical, but
  physics/attachment code that reads transforms gets surprised). S2 confirms the default.
- `bakeAxisConversion=false`, `useFileUnits=true`, `useFileScale=true`, `importBlendShapes=false`
  (we dropped them), `importVisibility=false`, `importCameras=false`, `importLights=false`.
- Leaf bones: Blender's `_end` bones break clip retargeting between files that disagree; always exported
  off.

## 2. Transform paths

Unity resolves Generic clip bindings against the Animator root. For an FBX imported as a prefab, the
root GameObject is the FBX file node; the Blender armature object is a child (e.g. `Armature`), bones
below it. Our `transform_path` (`09` §2.7) is `Armature/Root/Spine`. S2 verifies the exact strings by
inspecting `AnimationUtility.GetCurveBindings(clip)[i].path` on an imported companion clip.

## 3. URP Lit material (shader `Universal Render Pipeline/Lit`, metallic workflow)

| Property | Type | Set to |
|---|---|---|
| `_BaseMap` | Texture (sRGB) | basecolor |
| `_BaseColor` | Color | white (factors already baked into the PNG) |
| `_BumpMap` | Texture (Normal map) | normal; keyword `_NORMALMAP`; `_BumpScale` = normal scale |
| `_MetallicGlossMap` | Texture (linear) | metallic_smoothness; keyword `_METALLICSPECGLOSSMAP`; `_Smoothness` = 1.0 (acts as multiplier of map alpha); `_SmoothnessTextureChannel` = 0 (alpha of metallic map) |
| `_Metallic` | Float | ignored when map present; set 0 otherwise |
| `_OcclusionMap` | Texture (linear) | occlusion (sampled from **G**); keyword `_OCCLUSIONMAP`; `_OcclusionStrength` |
| `_EmissionMap` | Texture (sRGB) | emission; keyword `_EMISSION`; `_EmissionColor` (HDR colour, linear) = factor × strength; `material.globalIlluminationFlags = MaterialGlobalIlluminationFlags.RealtimeEmissive` (or `BakedEmissive`) |
| `_WorkflowMode` | Float | 1 (Metallic) |
| `_Surface` | Float | 0 (Opaque) |

Set via `material.SetTexture/SetFloat/SetColor`, `material.EnableKeyword`. URP's `BaseShaderGUI`
normally toggles keywords in the inspector; setting them programmatically is sufficient at runtime, but
call `UnityEditor.Rendering.Universal.ShaderGraph`-independent helper
`BaseShaderGUI.SetMaterialKeywords(material)` when available (reflection-free alternative: just enable
the keywords listed; validated in the manual checklist).

## 4. Texture import settings (`TextureImporter`)

| Map | `textureType` | `sRGBTexture` | Other |
|---|---|---|---|
| basecolor | Default | true | `alphaIsTransparency=false` |
| normal | NormalMap | (n/a) | `convertToNormalmap=false` |
| metallic_smoothness | Default | **false** | keep alpha: `alphaSource = TextureImporterAlphaSource.FromInput` |
| occlusion | Default | **false** | |
| emission | Default | true | |

Normal-map convention: Unity samples tangent-space normals with +Y up (OpenGL style); glTF uses the same.
No channel flip. (DirectX-style maps would need G inverted — not our case.)

## 5. Import ordering and dependencies

In `OnPreprocessModel`, `context.DependsOnArtifact(texturePath)` ensures textures are imported before the
model so `AssetDatabase.LoadAssetAtPath<Texture2D>` succeeds in `OnPostprocessModel`;
`context.DependsOnSourceAsset(manifestPath)` re-imports the model when the manifest changes.
`AssetPostprocessor.context` is available on Unity 2020.2+.

## 6. Materials created in a postprocessor

Create under `<folder>/Materials/<name>.mat` with `AssetDatabase.CreateAsset` if missing; otherwise load
and update in place (keeps GUID and scene references). Assign to renderers of the imported `GameObject` in
`OnPostprocessModel`; modifications to the hierarchy in that callback persist into the model prefab.
`materialImportMode = None` prevents Unity from generating its own materials.

## 7. `.meta` files and sync

Unity assigns a GUID per asset in its `.meta` file. `gp unity sync` keeps existing `.meta` files for
files at the same relative path (copies the new file over the old, leaves the `.meta`), so scene/prefab
references to the FBX, textures and materials survive re-runs. New files get fresh `.meta` on import;
removed files' `.meta` are deleted together with the file.

## 8. JsonUtility constraints (manifest)

`JsonUtility.FromJson<T>` supports public fields of `[Serializable]` classes, primitives, strings,
arrays and `List<T>`; **not** `Dictionary`, nullable primitives, or polymorphism. `null` for a string
becomes `""`; for an object becomes default-constructed. Hence the manifest uses arrays of
`{name,value}` records and empty strings — see `06` §10.

## 9. Manual verification script (for S2 and WP-12)

1. New Unity 6 URP project; add package from disk (`unity/com.studio.tinyworld`).
2. Copy `output/<id>` into `Assets/Art/Creatures/<id>`.
3. Expect console: `[TinyworldPipeline] <id>.fbx configured (Generic, avatar: copy|create, materials: 1)`.
4. Inspector checks per §1/§3/§4; drag into scene; create an Animator Controller from the `@` clips;
   play; verify deformation, scale (1 unit = 1 m — compare with vendor import side by side), no missing
   bones warnings.
5. `AnimationUtility.GetCurveBindings` dump (small Editor script in the package: `Tools/Tinyworld
   Pipeline/Dump Clip Paths`) to compare with `rig_fingerprint` transform paths.
