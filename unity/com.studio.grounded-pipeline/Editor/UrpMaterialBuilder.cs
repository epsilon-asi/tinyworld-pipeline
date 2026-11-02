using System.IO;
using UnityEditor;
using UnityEngine;

namespace GroundedPipeline.Editor
{
    /// <summary>
    /// Creates or updates <c>Materials/&lt;name&gt;.mat</c> next to the model FBX from a <see cref="ManifestMaterial"/>
    /// (appendix C §3, §6). Updating in place keeps the material GUID so scene and prefab references survive.
    /// </summary>
    public static class UrpMaterialBuilder
    {
        /// <summary>Subfolder (relative to the manifest folder) that receives generated materials.</summary>
        public const string MaterialsFolder = "Materials";

        /// <summary>Keyword that also switches the material's global-illumination flags.</summary>
        public const string EmissionKeyword = "_EMISSION";

        /// <summary>Project-relative path of the material asset for <paramref name="material"/>.</summary>
        public static string MaterialAssetPath(string folder, ManifestMaterial material) =>
            ManifestLocator.Join(ManifestLocator.Join(folder, MaterialsFolder), material.name + ".mat");

        /// <summary>
        /// Builds (or refreshes) the material asset for <paramref name="material"/>; textures are resolved relative to
        /// <paramref name="folder"/> (the manifest folder, project-relative). Returns the saved material.
        /// </summary>
        public static Material BuildOrUpdate(ManifestMaterial material, string folder)
        {
            var shader = Shader.Find(material.shader);
            if (shader == null)
            {
                throw new InvalidDataException($"shader '{material.shader}' not found (is URP installed?)");
            }
            var assetPath = MaterialAssetPath(folder, material);
            var existing = AssetDatabase.LoadAssetAtPath<Material>(assetPath);
            var result = existing != null ? existing : new Material(shader) { name = material.name };
            if (result.shader != shader)
            {
                result.shader = shader;
            }
            Configure(result, material, folder);
            if (existing == null)
            {
                EnsureFolder(assetPath);
                AssetDatabase.CreateAsset(result, assetPath);
            }
            EditorUtility.SetDirty(result);
            AssetDatabase.SaveAssetIfDirty(result);
            return result;
        }

        /// <summary>Applies textures, floats, colours and keywords from the manifest to <paramref name="target"/>.</summary>
        public static void Configure(Material target, ManifestMaterial material, string folder)
        {
            foreach (var texture in material.textures)
            {
                var texturePath = ManifestLocator.Join(folder, texture.file);
                var loaded = AssetDatabase.LoadAssetAtPath<Texture2D>(texturePath);
                if (loaded == null)
                {
                    Debug.LogWarning($"[GroundedPipeline] {texturePath}: texture not imported yet; slot {texture.slot} left empty");
                }
                target.SetTexture(texture.slot, loaded);
            }
            foreach (var entry in material.floats)
            {
                target.SetFloat(entry.name, entry.value);
            }
            foreach (var color in material.colors)
            {
                target.SetColor(color.name, new Color(color.r, color.g, color.b, color.a));
            }
            var emissive = false;
            foreach (var keyword in material.keywords)
            {
                target.EnableKeyword(keyword);
                emissive |= keyword == EmissionKeyword;
            }
            target.globalIlluminationFlags = emissive
                ? MaterialGlobalIlluminationFlags.RealtimeEmissive
                : MaterialGlobalIlluminationFlags.EmissiveIsBlack;
            SyncUrpKeywords(target);
        }

        /// <summary>
        /// Lets URP's shader GUI derive the remaining keywords/render state from the property values we set
        /// (same call the Lit inspector makes). Our explicit keywords above already suffice at runtime.
        /// </summary>
        private static void SyncUrpKeywords(Material target)
        {
#if GP_URP
            try
            {
                BaseShaderGUI.SetMaterialKeywords(target, UnityEditor.Rendering.Universal.ShaderGUI.LitGUI.SetMaterialKeywords);
            }
            catch (System.Exception exception)
            {
                Debug.LogWarning($"[GroundedPipeline] {target.name}: URP keyword sync skipped: {exception.Message}");
            }
#endif
        }

        private static void EnsureFolder(string assetPath)
        {
            var folder = ManifestLocator.ParentFolder(assetPath);
            if (string.IsNullOrEmpty(folder) || AssetDatabase.IsValidFolder(folder))
            {
                return;
            }
            var parent = ManifestLocator.ParentFolder(folder);
            var leaf = folder.Substring(parent.Length + 1);
            if (!AssetDatabase.IsValidFolder(parent))
            {
                EnsureFolder(folder);
            }
            AssetDatabase.CreateFolder(parent, leaf);
        }
    }
}
