using System;
using System.Collections.Generic;
using System.IO;
using UnityEditor;
using UnityEngine;

namespace GroundedPipeline.Editor
{
    /// <summary>
    /// Configures grounded_pipeline output on import (plan 12 §4.2): texture import settings, Generic rig with a
    /// copied (or created) Avatar, URP/Lit materials assigned to the imported renderers, and re-import of the model
    /// when its manifest changes. Every callback is isolated so one bad manifest never breaks unrelated imports.
    /// </summary>
    public sealed class GroundedPipelinePostprocessor : AssetPostprocessor
    {
        /// <summary>Runs after default postprocessors so our settings win.</summary>
        public override int GetPostprocessOrder() => 100;

        private const string AvatarModeCopy = "copy_from_source";

        private void OnPreprocessTexture()
        {
            try
            {
                var located = ManifestLocator.ForTexture(assetPath);
                if (located == null)
                {
                    return;
                }
                TextureImportRules.Apply((TextureImporter)assetImporter, located.Texture);
            }
            catch (Exception exception)
            {
                Report(assetPath, exception);
            }
        }

        private void OnPreprocessModel()
        {
            try
            {
                var located = ManifestLocator.ForModel(assetPath);
                if (located == null)
                {
                    return;
                }
                ConfigureModelImporter((ModelImporter)assetImporter, located);
            }
            catch (Exception exception)
            {
                Report(assetPath, exception);
            }
        }

        private void OnPostprocessModel(GameObject root)
        {
            try
            {
                var located = ManifestLocator.ForModel(assetPath);
                if (located == null || !located.IsModel)
                {
                    return;
                }
                var count = AssignMaterials(root, located);
                var avatarMode = located.Manifest.rig.avatar_mode == AvatarModeCopy ? "copy" : "create";
                Debug.Log($"[GroundedPipeline] {Path.GetFileName(assetPath)} configured (Generic, avatar: {avatarMode}, materials: {count})");
            }
            catch (Exception exception)
            {
                Report(assetPath, exception);
            }
        }

        private static void OnPostprocessAllAssets(
            string[] importedAssets, string[] deletedAssets, string[] movedAssets, string[] movedFromAssetPaths)
        {
            try
            {
                var imported = new HashSet<string>(importedAssets, StringComparer.OrdinalIgnoreCase);
                foreach (var path in importedAssets)
                {
                    if (!IsManifest(path))
                    {
                        continue;
                    }
                    ReimportReferencedModels(path, imported);
                }
            }
            catch (Exception exception)
            {
                Report("OnPostprocessAllAssets", exception);
            }
            finally
            {
                ManifestLocator.ClearCache();
            }
        }

        /// <summary>Applies plan 12 §4.2 model settings; shared by model and companion animation files.</summary>
        private void ConfigureModelImporter(ModelImporter importer, LocatedManifest located)
        {
            context.DependsOnSourceAsset(located.ManifestPath);
            importer.animationType = ModelImporterAnimationType.Generic;
            importer.materialImportMode = ModelImporterMaterialImportMode.None;
            importer.importNormals = ModelImporterNormals.Import;
            importer.importBlendShapes = false;
            importer.bakeAxisConversion = false;
            importer.useFileScale = true;
            importer.useFileUnits = true;
            importer.importVisibility = false;
            importer.importCameras = false;
            importer.importLights = false;

            if (located.IsAnimation)
            {
                importer.importAnimation = true;
                ConfigureAnimationAvatar(importer, located);
                return;
            }
            foreach (var material in located.Manifest.materials)
            {
                foreach (var texture in material.textures)
                {
                    context.DependsOnArtifact(located.Resolve(texture.file));
                }
            }
            ConfigureModelAvatar(importer, located);
        }

        private void ConfigureModelAvatar(ModelImporter importer, LocatedManifest located)
        {
            var rig = located.Manifest.rig;
            if (rig.avatar_mode == AvatarModeCopy && !string.IsNullOrEmpty(rig.source_avatar_asset_path))
            {
                var avatar = AssetDatabase.LoadAssetAtPath<Avatar>(rig.source_avatar_asset_path);
                if (avatar != null)
                {
                    importer.avatarSetup = ModelImporterAvatarSetup.CopyFromOther;
                    importer.sourceAvatar = avatar;
                    context.DependsOnArtifact(rig.source_avatar_asset_path);
                    return;
                }
                Debug.LogWarning($"[GroundedPipeline] {assetPath}: avatar copy requested but '{rig.source_avatar_asset_path}' has no Avatar; creating from this model");
            }
            else if (rig.avatar_mode == AvatarModeCopy)
            {
                Debug.LogWarning($"[GroundedPipeline] {assetPath}: avatar copy requested but source_avatar_asset_path is empty; creating from this model");
            }
            importer.avatarSetup = ModelImporterAvatarSetup.CreateFromThisModel;
            importer.sourceAvatar = null;
        }

        /// <summary>Companion clips share the model FBX's Avatar so bindings resolve identically.</summary>
        private void ConfigureAnimationAvatar(ModelImporter importer, LocatedManifest located)
        {
            var modelPath = located.ModelFbxPath;
            context.DependsOnArtifact(modelPath);
            var avatar = AssetDatabase.LoadAssetAtPath<Avatar>(modelPath);
            if (avatar != null)
            {
                importer.avatarSetup = ModelImporterAvatarSetup.CopyFromOther;
                importer.sourceAvatar = avatar;
                return;
            }
            Debug.LogWarning($"[GroundedPipeline] {assetPath}: model '{modelPath}' has no Avatar yet; creating from this file (re-import after the model imports)");
            importer.avatarSetup = ModelImporterAvatarSetup.CreateFromThisModel;
            importer.sourceAvatar = null;
        }

        /// <summary>Builds every manifest material and assigns it to the matching renderers. Returns the material count.</summary>
        private static int AssignMaterials(GameObject root, LocatedManifest located)
        {
            var renderers = root.GetComponentsInChildren<Renderer>(true);
            var count = 0;
            foreach (var record in located.Manifest.materials)
            {
                var material = UrpMaterialBuilder.BuildOrUpdate(record, located.Folder);
                count++;
                foreach (var renderer in renderers)
                {
                    if (record.renderers.Length != 0 && Array.IndexOf(record.renderers, renderer.name) < 0)
                    {
                        continue;
                    }
                    var slots = Math.Max(1, renderer.sharedMaterials.Length);
                    var shared = new Material[slots];
                    for (var i = 0; i < slots; i++)
                    {
                        shared[i] = material;
                    }
                    renderer.sharedMaterials = shared;
                }
            }
            return count;
        }

        /// <summary>Force-reimports the FBX files a changed manifest references, unless they were in the same batch.</summary>
        private static void ReimportReferencedModels(string manifestPath, HashSet<string> importedInBatch)
        {
            var manifest = ManifestLocator.LoadCached(manifestPath);
            if (manifest == null)
            {
                return;
            }
            var folder = ManifestLocator.ParentFolder(ManifestLocator.Normalize(manifestPath));
            var targets = new List<string> { ManifestLocator.Join(folder, manifest.fbx) };
            foreach (var animation in manifest.animations)
            {
                targets.Add(ManifestLocator.Join(folder, animation.file));
            }
            foreach (var target in targets)
            {
                if (importedInBatch.Contains(target) || !File.Exists(target))
                {
                    continue;
                }
                var captured = target;
                EditorApplication.delayCall += () => AssetDatabase.ImportAsset(captured, ImportAssetOptions.ForceUpdate);
            }
        }

        private static bool IsManifest(string path) =>
            string.Equals(Path.GetFileName(path), ManifestLocator.ManifestFileName, StringComparison.OrdinalIgnoreCase);

        private static void Report(string path, Exception exception) =>
            Debug.LogError($"[GroundedPipeline] {path}: {exception.Message}");
    }
}
