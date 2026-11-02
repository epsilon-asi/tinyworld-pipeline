using System;
using System.Collections.Generic;
using System.IO;
using UnityEditor;
using UnityEngine;

namespace GroundedPipeline.Editor
{
    /// <summary>Tools/Grounded Pipeline menu: bulk re-import, manifest validation, log access and the S2 clip-path dump.</summary>
    public static class PipelineMenu
    {
        private const string Root = "Tools/Grounded Pipeline/";

        /// <summary>Force-reimports every model and companion clip referenced by a manifest under Assets/.</summary>
        [MenuItem(Root + "Reimport All Pipeline Models")]
        public static void ReimportAllPipelineModels()
        {
            var count = 0;
            AssetDatabase.StartAssetEditing();
            try
            {
                foreach (var manifestPath in FindManifests())
                {
                    var manifest = ManifestLocator.LoadCached(manifestPath);
                    if (manifest == null)
                    {
                        continue;
                    }
                    var folder = ManifestLocator.ParentFolder(manifestPath);
                    foreach (var file in ReferencedFbx(manifest))
                    {
                        var path = ManifestLocator.Join(folder, file);
                        if (File.Exists(path))
                        {
                            AssetDatabase.ImportAsset(path, ImportAssetOptions.ForceUpdate);
                            count++;
                        }
                    }
                }
            }
            finally
            {
                AssetDatabase.StopAssetEditing();
                ManifestLocator.ClearCache();
            }
            Debug.Log($"[GroundedPipeline] re-imported {count} FBX file(s)");
        }

        /// <summary>Loads every manifest under Assets/ and reports schema problems and missing referenced files.</summary>
        [MenuItem(Root + "Validate Manifests")]
        public static void ValidateManifests()
        {
            var manifests = 0;
            var problems = 0;
            foreach (var manifestPath in FindManifests())
            {
                manifests++;
                PipelineManifest manifest;
                try
                {
                    manifest = PipelineManifest.Load(manifestPath);
                }
                catch (Exception exception)
                {
                    problems++;
                    Debug.LogError($"[GroundedPipeline] {manifestPath}: {exception.Message}");
                    continue;
                }
                var folder = ManifestLocator.ParentFolder(manifestPath);
                foreach (var file in ReferencedFiles(manifest))
                {
                    if (!File.Exists(ManifestLocator.Join(folder, file)))
                    {
                        problems++;
                        Debug.LogError($"[GroundedPipeline] {manifestPath}: referenced file missing: {file}");
                    }
                }
            }
            Debug.Log($"[GroundedPipeline] validated {manifests} manifest(s), {problems} problem(s)");
        }

        /// <summary>Reveals the Editor log file (where [GroundedPipeline] messages end up).</summary>
        [MenuItem(Root + "Open Log")]
        public static void OpenLog()
        {
            var logPath = Application.consoleLogPath;
            if (string.IsNullOrEmpty(logPath) || !File.Exists(logPath))
            {
                Debug.LogWarning("[GroundedPipeline] Editor log file not found");
                return;
            }
            EditorUtility.RevealInFinder(logPath);
        }

        /// <summary>
        /// Spike S2 helper (appendix C §9 step 5): logs the distinct transform paths bound by the selected
        /// AnimationClip(s) (or every clip inside a selected FBX) so they can be compared with the rig fingerprint's
        /// <c>transform_path</c> values (expected form: <c>Armature/Root/Spine</c>).
        /// </summary>
        [MenuItem(Root + "Dump Clip Paths")]
        public static void DumpClipPaths()
        {
            var clips = SelectedClips();
            if (clips.Count == 0)
            {
                Debug.LogWarning("[GroundedPipeline] select an AnimationClip or an FBX containing clips first");
                return;
            }
            foreach (var clip in clips)
            {
                var paths = new SortedSet<string>(StringComparer.Ordinal);
                var bindings = AnimationUtility.GetCurveBindings(clip);
                foreach (var binding in bindings)
                {
                    paths.Add(string.IsNullOrEmpty(binding.path) ? "<root>" : binding.path);
                }
                var lines = new List<string> { $"[GroundedPipeline] clip '{clip.name}': {bindings.Length} bindings, {paths.Count} paths" };
                foreach (var path in paths)
                {
                    lines.Add("  " + path);
                }
                Debug.Log(string.Join("\n", lines));
            }
        }

        private static List<AnimationClip> SelectedClips()
        {
            var clips = new List<AnimationClip>();
            foreach (var selected in Selection.objects)
            {
                if (selected is AnimationClip clip)
                {
                    clips.Add(clip);
                    continue;
                }
                var path = AssetDatabase.GetAssetPath(selected);
                if (string.IsNullOrEmpty(path))
                {
                    continue;
                }
                foreach (var sub in AssetDatabase.LoadAllAssetRepresentationsAtPath(path))
                {
                    if (sub is AnimationClip subClip && !subClip.name.StartsWith("__preview__", StringComparison.Ordinal))
                    {
                        clips.Add(subClip);
                    }
                }
            }
            return clips;
        }

        /// <summary>Project-relative paths of every manifest.json under Assets/.</summary>
        private static IEnumerable<string> FindManifests()
        {
            var dataPath = ManifestLocator.Normalize(Application.dataPath);
            var projectRoot = ManifestLocator.ParentFolder(dataPath) + "/";
            foreach (var absolute in Directory.GetFiles(dataPath, ManifestLocator.ManifestFileName, SearchOption.AllDirectories))
            {
                var normalized = ManifestLocator.Normalize(absolute);
                yield return normalized.StartsWith(projectRoot, StringComparison.OrdinalIgnoreCase)
                    ? normalized.Substring(projectRoot.Length)
                    : normalized;
            }
        }

        private static IEnumerable<string> ReferencedFbx(PipelineManifest manifest)
        {
            yield return manifest.fbx;
            foreach (var animation in manifest.animations)
            {
                yield return animation.file;
            }
        }

        private static IEnumerable<string> ReferencedFiles(PipelineManifest manifest)
        {
            foreach (var file in ReferencedFbx(manifest))
            {
                yield return file;
            }
            foreach (var material in manifest.materials)
            {
                foreach (var texture in material.textures)
                {
                    yield return texture.file;
                }
            }
        }
    }
}
