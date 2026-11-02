using System;
using System.Collections.Generic;
using System.IO;

namespace TinyworldPipeline.Editor
{
    /// <summary>A manifest that references the asset being imported, plus where it lives.</summary>
    public sealed class LocatedManifest
    {
        /// <summary>Parsed, validated manifest.</summary>
        public PipelineManifest Manifest;

        /// <summary>Path of <c>manifest.json</c> in the same form as the asset path (project-relative or absolute).</summary>
        public string ManifestPath;

        /// <summary>Folder containing the manifest (same path form), with forward slashes and no trailing slash.</summary>
        public string Folder;

        /// <summary>Asset path relative to <see cref="Folder"/> (forward slashes).</summary>
        public string RelativePath;

        /// <summary>Texture record when the asset is one of the manifest's textures; otherwise null.</summary>
        public ManifestTexture Texture;

        /// <summary>Animation record when the asset is one of the companion animation files; otherwise null.</summary>
        public ManifestAnimation Animation;

        /// <summary>True when the asset is the main model FBX.</summary>
        public bool IsModel;

        /// <summary>True when the asset is a companion animation FBX.</summary>
        public bool IsAnimation => Animation != null;

        /// <summary>Resolves a manifest-relative file to the same path form as the asset path.</summary>
        public string Resolve(string relativeFile) => ManifestLocator.Join(Folder, relativeFile);

        /// <summary>Path of the main model FBX.</summary>
        public string ModelFbxPath => Resolve(Manifest.fbx);
    }

    /// <summary>
    /// Finds the <c>manifest.json</c> that references an asset: the asset's own folder (model FBX) or its parent
    /// (<c>textures/</c>, <c>animations/</c> subfolders). Manifests are read with System.IO so lookup works before the
    /// TextAsset itself is imported. Parsed manifests are cached per path + last-write time;
    /// <see cref="ClearCache"/> is called at the end of every import batch.
    /// </summary>
    public static class ManifestLocator
    {
        /// <summary>File name the pipeline writes next to the FBX.</summary>
        public const string ManifestFileName = "manifest.json";

        private sealed class CacheEntry
        {
            public DateTime WriteTimeUtc;
            public PipelineManifest Manifest;
            public string Error;
        }

        private static readonly Dictionary<string, CacheEntry> Cache =
            new Dictionary<string, CacheEntry>(StringComparer.OrdinalIgnoreCase);

        /// <summary>Drops every cached manifest (new import session).</summary>
        public static void ClearCache() => Cache.Clear();

        /// <summary>Manifest + texture record for a texture asset, or null when it is not a pipeline texture.</summary>
        public static LocatedManifest ForTexture(string assetPath)
        {
            var located = Locate(assetPath);
            return located != null && located.Texture != null ? located : null;
        }

        /// <summary>Manifest for a model or companion animation FBX, or null when the asset is not pipeline output.</summary>
        public static LocatedManifest ForModel(string assetPath)
        {
            var located = Locate(assetPath);
            return located != null && (located.IsModel || located.IsAnimation) ? located : null;
        }

        /// <summary>Candidate manifest paths for an asset: its own folder first, then the parent folder.</summary>
        public static IEnumerable<string> CandidateManifestPaths(string assetPath)
        {
            var folder = ParentFolder(Normalize(assetPath));
            if (string.IsNullOrEmpty(folder))
            {
                yield break;
            }
            yield return Join(folder, ManifestFileName);
            var parent = ParentFolder(folder);
            if (!string.IsNullOrEmpty(parent))
            {
                yield return Join(parent, ManifestFileName);
            }
        }

        /// <summary>Finds the first candidate manifest that references <paramref name="assetPath"/>.</summary>
        public static LocatedManifest Locate(string assetPath)
        {
            if (string.IsNullOrEmpty(assetPath))
            {
                return null;
            }
            var normalized = Normalize(assetPath);
            foreach (var manifestPath in CandidateManifestPaths(normalized))
            {
                if (!File.Exists(manifestPath))
                {
                    continue;
                }
                var manifest = LoadCached(manifestPath);
                if (manifest == null)
                {
                    continue;
                }
                var folder = ParentFolder(manifestPath);
                var relative = RelativeTo(folder, normalized);
                var located = Match(manifest, relative);
                if (located == null)
                {
                    continue;
                }
                located.ManifestPath = manifestPath;
                located.Folder = folder;
                located.RelativePath = relative;
                return located;
            }
            return null;
        }

        /// <summary>Matches a manifest-relative path against fbx / textures[].file / animations[].file.</summary>
        public static LocatedManifest Match(PipelineManifest manifest, string relativePath)
        {
            if (manifest == null || string.IsNullOrEmpty(relativePath))
            {
                return null;
            }
            if (PipelineManifest.PathsEqual(manifest.fbx, relativePath))
            {
                return new LocatedManifest { Manifest = manifest, IsModel = true };
            }
            var animation = manifest.FindAnimation(relativePath);
            if (animation != null)
            {
                return new LocatedManifest { Manifest = manifest, Animation = animation };
            }
            var texture = manifest.FindTexture(relativePath);
            if (texture != null)
            {
                return new LocatedManifest { Manifest = manifest, Texture = texture };
            }
            return null;
        }

        /// <summary>Loads and validates a manifest through the write-time cache; null (and a cached error) when invalid.</summary>
        public static PipelineManifest LoadCached(string manifestPath)
        {
            var writeTime = File.GetLastWriteTimeUtc(manifestPath);
            if (Cache.TryGetValue(manifestPath, out var entry) && entry.WriteTimeUtc == writeTime)
            {
                return entry.Manifest;
            }
            entry = new CacheEntry { WriteTimeUtc = writeTime };
            try
            {
                entry.Manifest = PipelineManifest.Load(manifestPath);
            }
            catch (Exception exception)
            {
                entry.Error = exception.Message;
                UnityEngine.Debug.LogError($"[TinyworldPipeline] {manifestPath}: {exception.Message}");
            }
            Cache[manifestPath] = entry;
            return entry.Manifest;
        }

        /// <summary>Forward-slash form of a path.</summary>
        public static string Normalize(string path) => path == null ? null : path.Replace('\\', '/').TrimEnd('/');

        /// <summary>Folder part of a forward-slash path ("" when there is none).</summary>
        public static string ParentFolder(string path)
        {
            if (string.IsNullOrEmpty(path))
            {
                return "";
            }
            var index = path.LastIndexOf('/');
            return index < 0 ? "" : path.Substring(0, index);
        }

        /// <summary>Joins a folder and a relative file with a single forward slash.</summary>
        public static string Join(string folder, string relative)
        {
            relative = Normalize(relative);
            return string.IsNullOrEmpty(folder) ? relative : folder + "/" + relative.TrimStart('/');
        }

        /// <summary>Path of <paramref name="path"/> relative to <paramref name="folder"/>, or null when outside it.</summary>
        public static string RelativeTo(string folder, string path)
        {
            if (string.IsNullOrEmpty(folder))
            {
                return path;
            }
            var prefix = folder + "/";
            return path.StartsWith(prefix, StringComparison.OrdinalIgnoreCase) ? path.Substring(prefix.Length) : null;
        }
    }
}
