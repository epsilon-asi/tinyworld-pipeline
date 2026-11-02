using System;
using System.IO;
using UnityEngine;

namespace TinyworldPipeline.Editor
{
    /// <summary>
    /// JsonUtility DTOs mirroring <c>tinyworld_pipeline.domain.manifest</c> (plan 06 §10).
    /// Field names MUST stay identical to the pydantic models; <c>tests/unit/unity/test_manifest_cs_parity.py</c>
    /// enforces this. JsonUtility rules: public fields, primitives, strings, arrays, nested [Serializable] classes.
    /// </summary>
    [Serializable]
    public class PipelineManifest
    {
        /// <summary>Manifest schema this package understands.</summary>
        public const int SupportedSchemaVersion = 1;

        public int schema_version;
        public string pipeline_version;
        public string generated_at;
        public string model_id;
        public string display_name;
        public string fbx;
        public ManifestRig rig;
        public ManifestMaterial[] materials;
        public ManifestAnimation[] animations;
        public ManifestProvenance provenance;

        /// <summary>Reads and validates a manifest from an absolute or project-relative file path.</summary>
        /// <exception cref="InvalidDataException">The file is not a supported pipeline manifest.</exception>
        public static PipelineManifest Load(string jsonPath)
        {
            var json = File.ReadAllText(jsonPath);
            var manifest = FromJson(json);
            manifest.Validate(jsonPath);
            return manifest;
        }

        /// <summary>Parses manifest JSON text without validating it.</summary>
        public static PipelineManifest FromJson(string json)
        {
            var manifest = JsonUtility.FromJson<PipelineManifest>(json);
            if (manifest == null)
            {
                throw new InvalidDataException("manifest JSON is empty or malformed");
            }
            return manifest;
        }

        /// <summary>Throws when the manifest is unusable; <paramref name="origin"/> only decorates the message.</summary>
        /// <exception cref="InvalidDataException">Unsupported schema version or missing required fields.</exception>
        public void Validate(string origin = "manifest")
        {
            if (schema_version != SupportedSchemaVersion)
            {
                throw new InvalidDataException(
                    $"{origin}: schema_version {schema_version} is not supported (expected {SupportedSchemaVersion})");
            }
            if (string.IsNullOrEmpty(model_id))
            {
                throw new InvalidDataException($"{origin}: model_id is empty");
            }
            if (string.IsNullOrEmpty(fbx))
            {
                throw new InvalidDataException($"{origin}: fbx is empty");
            }
            if (rig == null || string.IsNullOrEmpty(rig.animation_type))
            {
                throw new InvalidDataException($"{origin}: rig block is missing");
            }
            materials = materials ?? Array.Empty<ManifestMaterial>();
            animations = animations ?? Array.Empty<ManifestAnimation>();
            foreach (var material in materials)
            {
                if (string.IsNullOrEmpty(material.name) || string.IsNullOrEmpty(material.shader))
                {
                    throw new InvalidDataException($"{origin}: material without name or shader");
                }
                material.renderers = material.renderers ?? Array.Empty<string>();
                material.textures = material.textures ?? Array.Empty<ManifestTexture>();
                material.floats = material.floats ?? Array.Empty<ManifestFloat>();
                material.colors = material.colors ?? Array.Empty<ManifestColor>();
                material.keywords = material.keywords ?? Array.Empty<string>();
            }
        }

        /// <summary>Returns true when <paramref name="file"/> (manifest-relative, forward slashes) is one of the animation files.</summary>
        public ManifestAnimation FindAnimation(string file)
        {
            if (animations == null)
            {
                return null;
            }
            foreach (var animation in animations)
            {
                if (PathsEqual(animation.file, file))
                {
                    return animation;
                }
            }
            return null;
        }

        /// <summary>Finds the texture record referencing <paramref name="file"/> across all materials.</summary>
        public ManifestTexture FindTexture(string file)
        {
            if (materials == null)
            {
                return null;
            }
            foreach (var material in materials)
            {
                if (material.textures == null)
                {
                    continue;
                }
                foreach (var texture in material.textures)
                {
                    if (PathsEqual(texture.file, file))
                    {
                        return texture;
                    }
                }
            }
            return null;
        }

        /// <summary>Case-insensitive, separator-insensitive comparison of manifest-relative paths.</summary>
        public static bool PathsEqual(string a, string b)
        {
            if (a == null || b == null)
            {
                return false;
            }
            return string.Equals(a.Replace('\\', '/'), b.Replace('\\', '/'), StringComparison.OrdinalIgnoreCase);
        }
    }

    /// <summary>Rig import settings (plan 06 §10).</summary>
    [Serializable]
    public class ManifestRig
    {
        public string animation_type;
        public string avatar_mode;
        public string source_avatar_asset_path;
        public string root_bone;
    }

    /// <summary>One URP material to build (plan 06 §10, appendix C §3).</summary>
    [Serializable]
    public class ManifestMaterial
    {
        public string name;
        public string shader;
        public string[] renderers;
        public ManifestTexture[] textures;
        public ManifestFloat[] floats;
        public ManifestColor[] colors;
        public string[] keywords;
    }

    /// <summary>Texture slot assignment plus its import colour space / type.</summary>
    [Serializable]
    public class ManifestTexture
    {
        public string slot;
        public string file;
        public bool srgb;
        public string kind;
    }

    /// <summary>Named float property.</summary>
    [Serializable]
    public class ManifestFloat
    {
        public string name;
        public float value;
    }

    /// <summary>Named colour property (linear components).</summary>
    [Serializable]
    public class ManifestColor
    {
        public string name;
        public float r;
        public float g;
        public float b;
        public float a;
    }

    /// <summary>Companion animation file (<c>&lt;id&gt;@&lt;clip&gt;.fbx</c>).</summary>
    [Serializable]
    public class ManifestAnimation
    {
        public string file;
        public string clip_name;
    }

    /// <summary>Provenance block; informational only for the importer.</summary>
    [Serializable]
    public class ManifestProvenance
    {
        public string source_fbx_sha256;
        public string source_license;
        public int subdivision_levels;
        public string subdivision_type;
        public string uv_decision;
        public string texture_provider;
        public string texture_task_id;
        public string texture_route;
        public int texture_seed;
        public string prompt_sha256;
        public float credits_consumed;
        public string rig_fidelity;
    }
}
