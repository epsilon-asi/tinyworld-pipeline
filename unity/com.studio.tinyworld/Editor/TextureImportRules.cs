using UnityEditor;

namespace TinyworldPipeline.Editor
{
    /// <summary>Applies appendix C §4 texture import settings from a manifest texture record.</summary>
    public static class TextureImportRules
    {
        /// <summary>Slot whose alpha channel carries smoothness and must survive import.</summary>
        public const string MetallicGlossSlot = "_MetallicGlossMap";

        /// <summary>Texture kind that imports as a tangent-space normal map.</summary>
        public const string NormalKind = "normal";

        /// <summary>Configures <paramref name="importer"/> for <paramref name="texture"/>.</summary>
        public static void Apply(TextureImporter importer, ManifestTexture texture)
        {
            if (importer == null || texture == null)
            {
                return;
            }
            var isNormal = texture.kind == NormalKind;
            importer.textureType = isNormal ? TextureImporterType.NormalMap : TextureImporterType.Default;
            if (isNormal)
            {
                // Unity and glTF are both +Y (OpenGL) tangent space; the PNG already is a normal map.
                importer.convertToNormalmap = false;
            }
            else
            {
                importer.sRGBTexture = texture.srgb;
            }
            importer.mipmapEnabled = true;
            importer.alphaIsTransparency = false;
            importer.textureCompression = TextureImporterCompression.Compressed;
            if (texture.slot == MetallicGlossSlot)
            {
                // Alpha carries smoothness (appendix C §4); never let Unity drop it.
                importer.alphaSource = TextureImporterAlphaSource.FromInput;
            }
        }
    }
}
