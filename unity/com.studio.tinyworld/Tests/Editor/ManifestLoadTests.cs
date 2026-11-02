using System.IO;
using NUnit.Framework;
using UnityEngine;

namespace TinyworldPipeline.Editor.Tests
{
    /// <summary>
    /// EditMode tests: JsonUtility round-trips the golden manifest (byte-identical to
    /// <c>tests/fixtures/manifest_golden.json</c>; <c>tests/unit/unity/test_package_files.py</c> keeps the copy below
    /// in sync) and <see cref="ManifestLocator"/> resolves assets in the manifest folder and its subfolders.
    /// </summary>
    public class ManifestLoadTests
    {
        // GOLDEN-BEGIN
        private const string GoldenJson = @"{
  ""schema_version"": 1,
  ""pipeline_version"": ""0.1.0"",
  ""generated_at"": ""2026-11-02T12:00:00Z"",
  ""model_id"": ""cube_rigged"",
  ""display_name"": ""Rigged Cube"",
  ""fbx"": ""cube_rigged.fbx"",
  ""rig"": {
    ""animation_type"": ""Generic"",
    ""avatar_mode"": ""copy_from_source"",
    ""source_avatar_asset_path"": ""Assets/Vendor/Cubes/cube_rigged.fbx"",
    ""root_bone"": ""Root""
  },
  ""materials"": [
    {
      ""name"": ""cube_rigged_mat"",
      ""shader"": ""Universal Render Pipeline/Lit"",
      ""renderers"": [],
      ""textures"": [
        {
          ""slot"": ""_BaseMap"",
          ""file"": ""textures/cube_rigged_basecolor.png"",
          ""srgb"": true,
          ""kind"": ""default""
        },
        {
          ""slot"": ""_BumpMap"",
          ""file"": ""textures/cube_rigged_normal.png"",
          ""srgb"": false,
          ""kind"": ""normal""
        },
        {
          ""slot"": ""_MetallicGlossMap"",
          ""file"": ""textures/cube_rigged_metallic_smoothness.png"",
          ""srgb"": false,
          ""kind"": ""default""
        }
      ],
      ""floats"": [
        {
          ""name"": ""_WorkflowMode"",
          ""value"": 1.0
        },
        {
          ""name"": ""_Surface"",
          ""value"": 0.0
        },
        {
          ""name"": ""_Smoothness"",
          ""value"": 1.0
        },
        {
          ""name"": ""_SmoothnessTextureChannel"",
          ""value"": 0.0
        },
        {
          ""name"": ""_BumpScale"",
          ""value"": 1.0
        }
      ],
      ""colors"": [
        {
          ""name"": ""_BaseColor"",
          ""r"": 1.0,
          ""g"": 1.0,
          ""b"": 1.0,
          ""a"": 1.0
        }
      ],
      ""keywords"": [
        ""_NORMALMAP"",
        ""_METALLICSPECGLOSSMAP""
      ]
    }
  ],
  ""animations"": [
    {
      ""file"": ""animations/cube_rigged@walk.fbx"",
      ""clip_name"": ""walk""
    }
  ],
  ""provenance"": {
    ""source_fbx_sha256"": ""3f3f3f3f3f3f3f3f3f3f3f3f3f3f3f3f3f3f3f3f3f3f3f3f3f3f3f3f3f3f3f3f"",
    ""source_license"": ""CC0 — synthetic fixture"",
    ""subdivision_levels"": 1,
    ""subdivision_type"": ""catmull_clark"",
    ""uv_decision"": ""kept"",
    ""texture_provider"": ""tripo"",
    ""texture_task_id"": ""task_cube_rigged_0001"",
    ""texture_route"": ""extract"",
    ""texture_seed"": 4242,
    ""prompt_sha256"": ""a7a7a7a7a7a7a7a7a7a7a7a7a7a7a7a7a7a7a7a7a7a7a7a7a7a7a7a7a7a7a7a7"",
    ""credits_consumed"": 20.0,
    ""rig_fidelity"": ""pass""
  }
}";
        // GOLDEN-END

        [Test]
        public void FromJson_ReadsEveryBlockOfTheGoldenManifest()
        {
            var manifest = PipelineManifest.FromJson(GoldenJson);
            manifest.Validate();

            Assert.AreEqual(1, manifest.schema_version);
            Assert.AreEqual("0.1.0", manifest.pipeline_version);
            Assert.AreEqual("cube_rigged", manifest.model_id);
            Assert.AreEqual("Rigged Cube", manifest.display_name);
            Assert.AreEqual("cube_rigged.fbx", manifest.fbx);

            Assert.AreEqual("Generic", manifest.rig.animation_type);
            Assert.AreEqual("copy_from_source", manifest.rig.avatar_mode);
            Assert.AreEqual("Assets/Vendor/Cubes/cube_rigged.fbx", manifest.rig.source_avatar_asset_path);
            Assert.AreEqual("Root", manifest.rig.root_bone);

            Assert.AreEqual(1, manifest.materials.Length);
            var material = manifest.materials[0];
            Assert.AreEqual("cube_rigged_mat", material.name);
            Assert.AreEqual("Universal Render Pipeline/Lit", material.shader);
            Assert.AreEqual(0, material.renderers.Length);
            Assert.AreEqual(3, material.textures.Length);
            Assert.AreEqual("_BaseMap", material.textures[0].slot);
            Assert.IsTrue(material.textures[0].srgb);
            Assert.AreEqual("default", material.textures[0].kind);
            Assert.AreEqual("_BumpMap", material.textures[1].slot);
            Assert.IsFalse(material.textures[1].srgb);
            Assert.AreEqual("normal", material.textures[1].kind);
            Assert.AreEqual("_MetallicGlossMap", material.textures[2].slot);
            Assert.AreEqual("textures/cube_rigged_metallic_smoothness.png", material.textures[2].file);
            Assert.AreEqual(5, material.floats.Length);
            Assert.AreEqual("_WorkflowMode", material.floats[0].name);
            Assert.AreEqual(1.0f, material.floats[0].value);
            Assert.AreEqual("_SmoothnessTextureChannel", material.floats[3].name);
            Assert.AreEqual(0.0f, material.floats[3].value);
            Assert.AreEqual(1, material.colors.Length);
            Assert.AreEqual("_BaseColor", material.colors[0].name);
            Assert.AreEqual(1.0f, material.colors[0].a);
            CollectionAssert.AreEqual(new[] { "_NORMALMAP", "_METALLICSPECGLOSSMAP" }, material.keywords);

            Assert.AreEqual(1, manifest.animations.Length);
            Assert.AreEqual("animations/cube_rigged@walk.fbx", manifest.animations[0].file);
            Assert.AreEqual("walk", manifest.animations[0].clip_name);

            Assert.AreEqual(1, manifest.provenance.subdivision_levels);
            Assert.AreEqual(4242, manifest.provenance.texture_seed);
            Assert.AreEqual(20.0f, manifest.provenance.credits_consumed);
            Assert.AreEqual("pass", manifest.provenance.rig_fidelity);
        }

        [Test]
        public void ToJson_RoundTripsWithoutLosingFields()
        {
            var manifest = PipelineManifest.FromJson(GoldenJson);
            var again = PipelineManifest.FromJson(JsonUtility.ToJson(manifest));
            Assert.AreEqual(JsonUtility.ToJson(manifest), JsonUtility.ToJson(again));
            Assert.AreEqual(manifest.materials[0].textures.Length, again.materials[0].textures.Length);
            Assert.AreEqual(manifest.provenance.prompt_sha256, again.provenance.prompt_sha256);
        }

        [Test]
        public void Validate_RejectsUnsupportedSchemaVersion()
        {
            var manifest = PipelineManifest.FromJson(GoldenJson.Replace("\"schema_version\": 1", "\"schema_version\": 2"));
            Assert.Throws<InvalidDataException>(() => manifest.Validate());
        }

        [Test]
        public void Locator_ResolvesModelTextureAndAnimationFromTheirFolders()
        {
            var root = Path.Combine(Path.GetTempPath(), "gp_locator_" + Path.GetRandomFileName()).Replace('\\', '/');
            try
            {
                Directory.CreateDirectory(root + "/textures");
                Directory.CreateDirectory(root + "/animations");
                File.WriteAllText(root + "/manifest.json", GoldenJson);
                ManifestLocator.ClearCache();

                var model = ManifestLocator.ForModel(root + "/cube_rigged.fbx");
                Assert.IsNotNull(model);
                Assert.IsTrue(model.IsModel);
                Assert.IsFalse(model.IsAnimation);
                Assert.AreEqual(root, model.Folder);
                Assert.AreEqual(root + "/manifest.json", model.ManifestPath);

                var texture = ManifestLocator.ForTexture(root + "/textures/cube_rigged_normal.png");
                Assert.IsNotNull(texture);
                Assert.AreEqual("_BumpMap", texture.Texture.slot);
                Assert.AreEqual("textures/cube_rigged_normal.png", texture.RelativePath);

                var animation = ManifestLocator.ForModel(root + "/animations/cube_rigged@walk.fbx");
                Assert.IsNotNull(animation);
                Assert.IsTrue(animation.IsAnimation);
                Assert.AreEqual("walk", animation.Animation.clip_name);
                Assert.AreEqual(root + "/cube_rigged.fbx", animation.ModelFbxPath);

                Assert.IsNull(ManifestLocator.ForModel(root + "/other.fbx"));
                Assert.IsNull(ManifestLocator.ForTexture(root + "/textures/unrelated.png"));
                Assert.IsNull(ManifestLocator.ForTexture(root + "/cube_rigged.fbx"));
            }
            finally
            {
                ManifestLocator.ClearCache();
                if (Directory.Exists(root))
                {
                    Directory.Delete(root, true);
                }
            }
        }

        [Test]
        public void Locator_CandidatesAreOwnFolderThenParent()
        {
            var candidates = new System.Collections.Generic.List<string>(
                ManifestLocator.CandidateManifestPaths("Assets/Art/cube_rigged/textures/cube_rigged_basecolor.png"));
            CollectionAssert.AreEqual(
                new[] { "Assets/Art/cube_rigged/textures/manifest.json", "Assets/Art/cube_rigged/manifest.json" },
                candidates);
        }
    }
}
