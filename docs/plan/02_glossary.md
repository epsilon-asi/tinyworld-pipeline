# 02 — Glossary

| Term | Meaning in this project |
|------|-------------------------|
| **Asset library** | The root folder (`library/`) holding `library.toml`, `style/`, and `models/`. The single source of truth for inputs, intermediates, outputs and state. |
| **Model** | One rigged, animated creature/animal/bug. Lives in `library/models/<model_id>/`. |
| **model_id** | Folder name of a model. `^[a-z0-9][a-z0-9_]{1,63}$`. Used in file names, manifests, logs. |
| **model.md** | The per-model spec file: YAML front matter (settings) + Markdown body (prompt sections). |
| **Style source** | `library/style/`: shared style images and base prompts that make all models look like one game. |
| **Stage** | One idempotent step of the pipeline with declared inputs, outputs and an input fingerprint. v1 stages: `discover`, `prepare`, `texture`, `verify`, `apply`, `export`, `handoff`. |
| **Route** | The texture-application strategy chosen by `verify`: `extract` (pull textures straight out of Tripo's GLB) or `bake` (transfer Tripo's textures onto our mesh by baking in Blender). |
| **Blender job** | A unit of Blender-side work executed in a fresh headless Blender process; input `job.json`, output `result.json`. v1 jobs: `probe`, `prepare_model`, `bake_textures`, `export_rigged`, `roundtrip_animation`. |
| **Blender-side / Orchestrator-side** | Code run by Blender's bundled Python (stdlib + `bpy` only) vs code run in the project venv (full dependency set). |
| **Scene API** | The `Protocol` (`blender_jobs/scene_api.py`) that Blender jobs use instead of calling `bpy` directly, enabling a fake implementation for unit tests. |
| **Rig fingerprint** | JSON description of a skeleton: ordered bones, parent map, rest-pose matrices, transform paths, armature object name. Computed at `prepare` (source) and `export` (output); compared by the Rig Fidelity Check. |
| **Rig Fidelity Check (RFC)** | Orchestrator-side comparison of two rig fingerprints with tolerances. Failure blocks `handoff`. |
| **Surface fingerprint** | Order-independent hash of a mesh's quantized `(x, y, z, u, v)` vertex rows, plus unique-position count and triangle count. Used by `verify` to decide the route. |
| **static.glb** | The subdivided, unwrapped, *unrigged* copy of the model sent to Tripo. |
| **prepared.blend** | Saved Blender file after `prepare`: rigged, subdivided, unwrapped mesh + armature + actions. Input to `bake_textures` and `export_rigged`. |
| **result.glb** | Tripo's textured output (`model_pbr.glb`), downloaded to `work/tripo/`. |
| **URP texture set** | The PNG set Unity's URP Lit shader expects: `basecolor` (sRGB), `normal` (OpenGL +Y), `metallic_smoothness` (R = metallic, A = smoothness), `occlusion` (G), `emission` (sRGB). |
| **manifest.json** | Engine-neutral description of a delivered model: files, rig settings, material/texture mapping, provenance. Consumed by the Unity postprocessor. |
| **Companion animation** | An FBX in `source/animations/` that contains only the skeleton + an action (vendor packs that ship clips separately). Round-tripped through Blender with identical export settings so transform paths match the output model. |
| **Input fingerprint** | SHA-256 over a stage's declared inputs (file contents + relevant config subset + code version). Equal fingerprint ⇒ stage skipped. |
| **state.json** | Per-model pipeline state (`.pipeline/state.json`): stage statuses, fingerprints, outputs, Tripo task IDs, credits. |
| **Run** | One invocation of `gp run`. Has a `run_id` (ULID), a log file and a `report.json`. |
| **Credits** | Tripo's billing unit. Recorded per task from `credits_consumed`. |
| **Spike** | A time-boxed investigation producing a findings document and a decision, not shipped code (see `18_…`). |
| **WP** | Work package — the unit of assignment to a sub-agent (see `16_…`). |
| **ADR** | Architecture Decision Record (`docs/adr/NNNN-title.md`). |
| **Generic rig** | Unity Mecanim rig type for non-humanoid skeletons. Clips bind to bones by transform path. |
| **Avatar** | Unity's description of a model's skeleton. We copy it from the source import when available so root-motion/mask settings carry over. |
| **FBX_SCALE_ALL** | Blender FBX exporter option applying unit scale to all objects, avoiding the ×100 armature scale Unity otherwise shows. |
| **Leaf bones** | Extra end bones Blender's FBX exporter can add. Always disabled here. |
| **Catmull-Clark / Simple** | Subdivision types: Catmull-Clark smooths the limit surface; Simple only splits faces. |
| **Palette atlas** | A tiny texture (e.g. 16×4 px) where every face's UVs point at a flat colour swatch. Common in free low-poly packs; cannot carry a painted texture → triggers unwrap. |
| **Selected-to-Active bake** | Blender Cycles bake mode transferring surface data from selected source object(s) onto the active target object's UVs. |
| **Spike S1 / S2** | S1: Tripo behaviour validation (UV preservation, image guidance on uploads, GLB structure). S2: Blender FBX round-trip fidelity and Unity import on three real free packs. |
