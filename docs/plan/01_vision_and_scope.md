# 01 — Vision, Scope, Success Criteria

## Vision

Build an in-house, repeatable asset pipeline that lets a small team produce a **visually unified
creature roster** in a *Grounded*-adjacent art style (smooth shading, hand-painted/stylized PBR,
saturated colours, no outlines) from heterogeneous inputs:

1. **Now (v1):** free or licensed models that are already rigged and animated. We keep their skeletons
   and animations untouched, raise their polygon density so they read as smooth rather than faceted,
   and re-texture them with Tripo so every creature shares one palette and material language.
2. **Later (v2+):** brand-new assets generated from our own style sources (style images and base
   prompts → concept images → extracted items → multi-view sheets → Tripo meshes → rigging). Only the
   architecture and contracts for this are fixed now.

Everything is driven from a **single on-disk asset library**: one folder per model containing its
source files, its prompt (Markdown), its reference images, intermediates, outputs and pipeline state.
The folder *is* the database.

## Goals (v1)

| ID | Goal | Measure |
|----|------|---------|
| G1 | Animations survive the pipeline unchanged | Rig fidelity check passes (bone names, hierarchy, transform paths, rest pose within tolerance); all source clips play correctly in Unity on the output model |
| G2 | Higher polygon density with smooth silhouettes | Configurable subdivision (0–3 levels, Catmull-Clark or Simple); weights and UVs interpolated automatically |
| G3 | Consistent textures across models | One shared style image + base prompt composed with per-model prompt; fixed seeds; PBR maps packed for URP |
| G4 | Zero manual steps per model after setup | `gp run` processes a whole library; resumable; idempotent |
| G5 | Unity-ready output | FBX + PNG set + `manifest.json`; Editor postprocessor configures Generic rig, Avatar, URP Lit materials automatically |
| G6 | Production quality code | Typed, tested (≥80 % line/branch/function), linted, CI-gated, documented, observable |
| G7 | Cost control | Dry-run credit estimate, per-run credit budget, per-model credit accounting |

## Non-goals (v1)

- Generating new meshes (future pipeline — interfaces only).
- Retargeting animations between different skeletons.
- Preserving blend shapes through subdivision (shape keys are dropped/skipped/failed per policy; a
  future WP may add a shape-key-preserving subdivision path).
- AI remesh + weight transfer path (explicitly deferred; the architecture leaves a `GeometryStrategy`
  seam for it — see `09_stage_prepare.md` §Extension).
- A GUI. The CLI plus the Unity Editor menu are the only user interfaces.
- Supporting Unreal or Godot output (manifest is engine-neutral; only the Unity consumer is built).
- Humanoid (Mecanim Humanoid) rigs. Creatures are Generic rigs. Humanoid is a one-flag extension in the
  Unity postprocessor but is untested in v1.

## Users and environments

- **Primary user:** the studio's technical artist / gameplay programmer running the CLI on a
  **macOS (Apple Silicon)** workstation. Linux is supported for CI. Windows as a *host* for `gp` is
  deferred (ADR-0002); outputs are plain files and are consumed by Unity on any OS.
- **Blender:** 4.5 LTS installed locally (4.2 LTS tolerated; 5.x: see `18_risks…` R-07).
- **Unity:** Unity 6 (6000.x) with URP. Generic rigs.
- **Tripo:** paid API tier with an API key (free tier outputs carry commercial-use restrictions —
  see R-12).

## Success criteria for the v1 release

1. Running `gp run` against a library containing at least three models from three different free
   publishers (e.g. Quaternius, Kenney, Omabuarts) completes with zero manual intervention and produces
   outputs that import into a clean Unity 6 URP project with materials assigned and all source
   animations playing.
2. Re-running `gp run` with no changes performs no Blender or Tripo work (all stages cached).
3. Changing only the prompt in one `model.md` re-runs only `texture → handoff` for that model.
4. `pytest --cov` reports ≥80 % line, branch and function coverage over `src/grounded_pipeline`
   (excluding the thin `bpy` adapter module, which is covered by the Blender integration suite).
5. `gp run --dry-run` prints an accurate credit estimate; a run that would exceed the configured budget
   refuses to start Tripo tasks.
6. All risks in `18_risks_decisions_and_spikes.md` marked **blocking** have a recorded decision.

## Assumptions and defaults (override in `library.toml`)

| Topic | Default | Rationale |
|-------|---------|-----------|
| Python | 3.12 (`>=3.12,<3.14`) | Modern typing, `tomllib`, independent of Blender's Python |
| Blender execution | subprocess `blender -b --python … -- --job …` | Works with any installed Blender; no ABI coupling |
| Subdivision | 1 level, Catmull-Clark, shade smooth on | Softens faceted free packs toward the target look; ~4× faces |
| Shape keys | `fail` (explicit per-model opt-in to `drop`) | Silent feature loss is worse than a one-line config edit |
| UV handling | `auto` (unwrap when palette/overlapping UVs detected) | Free packs commonly use 16×4 px palette atlases that cannot carry a painted texture |
| Texture provider | Tripo `v3.5-20260815`, quality `standard`, PBR on, `bake: true`, fixed seed | Newest model; seed gives reproducibility |
| Texture route | `auto` (extract if UVs preserved, else bake) | Resolves Tripo's undocumented UV behaviour at run time |
| Bake resolution | 2048 px, 16 samples, margin 16 px | Matches typical creature texel density |
| FBX export | Blender FBX, `add_leaf_bones=False`, all actions baked, `FBX_SCALE_ALL` | Unity-friendly; validated by Spike S2 |
| Unity material | `Universal Render Pipeline/Lit`, metallic workflow | URP default |
| Concurrency | 2 models in parallel, 1 Blender job at a time | Blender is CPU-bound; Tripo waits are I/O-bound |
| Credits budget | unlimited unless `tripo.max_credits_per_run` set | Opt-in guard |
