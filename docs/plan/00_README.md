# Grounded-Style Asset Pipeline — Implementation Plan

**Project codename:** `grounded_pipeline` (CLI: `gp`)
**Plan version:** 1.1 — 2026-11-02 (1.0 reviewed; corrections listed in `18_…` §Changelog)
**Status:** Ready for work-package assignment

---

## What this plan is

A complete, contract-level specification for a Python tool that takes **already rigged and animated
3D models** (free or licensed packs), **increases their polygon count via subdivision**, **retextures
them through the Tripo API** in a consistent art style, and delivers **Unity-ready FBX + textures +
manifest** with animations intact.

It is written so that each work package can be handed to an independent engineer or sub-agent with no
other context than this folder. Every boundary between components is a named contract (Python
`Protocol`, pydantic model, JSON schema, or CLI flag set) so packages can be built in parallel and
integrated without surprises.

The plan also fixes the **architecture and data contracts** for the future generative pipeline
(style sources → concept images → item extraction → multi-view → Tripo mesh generation → rig), so the
code written now slots into it without rework. The future pipeline is specified at the interface level
only (see `17_future_generative_pipeline.md`).

## Reading order

| # | File | Who must read it |
|---|------|------------------|
| 00 | `00_README.md` | Everyone |
| 01 | `01_vision_and_scope.md` | Everyone |
| 02 | `02_glossary.md` | Everyone |
| 03 | `03_architecture.md` | Everyone |
| 04 | `04_repository_layout.md` | Everyone writing code |
| 05 | `05_asset_library_layout.md` | Everyone writing code; the user (it is the on-disk contract) |
| 06 | `06_data_contracts.md` | Everyone writing code |
| 07 | `07_config_and_secrets.md` | WP-01, WP-05, WP-11 |
| 08 | `08_cli_spec.md` | WP-11, WP-10 |
| 09 | `09_stage_prepare.md` | WP-04, WP-03 |
| 10 | `10_stage_texture.md` | WP-05, WP-06 |
| 11 | `11_stage_verify_and_apply.md` | WP-07, WP-08 |
| 12 | `12_stage_export_and_unity_handoff.md` | WP-09, WP-12 |
| 13 | `13_orchestration_and_state.md` | WP-10 |
| 14 | `14_errors_and_observability.md` | Everyone writing code |
| 15 | `15_testing_strategy.md` | Everyone writing code |
| 16 | `16_work_packages.md` | Project lead; each WP owner reads their own entry |
| 17 | `17_future_generative_pipeline.md` | Architects; WP-10 (extension points) |
| 18 | `18_risks_decisions_and_spikes.md` | Project lead; WP-14 |
| 19 | `19_coding_standards.md` | Everyone writing code |
| A | `appendix_A_tripo_api.md` | WP-05, WP-06, WP-14 |
| B | `appendix_B_blender_notes.md` | WP-04, WP-08, WP-09 |
| C | `appendix_C_unity_notes.md` | WP-12 |

## How to use this plan with sub-agents

1. Start with **WP-00** (repo bootstrap) and **WP-14 Spike S1/S2** in parallel. Everything else depends
   on WP-00; S1 resolves the single largest unknown (Tripo UV preservation).
2. Hand each remaining WP to one agent. Give the agent: this README, `03`, `04`, `06`, `14`, `15`,
   `19`, its own WP entry in `16`, and the stage doc(s) its entry lists.
3. Each WP ends with a PR that satisfies the **Definition of Done** in `16_work_packages.md` §DoD:
   code + tests (≥80 % line/branch/function coverage for the package touched) + docs updated.
4. Integration happens on the `main` branch behind the contracts; no WP may change a contract without
   an ADR (`19_coding_standards.md` §ADRs) and a note in `18_risks_decisions_and_spikes.md`.

## Conventions used in these documents

- `MUST` / `MUST NOT` / `SHOULD` / `MAY` carry RFC-2119 meaning.
- Code blocks labelled `python` are **normative signatures**: implement exactly that name, module path
  and signature. Bodies shown are illustrative unless marked `# NORMATIVE`.
- File paths are relative to the repository root unless they start with `library/` (asset library root)
  or `Assets/` (Unity project root).
- `<model_id>` is the folder name of a model inside `library/models/`.
- "Blender-side" means code executed by Blender's bundled Python (no third-party packages available).
  "Orchestrator-side" means code executed by the project's own Python 3.12 virtual environment.

## One-paragraph summary of the v1 pipeline

```
library/models/<id>/source/*.fbx  ──▶ [prepare: Blender]  subdivide, (unwrap), fingerprint rig,
                                       export static.glb + prepared.blend
                                  ──▶ [texture: Tripo]    upload static.glb, POST /v3/models/texture
                                       with prompt + style image, poll, download result.glb
                                  ──▶ [verify]            compare surface fingerprints → route
                                  ──▶ [apply]             route=extract: pull PNGs from result.glb
                                                          route=bake:    Blender selected-to-active bake
                                       then pack to URP texture set
                                  ──▶ [export: Blender]   bind textures, export FBX (+ companion anim FBX),
                                       rig fidelity check against source fingerprint
                                  ──▶ [handoff]           manifest.json, copy to Unity, C# postprocessor
                                                          sets Generic rig, copies Avatar, builds URP materials
```

The rigged file **never leaves Blender**; only a static copy goes to Tripo. Animations therefore
cannot be damaged by the texturing provider, and the only run-time variable is whether the
`extract` or `bake` route runs.
