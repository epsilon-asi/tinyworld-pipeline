# 0004 — Bake base colour through an Emission shader; bake several targets into one image

Date: 2026-11-02 · Status: accepted

## Context

Plan 11 §5.2 baked base colour with Cycles `DIFFUSE` / `pass_filter={"COLOR"}`. Measured on Blender
4.5.12 (WP-08 integration test): a pure red texel with metallic 0.25 came back as (221, 0, 0); even
metallic 0 loses about 4 %. Blender 4.x's Principled BSDF scales the diffuse colour pass by
`(1 − metallic)` and subtracts the dielectric layer's energy. The plan's acceptance bar was a mean
error below 2/255; the diffuse pass gave 13/255.

Separately, plan 06 §5 declared `SceneApi.bake_map(target, …)` with a single target object, but a
prepared model may contain several skinned meshes sharing one UV atlas.

## Decision

- Base colour is baked with the same EMIT trick already used for metallic and occlusion: the
  source material's `Base Color` input is rewired into an Emission shader, baked with `EMIT` at one
  sample, then restored. This is exact (0/255 measured) and deterministic.
- Emission is baked with native `EMIT` and no rewiring, so the importer's `Emission Strength`
  (from `KHR_materials_emissive_strength`) is honoured.
- `SceneApi.bake_map(targets: list[str], sources, *, …)` bakes all target meshes into one image
  (`use_clear` only on the first). `FakeScene` mirrors it; the parity test enforces it.
- The `apply` stage's bake-route fingerprint includes `blender_jobs_code_sha()` so a change to
  `bake_textures.py` re-runs cached bakes (consistent with `prepare`).

## Consequences

- Positive: exact colour transfer; `BakeTexturesData` remains unchanged; faster (1 sample).
- Negative: `BakeTexturesData.samples` reports the configured sample count although base colour
  and the EMIT maps use one sample; normal and roughness still use the configured count.
- Neutral: plan 11 §5.2 and 06 §5 text updated; plan 18 risk R-14 strengthened by measurement.

## Alternatives considered

- Keep `DIFFUSE/COLOR` and divide by `(1 − metallic)`: wrong for metallic 1 and still off by the
  dielectric loss.
- Bake per target object into separate images: breaks the single-atlas contract of `prepare`.
