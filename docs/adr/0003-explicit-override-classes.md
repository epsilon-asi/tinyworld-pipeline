# 0003 — Per-model override models are explicit module-level classes

Date: 2026-11-02 · Status: accepted

## Context

Plan 1.0 §06 declared per-model overrides inline, e.g. `geometry: make_partial(GeometrySection) | None`.
Pydantic evaluates such an annotation at class-creation time and accepts it, but `mypy --strict`
(a Definition-of-Done gate) cannot type a call expression used as an annotation and reports an error
on every such field. The plan's own rule says code must pass mypy strict.

## Decision

`domain/model_spec.py` declares one explicit, hand-written class per section (`GeometryOverrides`,
`UvOverrides`, `TextureOverrides`, `VerifyOverrides`, `BakeOverrides`, `ExportOverrides`,
`UnityOverrides`): every section field repeated as `Optional` with default `None`, same constraints,
`extra="forbid"`. `TextureOverrides` and `UnityOverrides` add the two model-only fields
(`guidance_images`, `source_asset_path`), listed in `OVERRIDE_EXTRA_FIELDS` so layering skips them.
`config/layering.py` keeps `make_partial(section_cls, *, name)`; it is not used at runtime but by
`tests/unit/domain/test_overrides_parity.py`, which generates the expected partial for each section and
asserts field names, optionality and inner types match the hand-written class. Drift fails CI.

## Consequences

- Positive: mypy strict is clean; JSON schema for `model.md` front matter has stable, named types;
  error messages name the override class.
- Negative: a new section field must be added in two places (caught by the parity test).
- Neutral: this is a normative-signature deviation from plan 1.0 §06 §3, recorded as decision D-15.

## Alternatives considered

- Generate the classes at import with `make_partial` and declare them again under `TYPE_CHECKING`
  for mypy: the same duplication, plus a runtime/static split that is harder to read.
- `# type: ignore` on each field: hides real errors and violates `19` §1.
