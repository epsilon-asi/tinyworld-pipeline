# 16 — Work Packages

Each WP is sized S (≤ 1 day), M (2–3 days), L (4–6 days) for one engineer/agent. "Reads" lists the plan
documents the agent must read besides `00, 03, 04, 06, 14, 15, 19`. Every WP ends with a PR meeting the
Definition of Done (§DoD).

## Dependency graph and waves

```
Wave 0: WP-00 bootstrap ─────────────┐        WP-14a Spike S1 (Tripo)   WP-14b Spike S2 (Blender/Unity)
Wave 1: WP-01 domain+config ◀────────┘
        WP-16 fixtures & fakes (needs WP-01 types)
Wave 2: WP-02 library    WP-03 blender runner    WP-05 tripo client    WP-07 gltf+textures
Wave 3: WP-04 prepare job   WP-06 texture provider+stage   WP-08 bake job   WP-09 export+RFC
Wave 4: WP-10 orchestrator (needs all stages)   WP-12 unity handoff (needs WP-09)
Wave 5: WP-11 CLI   WP-13 observability polish   WP-15 docs/runbook   WP-17 end-to-end validation
```

Wave-2 packages are independent of each other and can run fully in parallel.

---

## WP-00 — Repository bootstrap (S)
**Depends:** none. **Reads:** 04, 19.
**Scope:** `pyproject.toml` exactly as `04` §2; `uv.lock`; `src/tinyworld_pipeline/__init__.py` (version
only); package skeleton with empty `__init__.py` files for every module directory in `04` §1; `justfile`;
`.pre-commit-config.yaml` (ruff, ruff-format, mypy, import-linter); `.importlinter` contracts encoding
`03` §2; CI workflow (`04` §4) including Blender download step and `tools/check_function_coverage.py`;
`docs/plan/` copy; ADR template + `0001-two-runtimes.md`, `0002-macos-primary-host.md`,
`0003-explicit-override-classes.md`.
**Acceptance:** `just ci` passes on an empty package with a placeholder test; CI green on Linux (and
locally on macOS);
import-linter fails if `domain` imports `httpx` (prove with a temporary test commit, then revert).

## WP-01 — Domain models and configuration (M)
**Depends:** WP-00. **Reads:** 05, 06, 07.
**Scope:** all pydantic models in `06` §§1–11 (`domain/*`, `blender/protocol.py`, `tripo/models.py`,
`texturing/provider.py` types only); `config/defaults.py`, `config/settings.py`, `config/layering.py`
with `make_partial`; `domain/errors.py` (`14` §2); `blender_jobs/protocol.py` dataclass mirrors +
`to_dict/from_dict`; parity test.
**Acceptance:** `tests/unit/domain`, `tests/unit/config` ≥ 95 %; `make_partial` only overrides
explicitly-set fields; JSON schema export for `Manifest` committed to `docs/schemas/manifest.schema.json`
(generated, checked in CI for drift).

## WP-02 — Asset library: discovery, parsing, validation, scaffolding (M)
**Depends:** WP-01. **Reads:** 05, 09 §1.
**Scope:** `library/frontmatter_io.py`, `discovery.py`, `validation.py` (every code in `05` §8),
`scaffold.py` (`init`, `new-model` templates), `paths.py` (`ModelPaths`); `texturing/prompt_builder.py`;
`pipeline/stages/discover.py`.
**Acceptance:** `tests/fixtures/library_minimal` validates clean; one fixture per error code fails with
that code; prompt builder property tests; discover stage fingerprint tests.

## WP-03 — Blender runner and executable discovery (M)
**Depends:** WP-01. **Reads:** 07 §4, 13 §6, appendix B.
**Scope:** `blender/locate.py`, `blender/runner.py` (`SubprocessBlenderRunner`), `blender_jobs/entry.py`
(argv parsing, `sys.path` injection via `GP_SRC_DIR`, dispatch table, result writing, exit codes,
exception → `JobError` with traceback), `blender_jobs/jobs/probe.py`, fake Blender executable for tests.
**Acceptance:** runner tests in `13` §10 pass with the fake executable; `probe` runs under real Blender
in the integration job; protocol-version mismatch rejected; timeout kills the process tree on POSIX
(test with the fake exe `sleep`); the Windows branch (`CREATE_NEW_PROCESS_GROUP`, `taskkill /T`) is
written behind `sys.platform` but only exercised when the Windows host port lands (ADR-0002).

## WP-04 — Blender job `prepare_model` (L)
**Depends:** WP-03, WP-16. **Reads:** 09, appendix B.
**Scope:** `blender_jobs/scene_api.py` (full Protocol incl. bake/material methods declared for later
WPs), `bpy_scene.py` (adapter), `uv_heuristics.py`, `rig_fingerprint.py`, `fbx_settings.py`
(import kwargs + export kwargs constants), `jobs/prepare_model.py` with `SubdivideStrategy`;
`pipeline/stages/prepare.py`.
**Acceptance:** `09` §2.10 tests; integration tests on all four FBX fixtures; `prepared.blend` reopens;
`static.glb` loads with `gltf/loader` (coordinate with WP-07 — until merged, use `pygltflib` directly in
the test); rig-mutation guard proven by a deliberately mutating fake strategy.

## WP-05 — Tripo client (M)
**Depends:** WP-01. **Reads:** 10 §§3–5, appendix A.
**Scope:** `tripo/client.py`, `upload.py` (multipart + presign), `polling.py` (`RateLimiter`,
`wait_for_task`), `budget.py`, `errors.py` mapping; recording hook; `gp tripo balance/task` command
bodies (CLI wiring in WP-11).
**Acceptance:** `10` §8 client tests with `respx` ≥ 90 %; key redaction test; ambiguous-create policy
implemented and tested; live smoke test file present (skipped without key).

## WP-06 — Texture provider and `texture` stage (M)
**Depends:** WP-05, WP-02. **Reads:** 10.
**Scope:** `texturing/tripo_provider.py` (guidance mapping, uploads, resume, `on_task_created`),
`pipeline/stages/texture.py`, budget integration, request/task JSON outputs,
`FakeTextureProvider` (coordinate with WP-16).
**Acceptance:** `10` §8 provider/stage tests; fingerprint sensitivity matrix; resume path makes zero
create calls.

## WP-07 — GLB tools and URP packing (M)
**Depends:** WP-01, WP-16 (glb_builder). **Reads:** 11 §§1–2, 5.1, 6.
**Scope:** `gltf/loader.py`, `fingerprint.py`, `verify.py` (`decide_route`), `extract.py`;
`textures/urp_pack.py`, `image_io.py`; `pipeline/stages/verify.py`; the extract half of
`pipeline/stages/apply.py` (bake half stubbed to raise until WP-08).
**Acceptance:** `11` §8 tests for loader/fingerprint/decide/extract/pack ≥ 95 %; hypothesis tests;
scale-normalisation path.

## WP-08 — Blender job `bake_textures` and bake route (L)
**Depends:** WP-04, WP-07. **Reads:** 11 §5.2, appendix B §3.
**Scope:** `jobs/bake_textures.py`; `SceneApi` bake methods in `bpy_scene.py`; bake half of
`pipeline/stages/apply.py` incl. metallic+roughness merge and multi-material fallback; `apply_report.json`.
**Acceptance:** fake-scene tests for the map loop and errors; integration test bakes a textured cube
onto itself within colour tolerance (mean abs error < 2/255 on basecolor, normal ≈ (128,128,255)).

## WP-09 — Export job, companion animation round-trip, Rig Fidelity Check (M)
**Depends:** WP-04. **Reads:** 12 §1.
**Scope:** `jobs/export_rigged.py`, `jobs/roundtrip_animation.py`, material binding in `bpy_scene.py`,
`pipeline/rig_fidelity.py`, `pipeline/stages/export.py`, clip naming.
**Acceptance:** `12` §5 tests; integration: export `rigged_cube` + `@walk` round-trip, RFC passes,
re-import in Blender yields identical bone names; `FBX_EXPORT_KWARGS` snapshot test.

## WP-10 — Orchestrator: runner, state store, hashing, locks, cancellation (L)
**Depends:** WP-02…WP-09 (interfaces; can start against fakes after WP-01/WP-16). **Reads:** 13.
**Scope:** `pipeline/stage.py`, `graph.py`, `runner.py`, `context.py`, `state_store.py`, `hashing.py`,
`timing.py`, `locks.py`, `report.py`; dry-run planner; invalidation; budget wiring.
**Acceptance:** `13` §10 tests ≥ 90 %; end-to-end run over `library_minimal` with all fakes produces
`output/<id>/manifest.json` matching the golden file; second run performs zero stage executions.

## WP-11 — CLI (M)
**Depends:** WP-10. **Reads:** 08.
**Scope:** `cli/app.py`, `render.py`, every command in `08` §2, `--json` outputs, exit codes,
`build_app(ctx_factory)` seam.
**Acceptance:** every command × exit code tested with `CliRunner`; `gp --help` snapshots committed;
`gp doctor` runs probe through the fake executable.

## WP-12 — Unity handoff: manifest, sync, C# package (M)
**Depends:** WP-09, WP-10 (handoff stage hook). **Reads:** 12 §§2–4, appendix C.
**Scope:** `unity/manifest_writer.py`, `unity/sync.py`, `pipeline/stages/handoff.py`; Unity package
files in `04` §1; EditMode test skeleton; C# parity test.
**Acceptance:** golden manifest; atomic swap tests; `.meta` preservation; manual checklist `12` §4.4
executed on Unity 6 URP with the `library_minimal` output and screenshots attached to the PR.

## WP-13 — Observability (S)
**Depends:** WP-00; integrates with everything. **Reads:** 14.
**Scope:** `observability/logging.py` (processors, sinks, redaction), `console.py`; per-stage log tee;
report environment block; log pruning.
**Acceptance:** redaction test; dual-sink test; a run log contains `run_id`/`model_id`/`stage` on every
line (asserted over a fake run).

## WP-14a — Spike S1: Tripo behaviour validation (S, time-boxed 1 day)
**Depends:** WP-05 (or raw `httpx` script). **Reads:** 10, appendix A, 18 (R-01, R-02, R-03).
**Scope:** With a real key: upload a small UV-mapped GLB (`rigged_cube` static export), texture it with
(a) text, (b) text+style_image, (c) image, (d) images; record whether vertex count/UVs are preserved,
GLB structure (primitives, materials, image formats, texture conventions), credits per call, timings.
**Deliverable:** `docs/spikes/S1-tripo-findings.md` + recorded fixtures (secrets stripped) + decisions
written into `18` R-01..R-03.

## WP-14b — Spike S2: Blender FBX round-trip and Unity import on real packs (S, 1 day)
**Depends:** WP-03/WP-04 drafts or a manual Blender script. **Reads:** 09, 12, appendices B, C.
**Scope:** Take one model each from Quaternius (CC0), Kenney (CC0) and Omabuarts free pack; round-trip
through `FBX_IMPORT_KWARGS`/`FBX_EXPORT_KWARGS` with `FBX_SCALE_ALL` and `FBX_SCALE_NONE`; import in
Unity 6 URP; check bone paths, scale, leaf bones, clip playback, root motion; check whether Unity's
transform path includes the armature object name.
**Deliverable:** `docs/spikes/S2-fbx-roundtrip-findings.md`; decisions into `18` R-04..R-06; any kwargs
change via ADR.

## WP-15 — Documentation and runbook (S)
**Depends:** WP-11. **Scope:** `README.md` (install, first run), `docs/runbook.md` (daily use, adding a
model, re-running a stage, budget), `docs/troubleshooting.md` (every `E_*` with fix), example library.

## WP-16 — Test fixtures and fakes (M)
**Depends:** WP-01. **Reads:** 15 §3, §5.
**Scope:** `tests/fakes/*` (FakeScene, FakeBlenderRunner, fake exe, FakeTextureProvider, glb_builder),
`tools/make_fixtures.py` + committed FBX fixtures, `tests/fixtures/library_minimal`, Tripo JSON fixtures
(placeholder until S1 records real ones), `conftest.py` fixtures.
**Acceptance:** fakes have their own tests (a fake with a bug poisons every suite); FBX fixtures load in
Blender integration smoke test.

## WP-17 — End-to-end validation on real assets (S)
**Depends:** everything. **Scope:** Build a real library with the three S2 models; run `gp run`; import
into Unity; record results and credits; file issues; update `01` success criteria checkboxes.

---

## Definition of Done (every WP)

1. Code matches the normative signatures/paths in the plan; deviations have an ADR linked in the PR.
2. `just ci` green: ruff, ruff-format, mypy `--strict`, import-linter, pytest with coverage gates.
3. Package-level coverage meets `15` §1 minimums; new error codes documented in the relevant plan doc.
4. Public functions have docstrings (Google style) stating inputs, outputs, raised errors.
5. Integration tests listed in the WP exist and pass under real Blender where applicable.
6. The WP's doc section is updated if behaviour or contracts were refined (same PR).
7. PR description: what, why, how tested, screenshots/logs for anything visual (Unity, console).
8. No TODOs without an issue link; no commented-out code; no `print`.

## Agent briefing template

```
You are implementing WP-XX of the tinyworld_pipeline project.
Read: docs/plan/00,03,04,06,14,15,19 + <WP reads>. Your WP entry is in docs/plan/16 §WP-XX.
Constraints: follow normative signatures exactly; do not modify other packages except where the WP says
"coordinate"; do not change contracts without writing an ADR and flagging it in your PR.
Deliver: code + tests meeting DoD; run `just ci`; summarize decisions and any open questions.
```
