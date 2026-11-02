# 15 — Testing Strategy

## 1. Targets

| Suite | Marker | Runs where | Gate |
|---|---|---|---|
| Unit | (none) | every PR, every OS in matrix | ≥ 80 % **line, branch and function** coverage over `src/tinyworld_pipeline` (omit list in `04` §2); zero flakes; < 90 s wall time with `-n auto` |
| Blender integration | `blender` | CI job with Blender 4.5 LTS; locally when `GP_BLENDER_EXE` set | must pass on `main` |
| Tripo live | `tripo_live` | developer machine only | manual; feeds Spike S1 |
| Unity EditMode | n/a (C#) | manual / optional CI | checklist in `12` §4.4 |

Coverage is measured with `pytest-cov --cov-branch`; **function coverage** is computed by
`tools/check_function_coverage.py`, which reads `coverage json` and counts functions (via `ast`) with at
least one executed line; threshold 80 %. Both gates run in CI (`just test`).

Per-package expectation (so 80 % overall isn't achieved by over-testing easy code):

| Package | Minimum |
|---|---|
| `domain`, `config`, `library`, `gltf`, `textures`, `texturing/prompt_builder` | 95 % |
| `pipeline` (runner, stages, state) | 90 % |
| `tripo` | 90 % |
| `blender/runner.py`, `blender/locate.py` | 85 % |
| `blender_jobs/jobs/*`, `uv_heuristics`, `rig_fingerprint` | 85 % (via `FakeScene`) |
| `cli` | 80 % |
| `blender_jobs/bpy_scene.py`, `blender_jobs/entry.py` | excluded from unit coverage; integration-tested |

## 2. Test layout and naming

```
tests/
  conftest.py                 # shared fixtures: tmp_library, fake_ctx, fake_scene, respx_mock, frozen_clock
  fakes/                      # importable test doubles (NOT fixtures): deterministic, documented
  fixtures/                   # static data: library_minimal/, fbx/, tripo/*.json, glb/, manifest_golden.json
  unit/<mirror of src>/test_<module>.py
  integration/blender/test_<job>.py
  integration/tripo/test_live_smoke.py
```

- Test names: `test_<unit>_<condition>_<expected>` (e.g. `test_decide_route_uv_mismatch_position_match_returns_bake`).
- Arrange / Act / Assert sections separated by a blank line; one behaviour per test; parametrize tables
  for decision functions.
- No `time.sleep`, no network, no real subprocess in unit tests (use injected `Sleeper`, `respx`, fake
  executable).
- `pytest-randomly` on: tests must be order-independent. Seeds fixed for Hypothesis (`derandomize=True`
  in CI profile).

## 3. Test doubles (normative behaviour)

### 3.1 `FakeScene` (`tests/fakes/fake_scene.py`) — implements `SceneApi`

In-memory scene: `objects: dict[str, FakeObject]` where `FakeObject(kind="MESH"|"ARMATURE",
verts, faces, tris, uv_layers: dict[str, list[list[(u,v)]]], materials, modifiers: list[Modifier],
shapekeys: bool, parent_armature, bones: list[BoneInfo], image_sizes)`. Operations mutate counts
predictably (`add_subsurf`+`apply_modifier` → faces×4^levels, verts updated with the Catmull-Clark
formula for closed quads, tris = faces×2), `smart_unwrap` replaces UVs with a synthetic non-overlapping
atlas, `clear_shapekeys` flips the flag, `export_*` records `(objs, path, kwargs)` calls and writes an
empty file at `path`, `rig_fingerprint` reads `bones`. Builders: `scene_with_rigged_mesh(levels_of_detail)`,
`scene_palette_uv()`, `scene_two_meshes_shapekeys()`, `scene_no_armature()`, `scene_two_armatures()`.

Every `SceneApi` method is exercised by at least one job test; a parity test asserts `FakeScene` and
`BpyScene` expose identical method names/signatures (`inspect.signature`).

### 3.2 `FakeBlenderRunner` (`tests/fakes/fake_blender_runner.py`)

Scripted: `FakeBlenderRunner(responses: dict[JobName, Callable[[JobRequest], JobResult | Exception]])`.
Records every request; writes the declared output files (empty or from a fixture) so stages' output
checks pass; can raise `BlenderTimeoutError` etc. on demand; counts calls (used to assert skipping).

### 3.3 Fake Blender executable (for `SubprocessBlenderRunner` tests)

`tests/fakes/fake_blender_exe.py` is launched via `sys.executable` with a wrapper shell script
registered as the executable; it parses `-- --job … --result …`, and based on `params["__fake"]`
(`"ok"`, `"job_error"`, `"crash"`, `"no_result"`, `"sleep:30"`) writes results / exits / sleeps.
Also answers `--version` with a configurable string for `locate.py` tests.

### 3.4 `FakeTextureProvider`

Returns a `TextureResult` pointing at a synthetic GLB produced by `glb_builder` (configurable: same
UVs as input, re-unwrapped UVs, remeshed, multi-material, with/without each map); configurable
credits; can raise provider errors; records `resume_task_id`; fires `on_task_created`.

### 3.5 `glb_builder` (`tests/fakes/glb_builder.py`)

Pure-Python construction of valid GLBs with `pygltflib`: `quad()`, `cube()`, `from_mesh(positions,
uvs, indices, *, split_seams, shuffle, scale, translate, primitives, materials, images)`. Embedded 2×2 or
4×4 PNGs with known pixel values so extraction/packing can be asserted exactly.

### 3.6 Tripo fixtures

`tests/fixtures/tripo/*.json` are recorded real responses (secrets stripped, URLs rewritten to
`https://fixture.local/…`). `respx` routes return them. A `conftest` fixture `tripo_happy_path` wires
upload → create → queued → running → success → download (serving a `glb_builder` GLB).

## 4. Unit test inventory per module (minimum)

| Module | Must-cover branches |
|---|---|
| `config/layering` | each precedence layer wins; explicit-default override; list replacement; env parsing of bool/int/list; unknown env warns; unknown toml key errors |
| `library/frontmatter_io` | valid; YAML error; extra key; missing `## prompt`; heading case-insensitivity; CRLF files |
| `library/validation` | every `E_*`/`W_*` in `05` §8 (one test each) |
| `library/discovery` | explicit primary; single fbx; ambiguous; globs sorted; tags filter |
| `texturing/prompt_builder` | placeholders, include_sections, whitespace, >600 warn, >1000 error, sha stable |
| `blender/locate` | each search location (monkeypatched FS), version regex, unsupported version policy |
| `blender/runner` | ok, job_error passthrough, crash, no_result, timeout+kill, cancel, semaphore, protocol version mismatch, env passed |
| `blender_jobs/jobs/prepare_model` | §2.10 list in `09` |
| `blender_jobs/uv_heuristics` | property tests + decision table |
| `blender_jobs/rig_fingerprint` | order stability, path building, hash sensitivity |
| `tripo/client` | §8 list in `10` |
| `tripo/budget`, `tripo/polling` | reserve/commit/release; exceed; thread-safety (two threads); rate limiter timing with fake clock |
| `texturing/tripo_provider` | four modes; resume; token upload failures; credits fallback |
| `gltf/*` | §8 list in `11` |
| `textures/urp_pack` | per-channel formulas; resampling; missing maps; factors; hypothesis |
| `pipeline/stages/*` | fingerprint inputs (changing each declared input changes fp; changing anything else doesn't); happy path; error path; outputs written |
| `pipeline/runner`, `state_store`, `hashing` | §10 list in `13` |
| `pipeline/rig_fidelity` | §5 in `12` |
| `unity/manifest_writer`, `unity/sync` | golden manifest; JsonUtility lint; atomic swap; `.meta` preservation; C# parity |
| `cli/*` | every command × success + each exit code; `--json` schema; `--dry-run` table |
| `observability/logging` | redaction processor; dual sinks; bound keys present |
| `domain/errors` | code uniqueness; exit codes; `Problem` rendering |

## 5. Integration tests (`@blender`)

Fixtures generated once by `tools/make_fixtures.py` (run inside Blender) and committed:

- `rigged_cube.fbx`: cube, 3-bone chain armature (`Root`→`Spine`→`Head`), skin weights, one 24-frame
  action `idle`, proper UVs, a 4×4 texture.
- `rigged_cube@walk.fbx`: same armature, action `walk`, no mesh.
- `palette_cube.fbx`: same but UVs collapsed onto a 16×4 palette texture (overlapping).
- `shapekey_cube.fbx`: with one shape key.

Tests: `prepare_model` on each (asserting counts, decisions, errors), `bake_textures` (textured cube →
itself; colour tolerance), `export_rigged` + `roundtrip_animation` (rig fidelity passes; FBX re-imports
in Blender with same bones), full `gp run --to-stage prepare` on `library_minimal` through the real
`SubprocessBlenderRunner`.

## 6. CLI tests

`CliRunner` + `build_app(ctx_factory)` injecting fakes; assertions on exit code, stdout (rich disabled
via `--no-color`/`--json`), state files written. Golden `--json` outputs validated against pydantic
models (`CliOutput`).

## 7. Determinism and flake policy

- All randomness seeded; Hypothesis `derandomize=True` in CI; `pytest-randomly` seed printed.
- Any flaky test is quarantined within 24 h (`@pytest.mark.flaky` is **not** allowed; fix or delete).
- Tests touching the filesystem use `tmp_path`; never the repo tree.

## 8. Definition of "tested" for a WP

A WP is tested when: every public function/class in its modules has at least one direct test; every
error code it can raise has a test; every config key it reads has a test proving it changes behaviour
(or the fingerprint); coverage thresholds in §1 are met for its package; and the integration test (if
listed in its WP entry) exists, even if marked and skipped in unit CI.
