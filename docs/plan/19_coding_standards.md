# 19 — Coding Standards, Conventions, ADRs

## 1. Language and typing

- Python 3.12+. `from __future__ import annotations` in every module.
- `mypy --strict` clean. No `Any` except at JSON boundaries (`dict[str, Any]` for `JobRequest.params`
  / `JobResult.data`, immediately validated into models). No `# type: ignore` without a reason code and
  comment.
- pydantic v2 for all boundary data; `@dataclass(frozen=True)` for internal value objects and for
  everything Blender-side. Models are immutable (`frozen=True`) unless they are mutable state
  (`ModelState`).
- `pathlib.Path` everywhere; strings only at subprocess/JSON boundaries. Paths in JSON are relative +
  forward-slash (`PurePosixPath` for serialisation).
- Enums: `StrEnum` for anything serialised.

## 2. Structure and style

- `ruff` config in `04` §2 is the law (line length 100, import sorting, bugbear, security, pylint
  subset). `ruff format` for formatting; no manual alignment.
- Module size ≤ ~400 lines; function ≤ ~50 lines; cyclomatic complexity ≤ 10 (`PLR0912/0915` active).
- One public concept per module; private helpers prefixed `_`.
- Docstrings (Google style) on every public module, class and function: purpose, Args, Returns, Raises.
  Document *why* in comments, not *what*.
- No `print` outside `cli/render.py` and `observability/console.py`. No module-level side effects
  (logging configured by the CLI entry point only).
- Dependency injection through constructors / `PipelineContext`; no global singletons; no
  `importlib`-time I/O (exception: `BLENDER_JOBS_CODE_SHA` computed lazily on first use and cached).
- Pure functions for all decision logic (heuristics, fingerprints, route decision, RFC, packing).
  Effects (file writes, subprocess, HTTP) in thin wrappers.
- Blender-side code: stdlib + `bpy`/`bmesh`/`mathutils` (+ Blender's bundled `numpy`) only; no f-string
  logging of large data;
  catch exceptions at the job boundary only (`entry.py`), never swallow inside jobs.

## 3. Error handling

- Raise `PipelineError` subclasses with precise messages and a `hint` where a one-line fix exists.
- Wrap third-party exceptions at the adapter boundary (`httpx` → `TripoTransientError`, etc.); include
  `cause`.
- Never catch `Exception` except at: stage boundary in the runner (to record `E_UNEXPECTED`), Blender
  `entry.py`, CLI top level.
- Resource cleanup with context managers; temp dirs via `tempfile.TemporaryDirectory` (kept when
  `GP_KEEP_TEMP=1`).

## 4. Files and atomicity

- All outputs written to a temp sibling then `os.replace`d (`pipeline/atomic.py` helpers:
  `atomic_write_bytes`, `atomic_write_json`, `atomic_replace_dir`).
- JSON written with `indent=2`, `sort_keys=True`, `ensure_ascii=False`, trailing newline.
- Never modify anything under `source/`, `references/`, `model.md`, `style/`.

## 5. Concurrency

- Threads only (no asyncio). Shared state guarded by `threading.Lock`; prefer immutable messages.
- Every blocking wait accepts a `CancelToken` and a timeout.
- Blender subprocesses are the only child processes; always `start_new_session=True` (POSIX) /
  `CREATE_NEW_PROCESS_GROUP` (Windows) so the tree can be killed.

## 6. Logging

- `structlog.get_logger(__name__)`; bind context, don't format it into messages.
- Event names are `noun.verb` constants in `observability/events.py`.
- Levels: DEBUG for per-request/per-poll detail; INFO for stage lifecycle; WARNING for degraded but
  continuing (fallback to bake, RFC warnings); ERROR for stage failures.

## 7. Tests

- See `15`. Additionally: tests import production code through public module paths; no reaching into
  `_private` names; fixtures over setup methods; `tmp_path` for all file I/O; `respx` for HTTP;
  property tests for pure numeric code.

## 8. Git and reviews

- Conventional Commits (`feat(tripo): …`, `fix(prepare): …`, `test(gltf): …`, `docs(plan): …`).
- One WP per PR where possible; PR template includes the DoD checklist (`16` §DoD).
- Required reviews: 1 (2 for contract changes). CI must be green. Squash merge.
- Branch names: `wp-07-gltf-tools`, `spike-s1-tripo`.

## 9. ADRs (`docs/adr/NNNN-title.md`)

Template:

```
# NNNN — Title
Date · Status (proposed/accepted/superseded by NNNN)
## Context
## Decision
## Consequences (positive, negative, neutral)
## Alternatives considered
```

An ADR is required for: any change to a normative signature/path in this plan; any change to
`FBX_IMPORT_KWARGS`/`FBX_EXPORT_KWARGS`/`GLB_EXPORT_KWARGS`; changes to the manifest schema; new
external dependencies; relaxing coverage or lint rules.

## 10. Security and privacy

- Secrets only via env/`.env`; `.env` gitignored; `SecretStr` type; redaction processor.
- Uploads to Tripo contain only mesh geometry and our own reference images; never vendor textures
  (respect pack licences) unless the user explicitly lists them as guidance images.
- Downloads are written to `work/` and checked for content type before use; no execution of downloaded
  content; Blender runs with `--factory-startup` (no third-party add-ons).
- Pinned dependency versions via `uv.lock`; `pip-audit`/`uv audit` in CI (non-blocking report).

## 11. Performance expectations (for sizing, not gates)

| Operation | Typical |
|---|---|
| Blender start + `prepare_model` on a 5 k tri model, level 1 | 5–15 s |
| Tripo texture task | 60–180 s |
| `verify` on 50 k tris | < 1 s |
| `bake_textures` 2 k, 16 samples, 6 maps, CPU | 1–4 min (GPU: 20–60 s) |
| `export_rigged` + 5 companion clips | 10–30 s |

Per model end-to-end: ~3–8 minutes; a 30-model library with concurrency 2 and one Blender at a time:
~1.5–3 hours, dominated by Tripo waits and bakes.
