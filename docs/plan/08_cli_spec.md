# 08 — CLI Specification (`gp`)

Built with `typer`. Every command accepts `--library PATH` (default: `./library`, or `GP_LIBRARY` env,
or the nearest ancestor directory containing `library.toml`). Output is human-readable by default; `--json`
switches every command to machine-readable JSON on stdout (logs still go to stderr/file).

## 1. Global options

| Flag | Default | Effect |
|---|---|---|
| `--library PATH` | auto | asset library root |
| `--json` | off | JSON output |
| `--log-level LEVEL` | config | DEBUG/INFO/WARNING/ERROR |
| `--blender PATH` | auto | Blender executable override |
| `--no-color` | off | disable rich colours |
| `--version` | | print version, exit 0 |

## 2. Commands

### `gp init [PATH]`
Scaffold a new library: `library.toml` (all defaults, commented), `style/style.md` (template),
`style/references/.gitkeep`, `models/.gitkeep`, `.gitignore`, `generative/.gitkeep`.
Fails with exit 2 if `library.toml` exists (use `--force` to overwrite the toml only).

### `gp new-model <model_id> --source <file.fbx> [--display-name TEXT] [--anim GLOB]... [--copy|--move|--link]`
Create `models/<model_id>/` with `model.md` template (display name, source.primary, prompt skeleton),
copy (default) / move / symlink the source FBX into `source/`, create `references/`. Validates
`model_id`. Exit 2 if the folder exists.

### `gp validate [--model ID]...`
Parse `library.toml`, `style/style.md`, every (or selected) `model.md`; resolve configs; run every rule
in `05_…` §8; check Blender presence/version; check `unity_import_dir`. Prints a table
`model_id | status | problems`. Exit 0 if no errors (warnings allowed), 3 if any error.

### `gp run [--model ID]... [--tag TAG]... [--from-stage S] [--to-stage S] [--force] [--force-stage S]... [--dry-run] [--route auto|extract|bake] [--concurrency N] [--max-credits X] [--keep-going/--fail-fast] [--resume-tasks/--no-resume-tasks]`

Behaviour:

- Selection: all models, or those named / carrying a tag. Unknown IDs → exit 2 before any work.
- Runs `validate` first; any error → exit 3.
- `--dry-run`: resolves everything, computes fingerprints, prints which stages would run per model and
  a **credit estimate** (`TextureProvider.estimate_credits` × models needing `texture`), performs no
  Blender or Tripo work, exit 0.
- `--from-stage/--to-stage`: restrict the stage range (inclusive). Stages before `--from-stage` must
  already be `succeeded` with valid outputs, else `E_STAGE_PREREQ` (exit 3).
- `--force` ignores fingerprints for all stages in range; `--force-stage S` for one stage (and
  downstream).
- `--route` overrides `verify.route` for this run.
- `--max-credits` overrides `tripo.max_credits_per_run`; the runner refuses to start a Tripo task that
  would push the run total over budget (`E_BUDGET`, exit 5), finishing non-Tripo work.
- `--keep-going` (default) continues other models when one fails; `--fail-fast` cancels the rest.
- `--resume-tasks` (default) re-attaches to an in-flight Tripo task recorded in state instead of
  creating a new one.
- Progress: rich progress bars per model (stage name, elapsed) unless `--json`.
- Writes `library/.pipeline/runs/<run_id>/report.json` always.

Exit codes: 0 all succeeded/skipped; 4 some models failed; 5 budget exceeded; 3 validation; 2 usage;
130 cancelled by user; 1 unexpected.

### `gp status [--model ID]...`
Table: `model_id | discover … handoff (✓ ✗ ○ ⟳) | credits | last run | last error`. Reads state only.

### `gp clean [--model ID]... (--work | --output | --state | --all) [--yes]`
Delete the chosen trees. Asks for confirmation unless `--yes`. Never touches `source/`, `references/`,
`model.md`.

### `gp tripo balance`
`GET /v3/account/balance` → prints balance/frozen. Exit 2 without API key.

### `gp tripo task <task_id>`
`GET /v3/tasks/{id}` → prints status/progress/output URLs/credits.

### `gp unity sync [--model ID]... [--dest PATH]`
Copy `output/<id>/` to `<unity_import_dir>/<id>/` atomically (stage into `<id>.tmp-<run_id>` then
rename; delete the previous folder's stale files). Exit 2 if no destination.

### `gp doctor`
Checks: Python version; Blender found + version + `probe` job round-trip (reports Cycles/FBX/glTF
availability); `TRIPO_API_KEY` present + balance call; library valid; free disk ≥ 5 GB; writes
`doctor.json` with `--json`. Exit 0 if all pass, 3 otherwise.

## 3. CLI → config mapping (normative subset)

| Flag | Config key |
|---|---|
| `--blender` | `blender.executable` |
| `--concurrency` | `library.concurrency` |
| `--route` | `verify.route` |
| `--max-credits` | `tripo.max_credits_per_run` |
| `--log-level` | `logging.level` |

No other flag mutates configuration; everything else is a run option recorded in `RunReport.options`.

## 4. Output contract for `--json`

Every command prints exactly one JSON document to stdout on completion:

```json
{"command": "run", "ok": false, "exit_code": 4, "run_id": "01J…", "report_path": "library/.pipeline/runs/01J…/report.json",
 "models": [{"model_id": "ant_worker", "result": "succeeded", "credits": 20.0}, …]}
```

Errors (any exit ≠ 0) include `"error": {"code": "...", "message": "..."}`.

## 5. Testing the CLI

`typer.testing.CliRunner` against a `tests/fixtures/library_minimal` copy in `tmp_path`, with
`FakeBlenderRunner` and `FakeTextureProvider` injected through a test-only entry point
(`grounded_pipeline.cli.app.build_app(ctx_factory=...)`). Every command and every exit code path has a
test; see `15_…` §6.
