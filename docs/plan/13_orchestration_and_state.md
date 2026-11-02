# 13 — Orchestration, State, Idempotency, Concurrency

## 1. Runner (`pipeline/runner.py`)

```python
class RunOptions(BaseModel):  # NORMATIVE
    models: list[ModelId] | None; tags: list[str]
    from_stage: StageName = "discover"; to_stage: StageName = "handoff"
    force: bool = False; force_stages: list[StageName] = []
    dry_run: bool = False; route_override: Literal["auto","extract","bake"] | None = None
    concurrency: int | None = None; max_credits: float | None = None
    keep_going: bool = True; resume_tasks: bool = True

class PipelineRunner:
    def __init__(self, ctx: PipelineContext, graph: StageGraph): ...
    def run(self, library: Library, options: RunOptions) -> RunReport
```

Algorithm:

```
1 acquire library/.pipeline/run.lock (filelock, timeout 0) → LibraryBusyError
2 run_id = ULID(); open run log; bind logger {run_id}
3 models = select(library, options)                      # unknown id → UsageError before any work
4 validate all selected (05 §8); errors → ValidationError (exit 3)
5 budget = CreditBudget(options.max_credits or cfg.tripo.max_credits_per_run or None)
6 plan = {m: plan_stages(m, options) for m in models}    # which stages run/skip + fingerprints (dry-run prints this)
7 if dry_run: report estimate; return
8 with ThreadPoolExecutor(concurrency) as pool: futures = {pool.submit(run_model, m): m}
   on KeyboardInterrupt: cancel.set(); wait; exit 130
9 aggregate ModelRunSummary → RunReport; write report.json; release lock; return
```

`run_model(m)`:

```
for stage in STAGE_ORDER[from..to]:
    if cancel.is_set(): mark cancelled; break
    rec = state.stages[stage]
    fp = stage.input_fingerprint(m, ctx)
    if not forced and rec.status == SUCCEEDED and rec.fingerprint == fp and outputs_exist(rec): mark SKIPPED; continue
    state.update(stage, RUNNING, fingerprint=fp, run_id, attempts+1); persist
    try: outcome = stage.run(m, ctx)
    except PipelineError as e: state.update(stage, FAILED, error=e); persist; invalidate_downstream(stage); raise/record
    state.update(stage, SUCCEEDED, outputs=outcome.outputs, extra=outcome.extra); persist
    invalidate_downstream(stage) if fingerprint changed vs previous record   # see §2
```

## 2. Idempotency and invalidation rules

- A stage's record is **valid** iff `status == SUCCEEDED`, `fingerprint == current input fingerprint`
  and every path in `outputs` exists.
- When a stage **runs** (not skipped), all downstream stages are set to `PENDING` with
  `extra.invalidated_by = <stage>`. Their fingerprints will differ anyway (they hash upstream outputs),
  but the explicit reset keeps `gp status` truthful even before downstream fingerprints are computed.
- `--force` marks stages in range as forced; `--force-stage S` forces S and (implicitly) downstream.
- `pipeline_version` major bump → whole state invalidated at `discover`.
- Deleting `work/` without touching state → stages with missing outputs become invalid and re-run;
  deleting `state.json` → everything re-runs (Tripo task ids are lost; `gp tripo task` can still
  recover a GLB manually).

## 3. Fingerprint composition (`pipeline/hashing.py`)

```python
def file_sha(path) -> Sha256                 # streamed, 1 MiB chunks, cached per run by (path, mtime, size)
def cfg_dump(section: BaseModel) -> str      # model_dump_json(exclude_none=True, sort keys) — stable
def sha_many(*parts: str) -> Sha256          # sha256 of "\x1f".join(parts)
def blender_jobs_code_sha() -> Sha256        # sha over sorted blender_jobs/**/*.py contents; lazy, cached (19 §2)
```

Per-stage inputs are listed in each stage doc and summarised in `07_…` §3. Fingerprints MUST NOT
include timestamps, run ids, absolute paths (use library-relative), or environment-dependent values
other than config.

## 4. State store (`pipeline/state_store.py`)

```python
class StateStore(Protocol):
    def load(self, model_id) -> ModelState
    def save(self, state: ModelState) -> None
    def update(self, model_id, stage, **changes) -> ModelState     # load-modify-save under lock
class JsonStateStore(StateStore):
    """<model>/.pipeline/state.json ; filelock on state.lock (timeout 30 s) ; write to tmp + os.replace ;
       schema validated on load; corrupt file → backed up as state.corrupt-<ts>.json and recreated (warning)"""
```

Per-stage logs: `<model>/.pipeline/logs/<stage>-<run_id>.log` contain Blender stdout/stderr and the
provider's request/response summaries for that stage.

## 5. `PipelineContext` and `ModelContext` (`pipeline/context.py`)

```python
@dataclass(frozen=True)
class PipelineContext:
    library: Library; options: RunOptions; run_id: RunId
    blender: BlenderRunner; provider: TextureProvider; state: StateStore
    budget: CreditBudget; cancel: CancelToken; clock: Clock; sleeper: Sleeper
    log: structlog.BoundLogger
@dataclass(frozen=True)
class ModelContext:
    resolved: ResolvedModel; paths: ModelPaths              # folder, source, references, work, output, pipeline
    log: structlog.BoundLogger                              # bound with model_id
    # convenience accessors: .config, .primary_fbx, .composed_prompt …
```

Stages receive both; they never construct adapters themselves. `cli/app.py` builds the production
context (`SubprocessBlenderRunner`, `TripoTextureProvider`, `JsonStateStore`); tests build fakes.

## 6. Blender runner (`blender/runner.py`)

```python
class BlenderRunner(Protocol):
    def run(self, req: JobRequest, *, timeout_s: int, log_path: Path, cancel: CancelToken) -> JobResult
class SubprocessBlenderRunner:
    def __init__(self, executable: Path, entry_script: Path, semaphore: BoundedSemaphore, env: Mapping[str,str]): ...
```

Execution: write `job.json` to a temp dir; command
`[exe, "-b", "--factory-startup", "--python-exit-code", "2", "--python", entry_script, "--", "--job", job_json, "--result", result_json]`
with `env` including `GP_SRC_DIR` (so `entry.py` can `sys.path.insert(0, GP_SRC_DIR)`) and
`PYTHONDONTWRITEBYTECODE=1`. Stdout/stderr streamed to `log_path`. Timeout → terminate/kill →
`BlenderTimeoutError`. Non-zero exit without `result.json` → `BlenderCrashError(tail of log)`.
`result.json` parsed → `JobResult`; `ok=False` → `BlenderJobError(result.error)`.

`--factory-startup` guarantees user add-ons/preferences cannot alter behaviour (reproducibility).

## 7. Locks

| Lock | Scope | Purpose |
|---|---|---|
| `library/.pipeline/run.lock` | one `gp run` per library | prevents two runs racing on state |
| `<model>/.pipeline/state.lock` | state file | serialises `update()` across worker threads |
| `BoundedSemaphore(blender_parallelism)` | process | caps concurrent Blender processes |
| `RateLimiter` | process | caps Tripo concurrency/request spacing |

## 8. Cancellation

`CancelToken` wraps a `threading.Event`. Checked: between stages; inside `wait_for_task` each poll;
inside `SubprocessBlenderRunner` (polls the process every 0.5 s). On cancel: running Blender jobs are
killed (their partial outputs are in `work/` and will be overwritten next run), Tripo tasks keep running
server-side and are resumed next time (task id persisted on creation). Exit code 130.

## 9. Dry-run output

```
model        stage      action   reason                                   est. credits
ant_worker   discover   run      fingerprint changed (model.md)
ant_worker   prepare    skip     up to date
ant_worker   texture    run      prompt sha changed                        20
ant_worker   verify..handoff run  upstream invalidated
TOTAL estimated credits: 20   (budget: 200)
```

## 10. Tests

- Runner with fakes: skip/run matrix; downstream invalidation; `--force`/`--force-stage`; `from/to`
  with prerequisite failure; keep-going vs fail-fast; budget exceeded stops texture but completes
  prepare; cancellation mid-run (fake stage that waits on the token); run lock busy; report contents;
  exit codes.
- `JsonStateStore`: atomic write (simulate crash between tmp write and replace), lock contention with two
  threads, corrupt-file recovery.
- `SubprocessBlenderRunner`: uses a **fake Blender executable** (a tiny Python script registered as
  `GP_BLENDER_EXE` in tests) that reads `job.json` and writes scripted `result.json` / exits with chosen
  codes / sleeps for timeout tests. This covers the runner's branches without real Blender.
- Hashing: stability across runs, relative-path invariance, `BLENDER_JOBS_CODE_SHA` changes when a job
  file changes (tmp copy of package).
