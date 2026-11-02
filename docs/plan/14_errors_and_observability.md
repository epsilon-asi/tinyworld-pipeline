# 14 — Errors, Logging, Observability

## 1. Principles

- **Expected failures are typed.** Everything the pipeline can anticipate is a `PipelineError` subclass
  with a stable `code` (used in state, reports, JSON output and tests). Unexpected exceptions propagate
  as `code = "E_UNEXPECTED"` with traceback in the run log, never swallowed.
- **Fail the model, not the run.** One model's error never stops others unless `--fail-fast`.
- **Every log line is attributable.** `run_id`, `model_id`, `stage`, and (where relevant) `job_id` /
  `task_id` are bound to the logger, not repeated by hand.
- **Secrets never appear anywhere.** The API key is a `SecretStr`; log processors redact
  `Authorization` headers and any string matching the key.

## 2. Exception hierarchy (`domain/errors.py`) — NORMATIVE

```python
class PipelineError(Exception):
    code: str = "E_PIPELINE"; exit_code: int = 1
    def __init__(self, message: str, *, detail: dict[str, Any] | None = None, cause: BaseException | None = None)

# usage / configuration
class UsageError(PipelineError):            code="E_USAGE";            exit_code=2
class ConfigError(PipelineError):           code="E_CONFIG";           exit_code=2
class ValidationError(PipelineError):       code="E_VALIDATION";       exit_code=3   # carries list[Problem]
class StagePrereqError(PipelineError):      code="E_STAGE_PREREQ";     exit_code=3
class LibraryBusyError(PipelineError):      code="E_LIBRARY_BUSY";     exit_code=2
# budget
class BudgetExceededError(PipelineError):   code="E_BUDGET";           exit_code=5
# blender
class BlenderNotFoundError(ConfigError):    code="E_BLENDER_NOT_FOUND"
class BlenderVersionError(ConfigError):     code="E_BLENDER_VERSION"
class BlenderTimeoutError(PipelineError):   code="E_BLENDER_TIMEOUT"
class BlenderCrashError(PipelineError):     code="E_BLENDER_CRASH"
class BlenderNoResultError(PipelineError):  code="E_BLENDER_NO_RESULT"
class BlenderJobError(PipelineError):       code = result.error.code   # "E_BJ_*" passthrough
class ProtocolDriftError(PipelineError):    code="E_PROTOCOL"          # JobResult.data failed validation
# tripo
class TripoError(PipelineError):            code="E_TRIPO"
class TripoAuthError(TripoError):           code="E_TRIPO_AUTH";       exit_code=2
class TripoApiError(TripoError):            code="E_TRIPO_API"         # non-zero envelope code; detail has api code/suggestion/request_id
class TripoRateLimited(TripoError):         code="E_TRIPO_RATE_LIMIT"
class TripoServerError(TripoError):         code="E_TRIPO_SERVER"
class TripoTransientError(TripoError):      code="E_TRIPO_TRANSIENT"
class TripoTaskFailed(TripoError):          code="E_TRIPO_TASK_FAILED"
class TripoTaskTimeout(TripoError):         code="E_TRIPO_TASK_TIMEOUT"
class TripoAmbiguousCreateError(TripoError):code="E_TRIPO_AMBIGUOUS_CREATE"
class TripoDownloadError(TripoError):       code="E_TRIPO_DOWNLOAD"
# verification / apply / export
class VerifySelfMismatchError(PipelineError): code="E_VERIFY_SELF_MISMATCH"
class GeometryChangedError(PipelineError):    code="E_GEOMETRY_CHANGED"
class UnsupportedGlbError(PipelineError):     code="E_GLB_UNSUPPORTED"
class MultiMaterialResultError(PipelineError):code="E_MULTI_MATERIAL"
class TexturePackError(PipelineError):        code="E_TEXTURE_PACK"
class RigFidelityError(PipelineError):        code="E_RIG_FIDELITY"    # detail: RigFidelityReport
class AnimNameCollisionError(PipelineError):  code="E_ANIM_NAME_COLLISION"
class HandoffError(PipelineError):            code="E_HANDOFF"
class CancelledError(PipelineError):          code="E_CANCELLED";       exit_code=130
```

Blender-side codes (`E_BJ_*`) are defined in `blender_jobs/protocol.py` as string constants and listed
in `09`, `11`, `12`. `tests/unit/test_error_codes.py` asserts every code is unique and documented
(scans these docs for the code string).

## 3. Problem objects for validation

```python
class Problem(BaseModel):
    code: str                 # E_* or W_*
    severity: Literal["error","warning"]
    model_id: ModelId | None
    path: str | None          # file or front-matter key path, e.g. "model.md:texture.quality"
    message: str
    hint: str | None          # the one-line fix
```

`ValidationError.problems: list[Problem]`; CLI renders them grouped by model with hints.

## 4. Logging (`observability/logging.py`)

- `structlog` with processors: `merge_contextvars`, `add_log_level`, `TimeStamper(fmt="iso", utc=True)`,
  `redact_secrets` (custom), `StackInfoRenderer`, `format_exc_info`, then **two sinks**:
  console (`ConsoleRenderer` colours unless `--no-color`/`--json`; level from config) and run file
  `library/.pipeline/runs/<run_id>/run.log` as JSON lines (always DEBUG).
- Stage logs additionally tee to `<model>/.pipeline/logs/<stage>-<run_id>.log`.
- Bound keys: `run_id`, `model_id`, `stage`, `job_id`, `task_id`, `attempt`.
- Event naming: `snake_case` verbs — `stage.start`, `stage.skip`, `stage.done`, `stage.fail`,
  `blender.spawn`, `blender.exit`, `tripo.request`, `tripo.response`, `tripo.poll`, `tripo.credits`,
  `verify.route`, `rfc.result`, `handoff.sync`.
- Timing: every `stage.done` carries `duration_s`; Blender jobs carry `blender_version`, `duration_s`.
- Third-party loggers (`httpx`, `httpcore`, `PIL`) set to WARNING; `httpx` request lines are replaced by
  our `tripo.request` events (method, path, size, elapsed) which never log bodies above DEBUG and never
  log image/model payloads.

## 5. Console UX (`cli/render.py`)

- `rich` progress: one task row per model (`model_id · stage · elapsed`), spinner while running, ✓/✗ on
  completion; overall bar `models done / total`.
- Warnings are collected and printed once at the end under “Warnings”, grouped by model.
- On failure, the console prints: code, message, hint (if any), and the path to the stage log.
- `--json` disables rich output entirely.

## 6. Run report (`domain/reports.py` → `library/.pipeline/runs/<run_id>/report.json`)

Contains `RunReport` (06 §11) plus `environment` (python, blender version/path, pipeline version, OS)
and `problems` (warnings surfaced during the run). `gp status` reads state only; `report.json` is the
audit trail per run.

## 7. Metrics worth tracking (recorded in `extra` of stage records, summarised in the report)

| Metric | Where |
|---|---|
| triangles before/after, uv decision | prepare |
| credits, task duration, Tripo task id, model version | texture |
| route, position/uv match | verify |
| bake seconds per map, resolution | apply |
| rig fidelity result + warnings | export |
| output bytes, synced to Unity? | handoff |

## 8. Debuggability guarantees

- Every stage leaves a `*_report.json`; every Blender job leaves `job.json` + `result.json` + log in
  `<model>/.pipeline/logs/` (last 5 kept per stage; older pruned).
- `GP_KEEP_TEMP=1` keeps the temp job directories.
- `gp run --model X --from-stage apply --force-stage apply` re-runs a single stage cheaply.
- `prepared.blend` can be opened in Blender's UI to inspect the exact mesh that was sent/baked.
