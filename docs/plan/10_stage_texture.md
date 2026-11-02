# 10 — Stage `texture` (Tripo client + provider)

**Purpose:** send `work/static.glb` to Tripo with our composed prompt / style image / guidance images
and obtain `work/tripo/result.glb`. The rigged file is never involved.

## 1. Stage (`pipeline/stages/texture.py`)

```python
class TextureStage:  # NORMATIVE
    name = "texture"
    def input_fingerprint(self, m, ctx):
        return sha_many(file_sha(work/"static.glb"), m.composed_prompt.sha256,
                        file_sha(style_image) if used, *[file_sha(p) for p in guidance_images],
                        cfg_dump(m.config.texture), "provider=tripo")
    def run(self, m, ctx):
        req = build_texture_request(m)                               # pure
        budget.reserve(ctx.provider.estimate_credits(req))           # raises BudgetExceededError
        resume = state.stages["texture"].extra.get("task_id") if options.resume_tasks else None
        res = ctx.provider.texture(req, resume_task_id=resume, cancel=ctx.cancel)
        budget.commit(res.credits_consumed)
        write_json(work/"tripo/request.json", req); write_json(work/"tripo/task.json", res)
        return StageOutcome("succeeded", outputs=["work/tripo/result.glb","work/tripo/task.json","work/tripo/request.json"],
                            extra={"task_id": res.task_id, "credits": res.credits_consumed})
```

**Resume semantics:** before calling the provider the stage persists `extra.task_id = <pending>`
as soon as the task is created (provider callback `on_task_created`). If the process dies mid-poll, the
next run (same fingerprint) re-attaches to that task instead of paying again. If the fingerprint changed,
the old task is abandoned (logged, not cancelled — Tripo has no cancel endpoint in the documented set).

## 2. `TripoTextureProvider` (`texturing/tripo_provider.py`)

```
texture(req):
  1 mesh_token = client.upload(req.mesh_path)                 # multipart ≤60 MB, presign above
  2 prompt = build_texture_prompt(req, client)                # uploads style/guidance images → tokens
  3 task_id = resume_task_id or client.create_texture_task(TextureTaskRequest(input=mesh_token, ...)).task_id
     on_task_created(task_id)
  4 detail = client.wait_for_task(task_id, poll_interval, timeout, cancel)
  5 glb = client.download(detail.output.model_url, work/"tripo/result.glb")
     preview = client.download(detail.output.rendered_image_url, work/"tripo/preview.png") if present
  6 return TextureResult(task_id, glb, preview, detail.credits_consumed or estimate, raw_output=detail.model_dump())
```

Guidance mapping (`build_texture_prompt`):

| `GuidanceMode` | `TexturePrompt` |
|---|---|
| `text` | `{text: prompt}` |
| `text+style` | `{text: prompt, style_image: {file_token: <token of style image>}}` |
| `image` | `{image: <token of guidance_images["front"]>}` |
| `images` | `{images: [{front: tok}, {left: tok}, …]}` — view-key array, `front` required, ≥ 2 views, missing views omitted |

Spike S1 decides whether `image`/`images` is accepted for uploaded meshes (R-02). Until then the
provider accepts those modes but `gp validate` emits `W_GUIDANCE_UNVERIFIED`.

`estimate_credits(req)`: table in `tripo/budget.py` — official pricing 2026-11-02: `fast: 10,
standard: 10, detailed: 20, extreme: 30` (no documented surcharge for a style image; `style_image: 0`).
Values are configurable (`[tripo.credit_table]`) because Tripo pricing changes; the real
`credits_consumed` from the task detail is what gets recorded. 1 credit ≈ US$0.01.

## 3. `TripoClient` (`tripo/client.py`)

```python
class TripoClient:  # NORMATIVE public surface
    def __init__(self, api_key: SecretStr, base_url: str, *, transport: httpx.BaseTransport | None = None,
                 sleeper: Sleeper, clock: Clock, rate_limiter: RateLimiter, timeout_s: float = 60): ...
    def upload(self, path: Path, *, presign_above_mb: int) -> str                      # file_token
    def create_texture_task(self, req: TextureTaskRequest) -> TaskCreateData
    def get_task(self, task_id: str) -> TaskDetail
    def wait_for_task(self, task_id: str, *, interval_s: float, timeout_s: float, cancel: CancelToken,
                      on_progress: Callable[[TaskDetail], None] | None = None) -> TaskDetail
    def download(self, url: str, dest: Path) -> Path                                    # streamed, sha256 logged
    def balance(self) -> BalanceData
```

Implementation rules:

- One `httpx.Client` per `TripoClient`, `headers={"Authorization": f"Bearer {key}"}`, `http2=False`
  (keeps the dependency set small), `timeout=httpx.Timeout(60, connect=10)`.
- **Upload:** `POST /files` multipart field `file`; for files > `presign_above_mb`: `POST /files/presign
  {"format":"glb"}` → `PUT presigned_url` (no auth header, `Content-Type: application/octet-stream`)
  → use `file_token`. Images (`style_image`, guidance) go through `POST /files` (≤ 20 MB; validated
  earlier).
- **Create:** `POST /models/texture` JSON body = `TextureTaskRequest.model_dump(exclude_none=True)`.
- **Envelope handling:** parse `ApiEnvelope`; `code != 0` → `TripoApiError(code, message, suggestion,
  request_id)`. HTTP 401/403 → `TripoAuthError`; 429 → `TripoRateLimited(retry_after)`; 5xx →
  `TripoServerError`; network/timeout → `TripoTransientError`.
- **Retries (tenacity):** transient + 5xx + 429 → exponential backoff `1s × 2^n`, max 6 attempts,
  jitter, honour `Retry-After`. Never retry a `create_texture_task` POST after a 2xx or an ambiguous
  timeout *without* first checking for an existing task id (POST is not idempotent → would bill twice).
  Policy: on ambiguous failure of create, raise `TripoAmbiguousCreateError` and let the user resolve via
  `gp tripo task`/`POST /tasks/list` lookup; the stage records the request sha so a human can match it.
- **Polling:** `interval_s` (default 2 s per docs), grows ×1.5 after 60 s up to 10 s; `timeout_s`
  default 900. Terminal statuses: `success` → return; `failed`/`cancelled`/`banned`/`expired` →
  `TripoTaskFailed(status, error_code, error_message)`; `unknown` → keep polling (counts toward
  timeout). `cancel.is_set()` → `CancelledError` (task id already persisted for resume). Docs ask for
  ≤ 1 request/s per task; the rate limiter enforces `min_request_interval_s`.
- **Download:** streaming to `dest.tmp` then `os.replace`; verify `Content-Length` when present; compute
  sha256; refuse HTML responses (`Content-Type` check) with `TripoDownloadError`. **Output URLs expire
  about 5 minutes after the task detail is fetched**: `wait_for_task` returns and the provider downloads
  immediately; on resume of an already-successful task, `get_task` is called again to obtain a fresh URL
  (HTTP 403 on download → one re-fetch of the task detail, then `TripoDownloadError`).
- **Redaction:** `__repr__`, logs and exception messages never include the key; `respx` tests assert
  the header value is not in any captured log line.
- **Recording:** when `GP_TRIPO__RECORD_DIR` is set, every request/response pair (minus auth headers)
  is written as JSON — used once to produce `tests/fixtures/tripo/*.json`.

## 4. Rate limiting (`tripo/polling.py`, `RateLimiter`)

Token bucket: capacity `max_concurrent_tasks`, refill by task completion; plus a minimum spacing
between any two requests (`min_request_interval_s`). Shared across threads via a `threading.Lock`.
Deterministic in tests via injected `Clock`.

## 5. Budget (`tripo/budget.py`)

```python
class CreditBudget:
    def __init__(self, limit: float | None): ...
    def reserve(self, estimate: float) -> Reservation      # raises BudgetExceededError
    def commit(self, actual: float) -> None                 # replaces reservation with actual
    def release(self, reservation) -> None                  # on failure before billing
    @property
    def spent(self) -> float
```

Thread-safe. The runner creates one budget per run; `RunReport.credits_total` = `spent`.

## 6. Prompt builder (`texturing/prompt_builder.py`)

```python
def compose_prompt(style: StyleDocument, doc: ModelDocument, spec: PromptSpec) -> ComposedPrompt  # pure
```

- Placeholders available: every style section (`base_prompt`, `material_notes`, `palette`, …),
  `model_prompt`, `display_name`, `tags` (comma-joined).
- Sections not listed in `include_sections` resolve to `""` so templates stay stable.
- Whitespace normalisation: collapse 3+ newlines to 2, strip; collapse runs of spaces.
- Length: > 1 000 chars → `PromptTooLongError`; > 600 → warning (long prompts dilute style guidance).
- `sources` records which section fed which placeholder for provenance.

## 7. Stage outputs

| File | Content |
|---|---|
| `work/tripo/request.json` | `TextureRequest` (paths relative, no tokens) + the exact `TextureTaskRequest` body sent (tokens included — tokens are not secrets) |
| `work/tripo/task.json` | `TextureResult` incl. raw task detail |
| `work/tripo/result.glb` | downloaded model |
| `work/tripo/preview.png` | Tripo render (if provided) — handy for eyeballing before bake |

## 8. Tests

- `TripoClient` with `respx`: upload small/large (presign path), envelope errors, 401, 429 with
  `Retry-After`, 5xx retry then success, timeout → transient, `wait_for_task` through
  queued→running→success, failed/banned, cancel mid-poll, download sha + HTML refusal, key redaction.
- `TripoTextureProvider` with a mocked client: all four guidance modes; resume path (no create call);
  `on_task_created` fires before polling; credit fallback to estimate when `credits_consumed` is null.
- `TextureStage`: fingerprint changes on prompt/style/seed/quality but not on `poll_interval_s`;
  budget reserve/commit/release on failure; outputs written atomically.
- `compose_prompt`: template placeholders, section inclusion, whitespace, length errors, sha stability.
- Live smoke (`@tripo_live`, manual): one `rigged_cube` texture run; asserts GLB downloads and
  `credits_consumed > 0`. Used by Spike S1.
