# 06 — Data Contracts

All contracts are pydantic v2 models (orchestrator side) with `model_config = ConfigDict(extra="forbid",
frozen=True)` unless stated. Blender-side mirrors are `@dataclass(frozen=True)` + hand-written
`to_dict()/from_dict()` because pydantic is unavailable inside Blender. **Field names and JSON keys are
identical on both sides**; a unit test (`tests/unit/blender/test_protocol_parity.py`) asserts that
pydantic `model_json_schema()` property names equal the dataclass field names for every mirrored type.

Conventions: snake_case keys; timestamps ISO-8601 UTC strings; hashes are lowercase hex SHA-256;
paths inside JSON are **relative to the model folder** and use forward slashes.

---

## 1. Identifiers (`domain/ids.py`)

```python
# NORMATIVE
from typing import Annotated
from pydantic import StringConstraints
ModelId = Annotated[str, StringConstraints(pattern=r"^[a-z0-9][a-z0-9_]{1,63}$")]
RunId   = Annotated[str, StringConstraints(pattern=r"^[0-9A-HJKMNP-TV-Z]{26}$")]  # ULID
Sha256  = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
```

## 2. Library configuration (`domain/library_config.py`)

Mirrors `library.toml` (§3 of `05_…`) one class per section:

```python
# NORMATIVE (abridged: every key in 05 §3 appears with the shown default)
class LibrarySection(BaseModel):
    name: str = "library"
    unity_import_dir: Path | None = None
    concurrency: int = Field(2, ge=1, le=16)
    blender_parallelism: int = Field(1, ge=1, le=8)

class BlenderSection(BaseModel):
    executable: Path | None = None
    allow_unsupported: bool = False
    job_timeout_s: int = Field(1800, ge=60)
    keep_blend_files: bool = True

class GeometrySection(BaseModel):
    strategy: Literal["subdivide"] = "subdivide"      # extension seam (09 §2.9)
    subdivision_levels: int = Field(1, ge=0, le=3)
    subdivision_type: Literal["catmull", "simple"] = "catmull"
    shade_smooth: bool = True
    shapekeys: Literal["fail", "drop", "skip"] = "fail"
    merge_material_slots: bool = True
    keep_only_one_uv_layer: bool = True
    include_unskinned: bool = True
    max_triangles: int = Field(500_000, ge=1_000)

class UvSection(BaseModel):
    mode: Literal["auto", "keep", "unwrap"] = "auto"
    palette_texture_max_px: int = 128
    overlap_ratio_threshold: float = Field(0.20, ge=0, le=1)
    smart_project_angle_deg: float = Field(66.0, gt=0, lt=90)
    island_margin: float = 0.02
    pack_margin: float = 0.01

class TextureSection(BaseModel):
    provider: Literal["tripo"] = "tripo"
    model_version: str = "v3.5-20260815"
    quality: Literal["fast", "standard", "detailed", "extreme"] = "standard"
    pbr: bool = True
    bake: bool = True
    delight: bool = True
    alignment: Literal["original_image", "geometry"] = "original_image"
    prompt_mode: Literal["text", "text+style", "image", "images"] = "text+style"
    seed: int = Field(1337, ge=0, le=2**31 - 1)
    style_image: Path = Path("style/references/primary.png")

class VerifySection(BaseModel):
    route: Literal["auto", "extract", "bake"] = "auto"
    position_quantum: float = 1e-4
    uv_quantum: float = 1e-4
    allow_geometry_change: bool = False

class BakeSection(BaseModel):
    resolution: Literal[512, 1024, 2048, 4096] = 2048
    samples: int = Field(16, ge=1, le=256)
    margin_px: int = Field(16, ge=0, le=64)
    cage_extrusion_ratio: float = 0.01
    max_ray_distance_ratio: float = 0.05
    maps: list[Literal["basecolor","normal","metallic","roughness","occlusion","emission"]] = [...]

class ExportSection(BaseModel):
    fbx_scale_option: Literal["FBX_SCALE_NONE","FBX_SCALE_UNITS","FBX_SCALE_CUSTOM","FBX_SCALE_ALL"] = "FBX_SCALE_ALL"
    roundtrip_companion_animations: bool = True
    embed_textures: bool = False

class UnitySection(BaseModel):
    avatar_mode: Literal["copy_from_source", "create_from_model"] = "copy_from_source"
    source_asset_root: str = ""
    shader: str = "Universal Render Pipeline/Lit"

class TripoSection(BaseModel):
    base_url: HttpUrl = HttpUrl("https://openapi.tripo3d.ai/v3")
    poll_interval_s: float = Field(2.0, ge=0.5)
    poll_timeout_s: int = Field(900, ge=30)
    max_concurrent_tasks: int = Field(4, ge=1, le=16)
    min_request_interval_s: float = 0.25
    max_credits_per_run: float = Field(0, ge=0)
    use_presign_above_mb: int = 60
    credit_table: dict[str, float] = {"fast": 10, "standard": 10, "detailed": 20, "extreme": 30, "style_image": 0}

class LoggingSection(BaseModel):
    level: Literal["DEBUG","INFO","WARNING","ERROR"] = "INFO"
    json_console: bool = False

class LibraryConfig(BaseModel):
    schema_version: Literal[1] = 1
    library: LibrarySection = LibrarySection()
    blender: BlenderSection = BlenderSection()
    geometry: GeometrySection = GeometrySection()
    uv: UvSection = UvSection()
    texture: TextureSection = TextureSection()
    verify: VerifySection = VerifySection()
    bake: BakeSection = BakeSection()
    export: ExportSection = ExportSection()
    unity: UnitySection = UnitySection()
    tripo: TripoSection = TripoSection()
    logging: LoggingSection = LoggingSection()
```

`EffectiveConfig = LibraryConfig` after layering (07). Per-model overrides are **partial** versions of
these sections: one explicit, hand-written `<Section>Overrides` class per section in
`domain/model_spec.py`, every field `Optional` with default `None` (mypy `--strict` cannot type-check a
call expression used inline as an annotation — ADR-0003). `make_partial(SectionModel, name=…)` in
`config/layering.py` generates the expected partial for the parity test
(`tests/unit/domain/test_overrides_parity.py`), which asserts names, optionality and inner types match.

## 3. Model spec (`domain/model_spec.py`)

```python
# NORMATIVE
class SourceSpec(BaseModel):
    primary: Path | None = None                 # relative to model folder
    animations: list[str] = ["source/animations/*.fbx"]   # globs
    license: str | None = None

class GuidanceImages(BaseModel):
    front: Path
    left: Path | None = None
    back: Path | None = None
    right: Path | None = None

class GeometryOverrides(BaseModel):            # every GeometrySection field, Optional, default None
    subdivision_levels: int | None = Field(None, ge=0, le=3)
    ...                                         # (UvOverrides, VerifyOverrides, BakeOverrides, ExportOverrides alike)

class TextureOverrides(BaseModel):              # every TextureSection field, Optional, plus:
    ...
    guidance_images: GuidanceImages | None = None

class PromptSpec(BaseModel):
    include_sections: list[str] = ["base_prompt"]
    template: str = "{base_prompt}\n\n{model_prompt}"

class UnityOverrides(BaseModel):                # every UnitySection field, Optional, plus:
    source_asset_path: str | None = None        # Unity path to vendor FBX (avatar copy)

class ModelFrontMatter(BaseModel):
    schema_version: Literal[1] = 1
    display_name: str
    source: SourceSpec = SourceSpec()
    geometry: GeometryOverrides | None = None
    uv: UvOverrides | None = None
    texture: TextureOverrides | None = None
    verify: VerifyOverrides | None = None
    bake: BakeOverrides | None = None
    export: ExportOverrides | None = None
    unity: UnityOverrides | None = None
    prompt: PromptSpec = PromptSpec()
    tags: list[str] = []

class ModelDocument(BaseModel):
    """Parsed model.md."""
    model_id: ModelId
    folder: Path                                 # absolute
    front: ModelFrontMatter
    sections: dict[str, str]                     # lower-cased heading → body ("prompt", "notes", …)

class ResolvedModel(BaseModel):
    """ModelDocument + effective config + resolved absolute paths. Output of discover."""
    doc: ModelDocument
    config: LibraryConfig                        # effective, after layering
    primary_fbx: Path
    animation_fbx: list[Path]
    style: "StyleDocument"
    composed_prompt: "ComposedPrompt"
```

## 4. Style spec (`domain/style_spec.py`, `domain/prompt.py`)

```python
# NORMATIVE
class StyleFrontMatter(BaseModel):
    schema_version: Literal[1] = 1
    name: str
    primary_reference: Path = Path("references/primary.png")
    negative_terms: list[str] = []

class StyleDocument(BaseModel):
    folder: Path
    front: StyleFrontMatter
    sections: dict[str, str]          # "base_prompt" REQUIRED

class ComposedPrompt(BaseModel):
    text: str                          # final prompt sent to provider
    sources: dict[str, str]            # placeholder → section used (for provenance)
    length: int
    sha256: Sha256
```

## 5. Blender job protocol (`blender/protocol.py` ↔ `blender_jobs/protocol.py`)

```python
# NORMATIVE
PROTOCOL_VERSION = 1
JobName = Literal["probe", "prepare_model", "bake_textures", "export_rigged", "roundtrip_animation"]

class JobRequest(BaseModel):
    protocol_version: Literal[1] = 1
    job: JobName
    job_id: str                        # ULID; appears in logs and result
    model_id: ModelId | None
    params: dict[str, Any]             # one of the *Params models below, dumped
    log_level: Literal["DEBUG","INFO","WARNING"] = "INFO"

class JobResult(BaseModel):
    protocol_version: Literal[1] = 1
    job: JobName
    job_id: str
    ok: bool
    blender_version: str               # "4.5.1"
    duration_s: float
    data: dict[str, Any] = {}          # one of the *Data models below, dumped
    warnings: list[str] = []
    error: JobError | None = None

class JobError(BaseModel):
    code: str                          # see 14_… §3 (E_BJ_*)
    message: str
    detail: dict[str, Any] = {}
    traceback: str | None = None
```

Exit codes from `entry.py`: `0` ok (result written, `ok` may still be false for *expected* failures
such as `E_BJ_SHAPEKEYS`), `2` unhandled exception (result written with traceback if possible),
`3` invalid job file. The runner treats a missing `result.json` as `E_BLENDER_NO_RESULT`.

### 5.1 `probe`

```python
class ProbeParams(BaseModel): pass
class ProbeData(BaseModel):
    blender_version: str
    python_version: str
    has_cycles: bool
    fbx_io_available: bool
    gltf_io_available: bool
```

### 5.2 `prepare_model`

```python
class PrepareModelParams(BaseModel):
    source_fbx: Path                                    # absolute
    out_blend: Path
    out_static_glb: Path
    out_report: Path                                    # prepare_report.json (also returned as data)
    geometry: GeometrySection
    uv: UvSection
    palette_hint_textures: list[Path] = []              # source/textures/* for palette detection

class Bone(BaseModel):
    name: str
    parent: str | None
    head: tuple[float, float, float]                    # armature space, rest pose
    tail: tuple[float, float, float]
    matrix_local: list[list[float]]                     # 4×4 row-major, rest pose, armature space
    transform_path: str                                 # "Armature/Root/Spine" (Unity-style)

class RigFingerprint(BaseModel):
    armature_object: str
    armature_data: str
    bone_count: int
    bones: list[Bone]                                   # depth-first, stable order
    actions: list[str]                                  # action names found in the file
    fingerprint_sha256: Sha256                          # over names+parents+quantized matrices

class UvStats(BaseModel):
    uv_layers: list[str]
    overlap_ratio: float                                # 0..1
    island_count: int
    source_texture_max_px: int | None
    decision: Literal["kept", "unwrapped", "created"]
    reason: str

class MeshStats(BaseModel):
    object_names: list[str]
    vertices_before: int
    faces_before: int
    vertices_after: int
    faces_after: int
    triangles_after: int
    shapekey_meshes: list[str]
    shapekeys_policy_applied: Literal["none", "dropped", "skipped"]
    material_slots_before: int
    material_slots_after: int

class PrepareModelData(BaseModel):
    rig: RigFingerprint
    uv: UvStats
    mesh: MeshStats
    static_glb_sha256: Sha256
    static_glb_triangles: int
    static_glb_unique_positions: int
    bbox_diagonal: float
```

### 5.3 `bake_textures`

```python
class BakeTexturesParams(BaseModel):
    prepared_blend: Path
    source_glb: Path                                    # Tripo result.glb
    out_dir: Path                                       # work/textures/raw
    bake: BakeSection
    bbox_diagonal: float                                # from PrepareModelData

class BakeTexturesData(BaseModel):
    maps: dict[str, Path]                               # "basecolor" → raw/basecolor.png …
    resolution: int
    samples: int
    bake_seconds: dict[str, float]
```

### 5.4 `export_rigged`

```python
class ExportRiggedParams(BaseModel):
    prepared_blend: Path
    textures_dir: Path | None                           # work/textures/urp ; None → no material binding
    out_fbx: Path
    out_rig_fingerprint: Path
    export: ExportSection
    material_name: str                                  # "<model_id>_mat"

class ExportRiggedData(BaseModel):
    rig: RigFingerprint
    fbx_sha256: Sha256
    actions_exported: list[str]
```

### 5.5 `roundtrip_animation`

```python
class RoundtripAnimationParams(BaseModel):
    source_fbx: Path
    out_fbx: Path
    export: ExportSection
    expected_bone_names: list[str]                      # from source rig fingerprint

class RoundtripAnimationData(BaseModel):
    rig: RigFingerprint
    actions_exported: list[str]
    missing_bones: list[str]                            # expected but absent
    extra_bones: list[str]
```

## 6. Surface fingerprint & verification (`domain/fingerprint.py`, `domain/reports.py`)

```python
# NORMATIVE
class SurfaceFingerprint(BaseModel):
    triangle_count: int
    unique_position_count: int
    position_set_sha256: Sha256         # sorted unique quantized (x,y,z)
    surface_sha256: Sha256              # sorted unique quantized (x,y,z,u,v)
    has_uv: bool
    primitive_count: int
    material_count: int
    quantum_position: float
    quantum_uv: float

class VerifyReport(BaseModel):
    ours: SurfaceFingerprint
    theirs: SurfaceFingerprint
    topology_match: bool                # triangle_count & unique_position_count equal
    position_match: bool                # position_set_sha256 equal
    uv_match: bool                      # surface_sha256 equal
    route: Literal["extract", "bake"]
    reason: str
    forced: bool                        # route forced by config
```

Decision function (pure, `gltf/verify.py`):

```python
def decide_route(ours, theirs, cfg: VerifySection) -> VerifyReport:  # NORMATIVE
    if cfg.route != "auto": route = cfg.route; forced = True
    elif uv_match and position_match: route = "extract"
    elif position_match or topology_match: route = "bake"
    elif cfg.allow_geometry_change: route = "bake"
    else: raise GeometryChangedError(...)
```

## 7. Texture provider contract (`texturing/provider.py`)

```python
# NORMATIVE
class GuidanceMode(StrEnum):
    TEXT = "text"; TEXT_STYLE = "text+style"; IMAGE = "image"; IMAGES = "images"

class TextureRequest(BaseModel):
    model_id: ModelId
    mesh_path: Path                      # static.glb
    mode: GuidanceMode
    prompt: str | None
    style_image: Path | None
    guidance_images: dict[str, Path]     # keys front/left/back/right
    model_version: str
    quality: str
    pbr: bool
    bake: bool
    delight: bool
    alignment: str
    seed: int
    request_sha256: Sha256               # over all of the above + mesh sha → cache key

class TextureResult(BaseModel):
    provider: Literal["tripo"]
    task_id: str
    result_glb: Path
    preview_png: Path | None
    credits_consumed: float
    started_at: str; finished_at: str
    raw_output: dict[str, Any]           # provider output verbatim (provenance)

class TextureProvider(Protocol):
    def estimate_credits(self, req: TextureRequest) -> float: ...
    def texture(self, req: TextureRequest, *, resume_task_id: str | None, cancel: CancelToken) -> TextureResult: ...
```

## 8. Tripo DTOs (`tripo/models.py`) — see `appendix_A` for the API

```python
# NORMATIVE
class ApiEnvelope(BaseModel, Generic[T]):
    code: int
    data: T | None = None
    status: str | None = None
    message: str | None = None
    suggestion: str | None = None
    request_id: str | None = None

class FileUploadData(BaseModel): file_token: str
class PresignData(BaseModel): presigned_url: HttpUrl; file_token: str; expires_in: int

class TripoFileRef(BaseModel):
    """Reference to an uploaded file or a public URL. Exactly one of the two is set."""
    file_token: str | None = None
    url: str | None = None

ViewName = Literal["front", "left", "back", "right"]

class TexturePrompt(BaseModel):
    text: str | None = None
    style_image: TripoFileRef | None = None      # object form {file_token}|{url}; only together with `text`
    image: str | None = None                     # file_token or URL (bare string accepted by the API)
    images: list[dict[ViewName, str]] | None = None
    # view-key array form: [{"front": tok}, {"left": tok}] — `front` required, ≥ 2 views, missing views omitted.
    # `text` / `image` / `images` are mutually exclusive (validator); `style_image` requires `text`.

class TextureTaskRequest(BaseModel):
    input: str                           # file_token | task_id | URL
    model: str
    texture_prompt: TexturePrompt | None = None
    pbr: bool = True
    texture_seed: int | None = None
    texture_alignment: Literal["original_image","geometry"] = "original_image"
    texture_quality: Literal["fast","standard","detailed","extreme"] = "standard"
    bake: bool = True
    delight: bool = True
    compress: Literal["geometry"] | None = None

class TaskCreateData(BaseModel): task_id: str
class TaskOutput(BaseModel):
    model_url: HttpUrl | None = None               # signed CDN URL; EXPIRES ~5 min after retrieval → download immediately
    rendered_image_url: HttpUrl | None = None
    normalize_scale: float | None = None           # present when input was a file_token/URL: Tripo rescaled the mesh
    normalize_translation: tuple[float, float, float] | None = None
    model_config = ConfigDict(extra="allow")       # forward compatible
class TaskDetail(BaseModel):
    task_id: str; type: str
    status: Literal["queued","running","success","failed","cancelled","banned","expired","unknown"]
    # terminal failures: failed / cancelled / banned / expired ; "unknown" → keep polling (bounded)
    progress: int = 0
    error_code: int | None = None
    error_message: str | None = None
    output: TaskOutput | None = None
    credits_consumed: float | None = None
    created_at: str | None = None; completed_at: str | None = None
class BalanceData(BaseModel): balance: float; frozen: float
```

## 9. Pipeline state (`domain/state.py`)

```python
# NORMATIVE
StageName = Literal["discover","prepare","texture","verify","apply","export","handoff"]
STAGE_ORDER: tuple[StageName, ...] = ("discover","prepare","texture","verify","apply","export","handoff")

class StageStatus(StrEnum):
    PENDING="pending"; RUNNING="running"; SUCCEEDED="succeeded"; FAILED="failed"; SKIPPED="skipped"

class StageRecord(BaseModel):
    status: StageStatus = StageStatus.PENDING
    fingerprint: Sha256 | None = None
    outputs: list[str] = []              # relative paths that must exist for the record to be valid
    started_at: str | None = None
    finished_at: str | None = None
    duration_s: float | None = None
    run_id: RunId | None = None
    error_code: str | None = None
    error_message: str | None = None
    attempts: int = 0
    extra: dict[str, Any] = {}           # e.g. texture: {"task_id":..., "credits":...}

class ModelState(BaseModel):
    schema_version: Literal[1] = 1
    model_id: ModelId
    pipeline_version: str
    stages: dict[StageName, StageRecord]
    last_run_id: RunId | None = None
    credits_total: float = 0.0
    model_config = ConfigDict(frozen=False)
```

`StageOutcome` returned by `Stage.run`:

```python
class StageOutcome(BaseModel):
    status: Literal["succeeded", "skipped"]
    outputs: list[str]
    extra: dict[str, Any] = {}
    warnings: list[str] = []
```

## 10. Manifest (`domain/manifest.py`) — JsonUtility-compatible

Unity's `JsonUtility` cannot deserialize dictionaries or nullable primitives, so the manifest uses
**arrays of objects** and empty strings instead of `null`.

```python
# NORMATIVE
class ManifestTexture(BaseModel):
    slot: str          # "_BaseMap" | "_BumpMap" | "_MetallicGlossMap" | "_OcclusionMap" | "_EmissionMap"
    file: str          # relative to manifest: "textures/<id>_basecolor.png"
    srgb: bool
    kind: str          # "default" | "normal"
class ManifestFloat(BaseModel): name: str; value: float
class ManifestColor(BaseModel): name: str; r: float; g: float; b: float; a: float
class ManifestMaterial(BaseModel):
    name: str
    shader: str
    renderers: list[str]              # mesh object names to assign this material to; [] = all
    textures: list[ManifestTexture]
    floats: list[ManifestFloat]
    colors: list[ManifestColor]
    keywords: list[str]
class ManifestRig(BaseModel):
    animation_type: str               # "Generic"
    avatar_mode: str                  # "copy_from_source" | "create_from_model"
    source_avatar_asset_path: str     # "" when unknown
    root_bone: str
class ManifestAnimation(BaseModel): file: str; clip_name: str
class ManifestProvenance(BaseModel):
    source_fbx_sha256: str
    source_license: str
    subdivision_levels: int
    subdivision_type: str
    uv_decision: str
    texture_provider: str
    texture_task_id: str
    texture_route: str
    texture_seed: int
    prompt_sha256: str
    credits_consumed: float
    rig_fidelity: str                 # "pass" | "pass_with_warnings"
class Manifest(BaseModel):
    schema_version: int = 1
    pipeline_version: str
    generated_at: str
    model_id: str
    display_name: str
    fbx: str
    rig: ManifestRig
    materials: list[ManifestMaterial]
    animations: list[ManifestAnimation]
    provenance: ManifestProvenance
```

The C# mirror (`PipelineManifest.cs`) MUST have identical field names (C# `[Serializable]` classes with
public fields). `tests/unit/unity/test_manifest_cs_parity.py` parses the C# file with a regex and asserts
field-name parity with the pydantic schema.

## 11. Run report (`domain/reports.py`)

```python
class ModelRunSummary(BaseModel):
    model_id: ModelId
    result: Literal["succeeded","failed","skipped","cancelled"]
    stages: dict[StageName, StageStatus]
    credits: float
    duration_s: float
    error_code: str | None = None
    error_message: str | None = None
class RunReport(BaseModel):
    run_id: RunId
    started_at: str; finished_at: str
    library: Path
    options: dict[str, Any]
    models: list[ModelRunSummary]
    credits_total: float
    exit_code: int
```

## 12. Error hierarchy — see `14_errors_and_observability.md` §2 (normative there).
