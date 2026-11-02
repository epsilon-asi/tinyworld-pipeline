"""``TextureStage`` (plan 10 §1, §8): fingerprint matrix, budget flow, outputs, resume."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pytest
import structlog

from fakes.fake_texture_provider import FakeTextureProvider
from fakes.glb_builder import cube, encode_image, solid_image, write_glb
from grounded_pipeline.domain.errors import (
    BudgetExceededError,
    StagePrereqError,
    TripoTaskFailedError,
)
from grounded_pipeline.domain.library_config import LibraryConfig, TextureSection, TripoSection
from grounded_pipeline.domain.model_spec import (
    ModelDocument,
    ModelFrontMatter,
    ResolvedModel,
    TextureOverrides,
)
from grounded_pipeline.domain.ports import CancelToken
from grounded_pipeline.domain.prompt import ComposedPrompt
from grounded_pipeline.domain.state import ModelState, StageName, StageRecord
from grounded_pipeline.domain.style_spec import StyleDocument, StyleFrontMatter
from grounded_pipeline.library.paths import LibraryPaths, ModelPaths
from grounded_pipeline.pipeline.context import ModelContext, PipelineContext
from grounded_pipeline.pipeline.hashing import cfg_dump, file_sha, sha_many
from grounded_pipeline.pipeline.options import RunOptions
from grounded_pipeline.pipeline.stage import Stage
from grounded_pipeline.pipeline.stages.texture import (
    EXTRA_REQUEST_SHA,
    EXTRA_TASK_ID,
    TextureStage,
)
from grounded_pipeline.texturing.request_builder import build_texture_request
from grounded_pipeline.tripo.budget import CreditBudget

PROMPT = "a red ant, stylised"
RUN_ID = "01ARZ3NDEKTSV4RRFFQ69G5FAV"


# --------------------------------------------------------------------------- test doubles
class MemoryStateStore:
    """In-memory ``StateStore`` recording every ``update`` call."""

    def __init__(self) -> None:
        self.states: dict[str, ModelState] = {}
        self.update_calls: list[tuple[str, str, dict[str, Any]]] = []

    def load(self, model_id: str) -> ModelState:
        if model_id not in self.states:
            self.states[model_id] = ModelState(model_id=model_id, pipeline_version="test")
        return self.states[model_id]

    def save(self, state: ModelState) -> None:
        self.states[state.model_id] = state

    def update(self, model_id: str, stage: StageName, **changes: Any) -> ModelState:
        state = self.load(model_id)
        record = state.stages.get(stage, StageRecord())
        state.stages[stage] = record.model_copy(update=changes)
        self.update_calls.append((model_id, stage, dict(changes)))
        return state


@dataclass(frozen=True)
class FakeLibrary:
    paths: LibraryPaths


# --------------------------------------------------------------------------- library builder
def _png(path: Path, rgba: tuple[int, int, int, int] = (10, 200, 10, 255)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encode_image(solid_image(rgba)))
    return path


def make_model(
    root: Path,
    *,
    model_id: str = "ant",
    prompt: str = PROMPT,
    texture: dict[str, Any] | None = None,
    tripo: dict[str, Any] | None = None,
    front_texture: TextureOverrides | None = None,
    mesh: bytes | None = None,
    with_static: bool = True,
) -> ModelContext:
    _png(root / "style" / "references" / "primary.png")
    folder = root / "models" / model_id
    folder.mkdir(parents=True, exist_ok=True)
    if with_static:
        write_glb(folder / "work" / "static.glb", cube() if mesh is None else mesh)
    config = LibraryConfig(
        texture=TextureSection(**(texture or {})), tripo=TripoSection(**(tripo or {}))
    )
    doc = ModelDocument(
        model_id=model_id,
        folder=folder,
        front=ModelFrontMatter(display_name="Ant", texture=front_texture),
        sections={"prompt": prompt},
    )
    resolved = ResolvedModel(
        doc=doc,
        config=config,
        primary_fbx=folder / "source" / "ant.fbx",
        animation_fbx=[],
        style=StyleDocument(
            folder=root / "style", front=StyleFrontMatter(name="s"), sections={"base_prompt": "x"}
        ),
        composed_prompt=ComposedPrompt(
            text=prompt,
            sources={},
            length=len(prompt),
            sha256=hashlib.sha256(prompt.encode()).hexdigest(),
        ),
    )
    return ModelContext(
        resolved=resolved,
        paths=ModelPaths.for_model(root / "models", model_id),
        log=structlog.get_logger("test").bind(model_id=model_id),
    )


def make_ctx(
    root: Path,
    provider: FakeTextureProvider | None = None,
    *,
    limit: float | None = 100.0,
    resume_tasks: bool = True,
) -> PipelineContext:
    """A real ``PipelineContext``; adapters the texture stage never touches are inert dummies."""
    return PipelineContext(
        library=cast("Any", FakeLibrary(paths=LibraryPaths.from_root(root))),
        options=RunOptions(resume_tasks=resume_tasks),
        run_id=RUN_ID,
        blender=cast("Any", object()),
        provider=provider or FakeTextureProvider(),
        state=MemoryStateStore(),
        budget=CreditBudget(limit),
        cancel=CancelToken(),
        clock=cast("Any", object()),
        sleeper=cast("Any", object()),
        log=structlog.get_logger("test"),
        pipeline_version="test",
        env={},
        cli_overrides={},
    )


def _store(ctx: PipelineContext) -> MemoryStateStore:
    return cast("MemoryStateStore", ctx.state)


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return tmp_path / "library"


# --------------------------------------------------------------------------- typing contracts
def test_stage_satisfies_stage_protocol(root: Path) -> None:
    stage: Stage = TextureStage()
    assert stage.name == "texture"
    assert stage.input_fingerprint(make_model(root), make_ctx(root))


# --------------------------------------------------------------------------- fingerprint
def test_fingerprint_matches_plan_formula(root: Path) -> None:
    model = make_model(root)
    expected = sha_many(
        file_sha(model.paths.work / "static.glb"),
        model.composed_prompt.sha256,
        file_sha(root / "style" / "references" / "primary.png"),
        cfg_dump(model.config.texture),
        "provider=tripo",
    )
    assert TextureStage().input_fingerprint(model, make_ctx(root)) == expected


def test_fingerprint_without_style_image_in_text_mode(root: Path) -> None:
    model = make_model(root, texture={"prompt_mode": "text"})
    expected = sha_many(
        file_sha(model.paths.work / "static.glb"),
        model.composed_prompt.sha256,
        cfg_dump(model.config.texture),
        "provider=tripo",
    )
    assert TextureStage().input_fingerprint(model, make_ctx(root)) == expected


@pytest.mark.parametrize(
    ("change", "differs"),
    [
        ({"prompt": "a blue ant"}, True),
        ({"texture": {"seed": 7}}, True),
        ({"texture": {"quality": "extreme"}}, True),
        ({"texture": {"prompt_mode": "text"}}, True),
        ({"texture": {"bake": False}}, True),
        ({"mesh": cube(scale=3.0)}, True),
        ({"tripo": {"poll_interval_s": 7.5}}, False),
        ({"tripo": {"poll_timeout_s": 120}}, False),
        ({"tripo": {"max_credits_per_run": 50}}, False),
    ],
    ids=[
        "prompt",
        "seed",
        "quality",
        "mode",
        "bake",
        "mesh",
        "poll_interval",
        "poll_timeout",
        "max_credits",
    ],
)
def test_fingerprint_sensitivity_matrix(
    tmp_path: Path, change: dict[str, Any], differs: bool
) -> None:
    root_a, root_b = tmp_path / "a", tmp_path / "b"
    fp_a = TextureStage().input_fingerprint(make_model(root_a), make_ctx(root_a))
    fp_b = TextureStage().input_fingerprint(make_model(root_b, **change), make_ctx(root_b))
    assert (fp_a != fp_b) is differs


def test_fingerprint_changes_with_style_image_bytes_and_guidance(tmp_path: Path) -> None:
    root_a, root_b = tmp_path / "a", tmp_path / "b"
    fp_a = TextureStage().input_fingerprint(make_model(root_a), make_ctx(root_a))
    model_b = make_model(root_b)
    _png(root_b / "style" / "references" / "primary.png", (1, 2, 3, 255))
    assert TextureStage().input_fingerprint(model_b, make_ctx(root_b)) != fp_a

    views = TextureOverrides(
        guidance_images=cast(
            "Any", {"front": "references/front.png", "left": "references/left.png"}
        )
    )
    kwargs: dict[str, Any] = {"texture": {"prompt_mode": "images"}, "front_texture": views}
    ma, mb = make_model(root_a, **kwargs), make_model(root_b, **kwargs)
    for m in (ma, mb):
        _png(m.paths.references / "front.png", (5, 5, 5, 255))
        _png(m.paths.references / "left.png", (6, 6, 6, 255))
    assert TextureStage().input_fingerprint(
        ma, make_ctx(root_a)
    ) == TextureStage().input_fingerprint(mb, make_ctx(root_b))
    _png(mb.paths.references / "left.png", (7, 7, 7, 255))
    assert TextureStage().input_fingerprint(
        ma, make_ctx(root_a)
    ) != TextureStage().input_fingerprint(mb, make_ctx(root_b))


def test_fingerprint_is_location_independent(tmp_path: Path) -> None:
    root_a = tmp_path / "a"
    fp_a = TextureStage().input_fingerprint(make_model(root_a), make_ctx(root_a))
    root_b = tmp_path / "moved" / "b"
    shutil.copytree(root_a, root_b)
    assert TextureStage().input_fingerprint(make_model(root_b), make_ctx(root_b)) == fp_a


# --------------------------------------------------------------------------- happy path
def test_happy_path_reserves_commits_and_writes_outputs(root: Path) -> None:
    provider = FakeTextureProvider(
        credits=12.5, normalize_scale=0.5, normalize_translation=(0, -0.25, 0)
    )
    model = make_model(root)
    ctx = make_ctx(root, provider, limit=100.0)

    outcome = TextureStage().run(model, ctx)

    assert outcome.status == "succeeded"
    assert outcome.outputs == [
        "work/tripo/result.glb",
        "work/tripo/task.json",
        "work/tripo/request.json",
    ]
    for rel in outcome.outputs:
        assert (model.paths.folder / rel).is_file(), rel
    assert (model.paths.work / "tripo" / "preview.png").is_file()
    assert not list((model.paths.work / "tripo").glob("*.tmp"))

    req = build_texture_request(
        model.resolved, static_glb=model.paths.work / "static.glb", library_root=root
    )
    assert outcome.extra == {
        EXTRA_TASK_ID: "fake_task_0001",
        "credits": 12.5,
        EXTRA_REQUEST_SHA: req.request_sha256,
        "normalize_scale": 0.5,
        "normalize_translation": [0.0, -0.25, 0.0],
    }
    assert ctx.budget.spent == 12.5
    assert ctx.budget.reserved == 0.0
    assert provider.estimate_calls == [req]
    assert provider.last_call.resume_task_id is None
    assert provider.last_call.request == req


def test_request_json_has_relative_paths_and_exact_task_body(root: Path) -> None:
    model = make_model(root)
    provider = FakeTextureProvider()
    TextureStage().run(model, make_ctx(root, provider))

    text = (model.paths.work / "tripo" / "request.json").read_text(encoding="utf-8")
    assert text.endswith("\n")
    doc = json.loads(text)
    assert list(doc) == sorted(doc)
    assert doc["request"]["mesh_path"] == "work/static.glb"
    assert doc["request"]["style_image"] == "../../style/references/primary.png"
    assert doc["request"]["guidance_images"] == {}
    assert doc["request"]["prompt"] == PROMPT
    assert doc["request"]["mode"] == "text+style"
    assert doc["task_request"]["model"] == "v3.5-20260815"
    assert doc["task_request"]["texture_prompt"] == {"text": PROMPT}
    assert doc["resumed_task_id"] is None

    task = json.loads((model.paths.work / "tripo" / "task.json").read_text(encoding="utf-8"))
    assert task["result_glb"] == "work/tripo/result.glb"
    assert task["preview_png"] == "work/tripo/preview.png"
    assert task["task_id"] == "fake_task_0001"
    assert task["raw_output"]["output"]["model_url"].startswith("fake://")
    assert doc["task_request"] == task["raw_output"]["request_body"]
    assert len(provider.calls) == 1


def test_task_json_without_preview(root: Path) -> None:
    model = make_model(root)
    TextureStage().run(model, make_ctx(root, FakeTextureProvider(preview=False)))
    task = json.loads((model.paths.work / "tripo" / "task.json").read_text(encoding="utf-8"))
    assert task["preview_png"] is None


def test_outcome_extra_normalize_fields_default_to_none(root: Path) -> None:
    outcome = TextureStage().run(make_model(root), make_ctx(root, FakeTextureProvider()))
    assert outcome.extra["normalize_scale"] is None
    assert outcome.extra["normalize_translation"] is None


# --------------------------------------------------------------------------- state / resume
def test_task_id_is_persisted_before_the_provider_returns(root: Path) -> None:
    provider = FakeTextureProvider(raise_=TripoTaskFailedError("boom"))
    model = make_model(root)
    ctx = make_ctx(root, provider)
    with pytest.raises(TripoTaskFailedError):
        TextureStage().run(model, ctx)
    req = provider.last_call.request
    assert _store(ctx).update_calls == [
        (
            "ant",
            "texture",
            {"extra": {EXTRA_TASK_ID: "fake_task_0001", EXTRA_REQUEST_SHA: req.request_sha256}},
        )
    ]
    assert ctx.state.load("ant").stages["texture"].extra[EXTRA_TASK_ID] == "fake_task_0001"


def test_failure_releases_reservation_and_reraises(root: Path) -> None:
    ctx = make_ctx(
        root, FakeTextureProvider(credits=10.0, raise_=TripoTaskFailedError("boom")), limit=15.0
    )
    with pytest.raises(TripoTaskFailedError):
        TextureStage().run(make_model(root), ctx)
    assert ctx.budget.spent == 0.0
    assert ctx.budget.reserved == 0.0
    assert ctx.budget.remaining == 15.0


def test_budget_exceeded_raises_before_any_provider_call(root: Path) -> None:
    provider = FakeTextureProvider(credits=10.0)
    ctx = make_ctx(root, provider, limit=5.0)
    with pytest.raises(BudgetExceededError):
        TextureStage().run(make_model(root), ctx)
    assert len(provider.estimate_calls) == 1
    assert provider.calls == []
    assert _store(ctx).update_calls == []
    assert ctx.budget.reserved == 0.0


def _seed_task(ctx: PipelineContext, model: ModelContext, task_id: str, sha: str | None) -> None:
    extra: dict[str, Any] = {EXTRA_TASK_ID: task_id}
    if sha is not None:
        extra[EXTRA_REQUEST_SHA] = sha
    state = ctx.state.load(model.model_id)
    state.stages["texture"] = StageRecord(extra=extra)


def test_resume_reuses_stored_task_id_when_request_sha_matches(root: Path) -> None:
    model = make_model(root)
    provider = FakeTextureProvider()
    ctx = make_ctx(root, provider, resume_tasks=True)
    sha = build_texture_request(
        model.resolved, static_glb=model.paths.work / "static.glb", library_root=root
    ).request_sha256
    _seed_task(ctx, model, "task_old", sha)

    outcome = TextureStage().run(model, ctx)

    assert provider.last_call.resume_task_id == "task_old"
    assert provider.created_task_ids == []
    assert outcome.extra[EXTRA_TASK_ID] == "task_old"
    doc = json.loads((model.paths.work / "tripo" / "request.json").read_text(encoding="utf-8"))
    assert doc["task_request"] is None
    assert doc["resumed_task_id"] == "task_old"


def test_resume_ignored_when_request_sha_differs(root: Path) -> None:
    model = make_model(root)
    provider = FakeTextureProvider()
    ctx = make_ctx(root, provider)
    _seed_task(ctx, model, "task_old", "0" * 64)
    TextureStage().run(model, ctx)
    assert provider.last_call.resume_task_id is None
    assert provider.created_task_ids == ["fake_task_0001"]


def test_resume_ignored_when_sha_was_never_recorded(root: Path) -> None:
    model = make_model(root)
    provider = FakeTextureProvider()
    ctx = make_ctx(root, provider)
    _seed_task(ctx, model, "task_old", None)
    TextureStage().run(model, ctx)
    assert provider.last_call.resume_task_id is None


def test_resume_tasks_false_ignores_stored_task_id(root: Path) -> None:
    model = make_model(root)
    provider = FakeTextureProvider()
    ctx = make_ctx(root, provider, resume_tasks=False)
    sha = build_texture_request(
        model.resolved, static_glb=model.paths.work / "static.glb", library_root=root
    ).request_sha256
    _seed_task(ctx, model, "task_old", sha)
    TextureStage().run(model, ctx)
    assert provider.last_call.resume_task_id is None
    assert provider.created_task_ids == ["fake_task_0001"]


def test_state_without_texture_record_means_no_resume(root: Path) -> None:
    model = make_model(root)
    provider = FakeTextureProvider()
    ctx = make_ctx(root, provider)
    ctx.state.load(model.model_id).stages.pop("texture")
    TextureStage().run(model, ctx)
    assert provider.last_call.resume_task_id is None


def test_no_stored_task_means_no_resume(root: Path) -> None:
    provider = FakeTextureProvider()
    TextureStage().run(make_model(root), make_ctx(root, provider))
    assert provider.last_call.resume_task_id is None


# --------------------------------------------------------------------------- prerequisites
def test_missing_static_glb_is_a_prereq_error_before_reserving(root: Path) -> None:
    provider = FakeTextureProvider()
    ctx = make_ctx(root, provider)
    with pytest.raises(StagePrereqError, match=r"static\.glb"):
        TextureStage().run(make_model(root, with_static=False), ctx)
    assert provider.estimate_calls == []
    assert ctx.budget.reserved == 0.0
