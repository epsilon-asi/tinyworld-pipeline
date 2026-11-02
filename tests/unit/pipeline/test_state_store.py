from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest
from filelock import FileLock

from grounded_pipeline.domain.errors import LibraryBusyError
from grounded_pipeline.domain.state import STAGE_ORDER, ModelState, StageRecord, StageStatus
from grounded_pipeline.library.paths import LibraryPaths
from grounded_pipeline.pipeline.state_store import (
    InMemoryStateStore,
    JsonStateStore,
    StateStore,
    apply_stage_changes,
)

MODEL = "cube"
RUN_ID = "01ARZ3NDEKTSV4RRFFQ69G5FAV"


@pytest.fixture
def paths(tmp_path: Path) -> LibraryPaths:
    paths = LibraryPaths.from_root(tmp_path / "lib")
    paths.model(MODEL).folder.mkdir(parents=True)
    return paths


@pytest.fixture
def json_store(paths: LibraryPaths) -> JsonStateStore:
    return JsonStateStore(paths, pipeline_version="0.1.0")


@pytest.fixture(params=["json", "memory"])
def store(request: pytest.FixtureRequest, paths: LibraryPaths) -> StateStore:
    if request.param == "json":
        return JsonStateStore(paths.model, pipeline_version="0.1.0")
    return InMemoryStateStore("0.1.0")


# --------------------------------------------------------------------------- parity (both stores)
def test_load_missing_returns_fresh_state(store: StateStore) -> None:
    state = store.load(MODEL)
    assert state.model_id == MODEL
    assert state.pipeline_version == "0.1.0"
    assert set(state.stages) == set(STAGE_ORDER)
    assert all(r.status == StageStatus.PENDING for r in state.stages.values())


def test_save_then_load_roundtrip(store: StateStore) -> None:
    state = store.load(MODEL)
    state.last_run_id = RUN_ID
    state.credits_total = 2.5
    store.save(state)
    again = store.load(MODEL)
    assert again.last_run_id == RUN_ID
    assert again.credits_total == 2.5
    assert again is not state


def test_update_merges_extra_and_replaces_fields(store: StateStore) -> None:
    store.update(MODEL, "texture", status=StageStatus.RUNNING, attempts=1, extra={"task_id": "t1"})
    state = store.update(
        MODEL,
        "texture",
        status=StageStatus.SUCCEEDED,
        outputs=["work/a"],
        extra={"credits": 10},
    )
    rec = state.stages["texture"]
    assert rec.status == StageStatus.SUCCEEDED
    assert rec.attempts == 1
    assert rec.extra == {"task_id": "t1", "credits": 10}
    assert rec.outputs == ["work/a"]
    assert store.load(MODEL).stages["prepare"].status == StageStatus.PENDING


def test_update_extra_replace(store: StateStore) -> None:
    store.update(MODEL, "apply", extra={"a": 1, "invalidated_by": "verify"})
    rec = store.update(MODEL, "apply", extra_replace={"b": 2}, extra={"c": 3}).stages["apply"]
    assert rec.extra == {"b": 2, "c": 3}


def test_update_unknown_field_is_type_error(store: StateStore) -> None:
    with pytest.raises(TypeError, match="unknown StageRecord field"):
        store.update(MODEL, "apply", bogus=1)


def test_model_level_helpers(store: StateStore) -> None:
    assert isinstance(store, JsonStateStore | InMemoryStateStore)
    store.set_last_run(MODEL, RUN_ID)
    store.add_credits(MODEL, 1.25)
    state = store.add_credits(MODEL, 0.75)
    assert state.last_run_id == RUN_ID
    assert state.credits_total == 2.0

    def bump(s: ModelState) -> None:
        s.pipeline_version = "0.2.0"

    assert store.transform(MODEL, bump).pipeline_version == "0.2.0"
    assert store.load(MODEL).pipeline_version == "0.2.0"


def test_parity_between_json_and_memory(paths: LibraryPaths) -> None:
    stores: list[StateStore] = [
        JsonStateStore(paths, pipeline_version="0.1.0"),
        InMemoryStateStore("0.1.0"),
    ]
    for s in stores:
        s.update(MODEL, "discover", status=StageStatus.SUCCEEDED, fingerprint="a" * 64)
        s.update(MODEL, "prepare", status=StageStatus.FAILED, error_code="E_X", attempts=2)
        s.update(MODEL, "prepare", extra={"k": "v"})
    dumps = [s.load(MODEL).model_dump() for s in stores]
    assert dumps[0] == dumps[1]


# --------------------------------------------------------------------------- JSON specifics
def test_json_store_writes_atomically_and_no_tmp_left(
    json_store: JsonStateStore, paths: LibraryPaths
) -> None:
    json_store.update(MODEL, "discover", status=StageStatus.RUNNING)
    model_paths = paths.model(MODEL)
    assert model_paths.state_file.is_file()
    data = json.loads(model_paths.state_file.read_text(encoding="utf-8"))
    assert data["stages"]["discover"]["status"] == "running"
    assert data["schema_version"] == 1
    assert [p.name for p in model_paths.pipeline.iterdir() if ".tmp-" in p.name] == []
    assert json_store.paths(MODEL) == model_paths


def test_json_store_accepts_callable_resolver(paths: LibraryPaths) -> None:
    store = JsonStateStore(paths.model, pipeline_version="0.1.0")
    assert store.paths(MODEL) == paths.model(MODEL)


def test_crash_between_tmp_and_replace_keeps_old_state(
    json_store: JsonStateStore, paths: LibraryPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    json_store.update(MODEL, "discover", status=StageStatus.SUCCEEDED)
    before = paths.model(MODEL).state_file.read_text()

    def boom(self: Path, other: Path) -> Path:
        raise OSError("simulated crash")

    monkeypatch.setattr(Path, "replace", boom)
    with pytest.raises(OSError, match="simulated"):
        json_store.update(MODEL, "discover", status=StageStatus.FAILED)
    monkeypatch.undo()
    assert paths.model(MODEL).state_file.read_text() == before
    assert json_store.load(MODEL).stages["discover"].status == StageStatus.SUCCEEDED


def test_lock_contention_two_threads(json_store: JsonStateStore) -> None:
    rounds = 25

    def worker() -> None:
        for _ in range(rounds):
            state = json_store.update(MODEL, "texture", extra={})
            json_store.update(
                MODEL, "texture", attempts=state.stages["texture"].attempts
            )  # no-op write

    def incrementer() -> None:
        for _ in range(rounds):

            def bump(s: ModelState) -> None:
                rec = s.stages["texture"]
                s.stages = {
                    **s.stages,
                    "texture": rec.model_copy(update={"attempts": rec.attempts + 1}),
                }

            json_store.transform(MODEL, bump)

    threads = [threading.Thread(target=incrementer) for _ in range(2)]
    threads.append(threading.Thread(target=worker))
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    assert json_store.load(MODEL).stages["texture"].attempts == 2 * rounds


def test_lock_timeout_is_library_busy(paths: LibraryPaths) -> None:
    store = JsonStateStore(paths, pipeline_version="0.1.0", lock_timeout_s=0.05)
    lock_file = paths.model(MODEL).lock_file
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    held = threading.Event()
    release = threading.Event()

    def holder() -> None:
        with FileLock(str(lock_file)):
            held.set()
            release.wait(5)

    thread = threading.Thread(target=holder)
    thread.start()
    assert held.wait(5)
    try:
        with pytest.raises(LibraryBusyError, match="timed out"):
            store.update(MODEL, "discover", status=StageStatus.RUNNING)
    finally:
        release.set()
        thread.join(5)


def test_corrupt_json_is_backed_up_and_recreated(
    json_store: JsonStateStore, paths: LibraryPaths
) -> None:
    file = paths.model(MODEL).state_file
    file.parent.mkdir(parents=True)
    file.write_text("{not json", encoding="utf-8")
    state = json_store.load(MODEL)
    assert state.stages["discover"].status == StageStatus.PENDING
    backups = sorted(file.parent.glob("state.corrupt-*.json"))
    assert len(backups) == 1
    assert backups[0].read_text() == "{not json"
    assert json.loads(file.read_text())["model_id"] == MODEL


def test_unschemad_json_is_backed_up(json_store: JsonStateStore, paths: LibraryPaths) -> None:
    file = paths.model(MODEL).state_file
    file.parent.mkdir(parents=True)
    file.write_text(json.dumps({"schema_version": 1, "model_id": MODEL, "unknown": 1}))
    json_store.load(MODEL)
    file.write_text(json.dumps({"model_id": "other", "pipeline_version": "0.1.0"}))
    assert json_store.load(MODEL).model_id == MODEL
    assert len(list(file.parent.glob("state.corrupt-*.json"))) == 2


def test_backup_name_never_collides(
    json_store: JsonStateStore, paths: LibraryPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "grounded_pipeline.pipeline.state_store.compact_utc_timestamp", lambda: "FIXED"
    )
    file = paths.model(MODEL).state_file
    file.parent.mkdir(parents=True)
    for _ in range(2):
        file.write_text("garbage")
        json_store.load(MODEL)
    names = sorted(p.name for p in file.parent.glob("state.corrupt-*.json"))
    assert names == ["state.corrupt-FIXED-1.json", "state.corrupt-FIXED.json"]


# --------------------------------------------------------------------------- in-memory specifics
def test_in_memory_load_returns_copies_and_counts_saves() -> None:
    store = InMemoryStateStore()
    state = store.load(MODEL)
    state.credits_total = 99
    assert store.load(MODEL).credits_total == 0
    store.save(state)
    assert store.saves == 1
    assert store.known_models() == [MODEL]
    assert store.load(MODEL).pipeline_version == "0.0.0"


def test_apply_stage_changes_validates_types() -> None:
    with pytest.raises(ValueError, match="attempts"):
        apply_stage_changes(StageRecord(), {"attempts": "many"})
