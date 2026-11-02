"""State store port and its two implementations (plan 13 §4).

* :class:`StateStore` — the port stages and the runner depend on.
* :class:`JsonStateStore` — ``<model>/.pipeline/state.json`` guarded by ``state.lock`` (filelock,
  30 s timeout), written atomically, schema-validated on load; a corrupt file is backed up as
  ``state.corrupt-<ts>.json`` and recreated with a warning.
* :class:`InMemoryStateStore` — dict-backed, thread-safe, identical semantics; for tests.

``update(model_id, stage, **changes)`` applies ``changes`` to the stage's :class:`StageRecord`
(``status``, ``fingerprint``, ``outputs``, ``started_at``, ``finished_at``, ``duration_s``,
``run_id``, ``error_code``, ``error_message``, ``attempts``; ``extra`` is *merged*). Pass
``extra_replace=`` to overwrite ``extra`` wholesale. Model-level fields have dedicated methods
(:meth:`JsonStateStore.set_last_run`, :meth:`JsonStateStore.add_credits`).
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import structlog
from filelock import FileLock, Timeout
from pydantic import ValidationError as PydanticValidationError

from grounded_pipeline.domain.errors import LibraryBusyError
from grounded_pipeline.domain.state import ModelState, StageName, StageRecord
from grounded_pipeline.library.paths import ModelPaths
from grounded_pipeline.pipeline.atomic import atomic_write_json
from grounded_pipeline.pipeline.timing import compact_utc_timestamp

log = structlog.get_logger(__name__)

STATE_LOCK_TIMEOUT_S = 30.0
_STAGE_FIELDS = frozenset(StageRecord.model_fields)


class StateStore(Protocol):
    """Load/save/update ``<model>/.pipeline/state.json``."""

    def load(self, model_id: str) -> ModelState:
        """Return the model's state, creating a fresh one when none exists."""
        ...

    def save(self, state: ModelState) -> None:
        """Persist ``state`` atomically."""
        ...

    def update(self, model_id: str, stage: StageName, **changes: Any) -> ModelState:  # noqa: ANN401
        """Load-modify-save ``stages[stage]`` under the model's lock; returns the new state."""
        ...


# --------------------------------------------------------------------------- shared helpers
def apply_stage_changes(
    record: StageRecord, changes: dict[str, Any], *, extra_replace: dict[str, Any] | None = None
) -> StageRecord:
    """Return ``record`` with ``changes`` applied (``extra`` merged, other fields replaced).

    Args:
        record: Current stage record.
        changes: Field → new value. ``extra`` is merged into the existing dict.
        extra_replace: When given, replaces ``extra`` entirely (then ``changes["extra"]`` is
            merged on top).

    Raises:
        TypeError: ``changes`` names a field :class:`StageRecord` does not have.
        pydantic.ValidationError: A value has the wrong type.
    """
    unknown = set(changes) - _STAGE_FIELDS
    if unknown:
        msg = f"unknown StageRecord field(s): {', '.join(sorted(unknown))}"
        raise TypeError(msg)
    data = record.model_dump()
    extra = dict(record.extra) if extra_replace is None else dict(extra_replace)
    if "extra" in changes:
        extra.update(changes["extra"])
    data.update({k: v for k, v in changes.items() if k != "extra"})
    data["extra"] = extra
    return StageRecord.model_validate(data)


def _apply(
    state: ModelState,
    stage: StageName,
    changes: dict[str, Any],
    extra_replace: dict[str, Any] | None,
) -> None:
    stages = dict(state.stages)
    stages[stage] = apply_stage_changes(
        stages.get(stage, StageRecord()), changes, extra_replace=extra_replace
    )
    state.stages = stages


@runtime_checkable
class PathsResolver(Protocol):
    """Anything with ``.model(model_id) -> ModelPaths`` (``LibraryPaths``, ``Library.paths``)."""

    def model(self, model_id: str) -> ModelPaths:
        """Paths of ``models/<model_id>/``."""
        ...


# --------------------------------------------------------------------------- JSON store
class JsonStateStore:
    """File-backed :class:`StateStore` (plan 13 §4, §7).

    Args:
        paths: A :class:`PathsResolver` (e.g. ``library.paths``) or a callable
            ``model_id -> ModelPaths``.
        pipeline_version: Version stamped into freshly created states.
        lock_timeout_s: Seconds to wait for ``state.lock`` in :meth:`update`.
    """

    def __init__(
        self,
        paths: PathsResolver | Callable[[str], ModelPaths],
        *,
        pipeline_version: str,
        lock_timeout_s: float = STATE_LOCK_TIMEOUT_S,
    ) -> None:
        self._resolve: Callable[[str], ModelPaths] = (
            paths.model if isinstance(paths, PathsResolver) else paths
        )
        self._pipeline_version = pipeline_version
        self._lock_timeout_s = lock_timeout_s

    def paths(self, model_id: str) -> ModelPaths:
        """Paths of the model folder (``state.json`` is ``paths.state_file``)."""
        return self._resolve(model_id)

    def fresh(self, model_id: str) -> ModelState:
        """A new, all-pending state for ``model_id``."""
        return ModelState(model_id=model_id, pipeline_version=self._pipeline_version)

    def load(self, model_id: str) -> ModelState:
        """Read ``state.json``; missing → fresh state; corrupt → backed up, recreated, warned."""
        file = self.paths(model_id).state_file
        if not file.is_file():
            return self.fresh(model_id)
        try:
            data = json.loads(file.read_text(encoding="utf-8"))
            state = ModelState.model_validate(data)
        except (json.JSONDecodeError, UnicodeDecodeError, PydanticValidationError) as exc:
            backup = self._backup_corrupt(file)
            log.warning(
                "state_corrupt_recreated",
                model_id=model_id,
                file=str(file),
                backup=str(backup),
                error=str(exc).splitlines()[0],
            )
            state = self.fresh(model_id)
            self.save(state)
            return state
        if state.model_id != model_id:
            backup = self._backup_corrupt(file)
            log.warning(
                "state_model_id_mismatch_recreated",
                model_id=model_id,
                found=state.model_id,
                backup=str(backup),
            )
            state = self.fresh(model_id)
            self.save(state)
        return state

    def save(self, state: ModelState) -> None:
        """Write ``state.json`` atomically (temp sibling + ``os.replace``)."""
        atomic_write_json(self.paths(state.model_id).state_file, state)

    def update(
        self,
        model_id: str,
        stage: StageName,
        *,
        extra_replace: dict[str, Any] | None = None,
        **changes: Any,  # noqa: ANN401
    ) -> ModelState:
        """Load-modify-save one stage record under ``state.lock``.

        Args:
            model_id: Model folder name.
            stage: Stage whose record changes.
            extra_replace: Replace ``extra`` wholesale instead of merging.
            **changes: :class:`StageRecord` fields; ``extra`` is merged.

        Returns:
            The persisted state.

        Raises:
            LibraryBusyError: ``state.lock`` could not be acquired within the timeout.
            TypeError: Unknown field name.
        """
        with self._locked(model_id):
            state = self.load(model_id)
            _apply(state, stage, changes, extra_replace)
            self.save(state)
            return state

    def set_last_run(self, model_id: str, run_id: str) -> ModelState:
        """Record the id of the run that last touched this model."""

        def _set(state: ModelState) -> None:
            state.last_run_id = run_id

        return self.transform(model_id, _set)

    def add_credits(self, model_id: str, amount: float) -> ModelState:
        """Add ``amount`` credits to the model's lifetime total."""

        def _add(state: ModelState) -> None:
            state.credits_total = round(state.credits_total + float(amount), 6)

        return self.transform(model_id, _add)

    def transform(self, model_id: str, fn: Callable[[ModelState], None]) -> ModelState:
        """Load, apply ``fn`` in place, save — all under the model lock."""
        with self._locked(model_id):
            state = self.load(model_id)
            fn(state)
            self.save(state)
            return state

    # ----------------------------------------------------------------- internals
    @contextmanager
    def _locked(self, model_id: str) -> Iterator[None]:
        lock_file = self.paths(model_id).lock_file
        lock_file.parent.mkdir(parents=True, exist_ok=True)
        lock = FileLock(str(lock_file))
        try:
            lock.acquire(timeout=self._lock_timeout_s)
        except Timeout as exc:
            raise LibraryBusyError(
                f"timed out after {self._lock_timeout_s:g}s waiting for {lock_file}",
                detail={"lock": str(lock_file), "model_id": model_id},
                hint="another process is updating this model's state; retry or remove a stale lock",
                cause=exc,
            ) from exc
        try:
            yield
        finally:
            lock.release()

    @staticmethod
    def _backup_corrupt(file: Path) -> Path:
        stamp = compact_utc_timestamp()
        backup = file.with_name(f"state.corrupt-{stamp}.json")
        counter = 1
        while backup.exists():
            backup = file.with_name(f"state.corrupt-{stamp}-{counter}.json")
            counter += 1
        file.replace(backup)
        return backup


# --------------------------------------------------------------------------- in-memory store
class InMemoryStateStore:
    """Dict-backed :class:`StateStore` with the same semantics as :class:`JsonStateStore`.

    Every ``load`` returns a deep copy, so callers cannot mutate stored state by accident; every
    mutation goes through :meth:`save`/:meth:`update`. ``saves`` counts persisted writes (tests).

    Args:
        pipeline_version: Version stamped into freshly created states.
    """

    def __init__(self, pipeline_version: str = "0.0.0") -> None:
        self._pipeline_version = pipeline_version
        self._states: dict[str, ModelState] = {}
        self._lock = threading.RLock()
        self.saves = 0

    def fresh(self, model_id: str) -> ModelState:
        """A new, all-pending state for ``model_id``."""
        return ModelState(model_id=model_id, pipeline_version=self._pipeline_version)

    def load(self, model_id: str) -> ModelState:
        """Return a deep copy of the stored state (fresh when unknown)."""
        with self._lock:
            state = self._states.get(model_id)
            return state.model_copy(deep=True) if state is not None else self.fresh(model_id)

    def save(self, state: ModelState) -> None:
        """Store a deep copy of ``state``."""
        with self._lock:
            self._states[state.model_id] = state.model_copy(deep=True)
            self.saves += 1

    def update(
        self,
        model_id: str,
        stage: StageName,
        *,
        extra_replace: dict[str, Any] | None = None,
        **changes: Any,  # noqa: ANN401
    ) -> ModelState:
        """Same contract as :meth:`JsonStateStore.update`."""
        with self._lock:
            state = self.load(model_id)
            _apply(state, stage, changes, extra_replace)
            self.save(state)
            return state

    def set_last_run(self, model_id: str, run_id: str) -> ModelState:
        """Record the id of the run that last touched this model."""

        def _set(state: ModelState) -> None:
            state.last_run_id = run_id

        return self.transform(model_id, _set)

    def add_credits(self, model_id: str, amount: float) -> ModelState:
        """Add ``amount`` credits to the model's lifetime total."""

        def _add(state: ModelState) -> None:
            state.credits_total = round(state.credits_total + float(amount), 6)

        return self.transform(model_id, _add)

    def transform(self, model_id: str, fn: Callable[[ModelState], None]) -> ModelState:
        """Load, apply ``fn`` in place, save — atomically."""
        with self._lock:
            state = self.load(model_id)
            fn(state)
            self.save(state)
            return state

    def known_models(self) -> list[str]:
        """Ids with a stored state (tests)."""
        with self._lock:
            return sorted(self._states)


__all__ = [
    "STATE_LOCK_TIMEOUT_S",
    "InMemoryStateStore",
    "JsonStateStore",
    "PathsResolver",
    "StateStore",
    "apply_stage_changes",
]
