"""Stage graph (plan 03 §4, 13 §1).

v1 is the linear chain ``discover → prepare → texture → verify → apply → export → handoff``. The
graph stores explicit edges so a DAG can be introduced later, but today it validates that the
stages form a contiguous-order subsequence of :data:`STAGE_ORDER` and derives the edges itself.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

from tinyworld_pipeline.domain.errors import UsageError
from tinyworld_pipeline.domain.state import (
    STAGE_ORDER,
    StageName,
    downstream_of,
    stage_index,
    stages_between,
)
from tinyworld_pipeline.pipeline.stage import Stage


class StageGraph:
    """An ordered collection of :class:`Stage` instances with linear edges.

    Args:
        stages: Stage instances with distinct names, listed in :data:`STAGE_ORDER` order.

    Raises:
        ValueError: Duplicate names, unknown names, or names out of canonical order.
    """

    def __init__(self, stages: Sequence[Stage]) -> None:
        names = [s.name for s in stages]
        if len(set(names)) != len(names):
            msg = f"duplicate stage names: {names}"
            raise ValueError(msg)
        unknown = [n for n in names if n not in STAGE_ORDER]
        if unknown:
            msg = f"unknown stage name(s): {unknown}"
            raise ValueError(msg)
        indices = [stage_index(n) for n in names]
        if indices != sorted(indices):
            msg = f"stages must follow STAGE_ORDER, got {names}"
            raise ValueError(msg)
        self._stages: dict[StageName, Stage] = {s.name: s for s in stages}
        self._names: tuple[StageName, ...] = tuple(names)
        self._edges: tuple[tuple[StageName, StageName], ...] = tuple(
            zip(self._names, self._names[1:], strict=False)
        )

    @classmethod
    def linear(cls, stages: Sequence[Stage]) -> StageGraph:
        """Build the v1 linear graph (alias of the constructor, kept for intent)."""
        return cls(stages)

    # ----------------------------------------------------------------- queries
    @property
    def names(self) -> tuple[StageName, ...]:
        """Stage names in execution order."""
        return self._names

    @property
    def edges(self) -> tuple[tuple[StageName, StageName], ...]:
        """``(upstream, downstream)`` pairs; linear in v1."""
        return self._edges

    def by_name(self, name: StageName) -> Stage:
        """The stage called ``name``.

        Raises:
            UsageError: The graph has no such stage.
        """
        try:
            return self._stages[name]
        except KeyError:
            raise UsageError(
                f"stage {name!r} is not part of this pipeline",
                hint="available stages: " + ", ".join(self._names),
            ) from None

    def stages_between(self, start: StageName, end: StageName) -> list[Stage]:
        """Stages of the inclusive canonical range ``start..end`` that exist in this graph.

        Raises:
            UsageError: ``start`` is after ``end``, or either endpoint is missing from the graph.
        """
        try:
            wanted = stages_between(start, end)
        except ValueError as exc:
            raise UsageError(str(exc), hint="--from-stage must not be after --to-stage") from exc
        for endpoint in (start, end):
            self.by_name(endpoint)
        return [self._stages[n] for n in wanted if n in self._stages]

    def downstream(self, name: StageName) -> tuple[StageName, ...]:
        """Names of every stage strictly after ``name`` that exists in this graph."""
        return tuple(n for n in downstream_of(name) if n in self._stages)

    def upstream(self, name: StageName) -> tuple[StageName, ...]:
        """Names of every stage strictly before ``name`` that exists in this graph."""
        cut = stage_index(name)
        return tuple(n for n in self._names if stage_index(n) < cut)

    def __iter__(self) -> Iterator[Stage]:
        return iter(self._stages.values())

    def __len__(self) -> int:
        return len(self._stages)

    def __contains__(self, name: object) -> bool:
        return name in self._stages


__all__ = ["StageGraph"]
