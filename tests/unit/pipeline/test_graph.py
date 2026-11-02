from __future__ import annotations

from itertools import pairwise
from typing import cast

import pytest

from tinyworld_pipeline.domain.errors import UsageError
from tinyworld_pipeline.domain.state import STAGE_ORDER, StageName
from tinyworld_pipeline.pipeline.graph import StageGraph

from .conftest import FakeStage, fake_graph


def test_linear_graph_names_edges_and_lookup() -> None:
    graph, stages = fake_graph()
    assert graph.names == STAGE_ORDER
    assert graph.edges == tuple(pairwise(STAGE_ORDER))
    assert graph.by_name("verify") is stages["verify"]
    assert "verify" in graph
    assert "nope" not in graph
    assert len(graph) == 7
    assert [s.name for s in graph] == list(STAGE_ORDER)
    assert StageGraph.linear(list(stages.values())).names == STAGE_ORDER


def test_stages_between_downstream_upstream() -> None:
    graph, _ = fake_graph()
    assert [s.name for s in graph.stages_between("texture", "apply")] == [
        "texture",
        "verify",
        "apply",
    ]
    assert graph.downstream("verify") == ("apply", "export", "handoff")
    assert graph.downstream("handoff") == ()
    assert graph.upstream("texture") == ("discover", "prepare")
    assert graph.upstream("discover") == ()


def test_partial_graph_keeps_only_present_stages() -> None:
    graph = StageGraph([FakeStage("discover"), FakeStage("verify"), FakeStage("handoff")])
    assert [s.name for s in graph.stages_between("discover", "handoff")] == [
        "discover",
        "verify",
        "handoff",
    ]
    assert graph.downstream("discover") == ("verify", "handoff")
    with pytest.raises(UsageError, match="not part of this pipeline"):
        graph.stages_between("prepare", "handoff")
    with pytest.raises(UsageError):
        graph.by_name("texture")


def test_invalid_range_is_usage_error() -> None:
    graph, _ = fake_graph()
    with pytest.raises(UsageError, match="after"):
        graph.stages_between("apply", "texture")


@pytest.mark.parametrize(
    ("names", "message"),
    [
        (["discover", "discover"], "duplicate"),
        (["discover", "bogus"], "unknown"),
        (["prepare", "discover"], "STAGE_ORDER"),
    ],
)
def test_constructor_validation(names: list[str], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        StageGraph([FakeStage(cast("StageName", n)) for n in names])
