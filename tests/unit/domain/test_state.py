from __future__ import annotations

import pytest
from pydantic import ValidationError

from tinyworld_pipeline.domain.state import (
    STAGE_ORDER,
    ModelState,
    StageOutcome,
    StageRecord,
    StageStatus,
    downstream_of,
    empty_stage_records,
    stage_index,
    stages_between,
)


def test_stage_order_is_the_v1_linear_graph() -> None:
    assert STAGE_ORDER == ("discover", "prepare", "texture", "verify", "apply", "export", "handoff")
    assert [stage_index(s) for s in STAGE_ORDER] == list(range(7))  # type: ignore[arg-type]


def test_stages_between_is_inclusive_and_rejects_reversed_ranges() -> None:
    assert stages_between("prepare", "verify") == ("prepare", "texture", "verify")
    assert stages_between("apply", "apply") == ("apply",)
    with pytest.raises(ValueError, match="after"):
        stages_between("verify", "prepare")


def test_downstream_of_excludes_the_stage_itself() -> None:
    assert downstream_of("export") == ("handoff",)
    assert downstream_of("handoff") == ()
    assert downstream_of("discover") == STAGE_ORDER[1:]


def test_new_model_state_has_pending_records_for_every_stage() -> None:
    state = ModelState(model_id="ant", pipeline_version="0.1.0")

    assert set(state.stages) == set(STAGE_ORDER)
    assert all(r == StageRecord() for r in state.stages.values())
    assert state.stages["prepare"].status is StageStatus.PENDING
    assert state.credits_total == 0.0
    assert empty_stage_records()["handoff"].attempts == 0


def test_model_state_is_mutable_but_validated() -> None:
    state = ModelState(model_id="ant", pipeline_version="0.1.0")
    state.credits_total = 12.5
    assert state.credits_total == 12.5
    with pytest.raises(ValidationError):
        state.credits_total = "lots"  # type: ignore[assignment]


def test_stage_record_and_outcome_round_trip_json() -> None:
    rec = StageRecord(
        status=StageStatus.SUCCEEDED, fingerprint="a" * 64, outputs=["work/x"], extra={"t": 1}
    )
    assert StageRecord.model_validate_json(rec.model_dump_json()) == rec
    out = StageOutcome(status="succeeded", outputs=["work/x"])
    assert out.warnings == []
    assert out.extra == {}
    with pytest.raises(ValidationError):
        StageOutcome(status="failed", outputs=[])


def test_stage_status_serialises_as_plain_strings() -> None:
    assert StageStatus.SKIPPED.value == "skipped"
    assert StageRecord(status="running").status is StageStatus.RUNNING
