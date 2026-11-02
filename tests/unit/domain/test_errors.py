from __future__ import annotations

import re
from pathlib import Path

import pytest

from tinyworld_pipeline.domain import errors
from tinyworld_pipeline.domain.errors import (
    BlenderJobError,
    BlenderNotFoundError,
    BudgetExceededError,
    CancelledError,
    ConfigError,
    PipelineError,
    Problem,
    TripoAuthError,
    TripoRateLimited,
    TripoRateLimitedError,
    UsageError,
    ValidationError,
    all_error_classes,
)

PLAN_DIR = Path(__file__).resolve().parents[3] / "docs" / "plan"


def test_every_error_code_is_unique() -> None:
    classes = [c for c in all_error_classes() if c is not BlenderJobError]
    codes = [c.code for c in classes]
    assert len(codes) == len(set(codes)), sorted(c for c in codes if codes.count(c) > 1)


def test_every_error_code_is_documented_in_the_plan() -> None:
    plan_text = "\n".join(p.read_text(encoding="utf-8") for p in PLAN_DIR.glob("*.md"))
    documented = set(re.findall(r"\bE_[A-Z0-9_]+\b", plan_text))
    missing = {c.code for c in all_error_classes()} - documented - {"E_BJ"}
    assert not missing, f"undocumented error codes: {sorted(missing)}"


@pytest.mark.parametrize(
    ("cls", "exit_code"),
    [
        (PipelineError, 1),
        (UsageError, 2),
        (ConfigError, 2),
        (BlenderNotFoundError, 2),
        (TripoAuthError, 2),
        (ValidationError, 3),
        (BudgetExceededError, 5),
        (CancelledError, 130),
    ],
)
def test_exit_codes_follow_plan_08(cls: type[PipelineError], exit_code: int) -> None:
    assert cls.exit_code == exit_code


def test_pipeline_error_carries_message_detail_hint_and_cause() -> None:
    cause = RuntimeError("boom")
    err = ConfigError("bad", detail={"k": 1}, hint="fix it", cause=cause)

    assert str(err) == "bad"
    assert err.__cause__ is cause
    assert err.to_dict() == {
        "code": "E_CONFIG",
        "message": "bad",
        "hint": "fix it",
        "detail": {"k": 1},
    }


def test_pipeline_error_to_dict_omits_empty_optional_parts() -> None:
    assert PipelineError("x").to_dict() == {"code": "E_PIPELINE", "message": "x"}


def test_validation_error_counts_errors_not_warnings() -> None:
    problems = [
        Problem(code="E_MODEL_ID", severity="error", message="bad id"),
        Problem(code="W_LICENSE_MISSING", severity="warning", message="no licence"),
    ]
    err = ValidationError(problems)

    assert err.message == "1 validation error(s)"
    assert err.problems == problems
    assert err.detail["problems"][0]["code"] == "E_MODEL_ID"


def test_problem_render_includes_path_and_hint() -> None:
    p = Problem(
        code="E_UNKNOWN_KEY",
        severity="error",
        model_id="ant",
        path="model.md:texture.qualty",
        message="unknown key",
        hint="did you mean quality?",
    )
    assert (
        p.render()
        == "[E_UNKNOWN_KEY] model.md:texture.qualty — unknown key (did you mean quality?)"
    )
    assert Problem(code="E_X", severity="error", message="m").render() == "[E_X] m"


def test_blender_job_error_reports_the_job_code() -> None:
    err = BlenderJobError(
        "E_BJ_SHAPEKEYS", "shape keys", detail={"meshes": ["Eye"]}, traceback="tb"
    )

    assert err.job_code == "E_BJ_SHAPEKEYS"
    assert err.traceback == "tb"
    assert err.to_dict()["code"] == "E_BJ_SHAPEKEYS"
    assert err.exit_code == 1


def test_rate_limited_error_keeps_retry_after_and_plan_alias() -> None:
    err = TripoRateLimitedError("slow down", retry_after_s=2.5)
    assert err.retry_after_s == 2.5
    assert TripoRateLimited is TripoRateLimitedError
    assert errors.TripoTaskFailed is errors.TripoTaskFailedError
    assert errors.TripoTaskTimeout is errors.TripoTaskTimeoutError
