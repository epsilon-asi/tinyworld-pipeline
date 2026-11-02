from __future__ import annotations

import threading
from pathlib import Path

from grounded_pipeline.domain.fingerprint import SurfaceFingerprint, VerifyReport
from grounded_pipeline.domain.ports import CancelToken
from grounded_pipeline.domain.prompt import PROMPT_MAX_CHARS, PROMPT_WARN_CHARS, ComposedPrompt
from grounded_pipeline.domain.reports import ModelRunSummary, RunEnvironment, RunReport
from grounded_pipeline.domain.style_spec import StyleDocument, StyleFrontMatter


def _fp(**overrides: object) -> SurfaceFingerprint:
    base: dict[str, object] = {
        "triangle_count": 12,
        "unique_position_count": 8,
        "position_set_sha256": "a" * 64,
        "surface_sha256": "b" * 64,
        "has_uv": True,
        "primitive_count": 1,
        "material_count": 1,
        "quantum_position": 1e-4,
        "quantum_uv": 1e-4,
    }
    base.update(overrides)
    return SurfaceFingerprint.model_validate(base)


def test_verify_report_round_trips() -> None:
    report = VerifyReport(
        ours=_fp(),
        theirs=_fp(surface_sha256="c" * 64),
        topology_match=True,
        position_match=True,
        uv_match=False,
        route="bake",
        reason="uv mismatch",
        forced=False,
    )
    assert VerifyReport.model_validate_json(report.model_dump_json()) == report


def test_cancel_token_is_thread_safe_flag() -> None:
    token = CancelToken()
    assert not token.is_set()
    assert token.wait(0.001) is False

    threading.Thread(target=token.set).start()
    assert token.wait(1.0) is True
    assert token.is_set()


def test_prompt_limits_and_composed_prompt() -> None:
    assert PROMPT_WARN_CHARS < PROMPT_MAX_CHARS == 1000
    cp = ComposedPrompt(text="ant", sources={"model_prompt": "prompt"}, length=3, sha256="0" * 64)
    assert cp.length == len(cp.text)


def test_style_document_defaults() -> None:
    doc = StyleDocument(
        folder=Path("/lib/style"), front=StyleFrontMatter(name="s"), sections={"base_prompt": "x"}
    )
    assert doc.front.primary_reference == Path("references/primary.png")
    assert doc.front.negative_terms == []


def test_run_report_builds_from_summaries() -> None:
    summary = ModelRunSummary(
        model_id="ant",
        result="succeeded",
        stages={"discover": "succeeded"},
        credits=10,
        duration_s=1.5,
    )
    report = RunReport(
        run_id="01ARZ3NDEKTSV4RRFFQ69G5FAV",
        started_at="t0",
        finished_at="t1",
        library=Path("/lib"),
        options={"force": False},
        models=[summary],
        credits_total=10,
        exit_code=0,
        environment=RunEnvironment(python="3.12", pipeline_version="0.1.0", os="darwin"),
    )
    assert report.problems == []
    assert report.models[0].stages["discover"] == "succeeded"
