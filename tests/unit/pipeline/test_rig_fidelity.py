"""Rig Fidelity Check (plan 12 §1.4): every check x pass/warn/fail in both modes, boundaries."""

from __future__ import annotations

from typing import Any

import pytest

from fakes.fake_blender_runner import sample_rig
from tinyworld_pipeline.blender.protocol import Bone, RigFingerprint
from tinyworld_pipeline.pipeline.rig_fidelity import (
    CHECK_NAMES,
    RfcMode,
    RfcTolerances,
    RigFidelityReport,
    Severity,
    rig_fidelity_check,
)

TOL = RfcTolerances()
MODES: tuple[RfcMode, ...] = ("model", "animation")


# --------------------------------------------------------------------------- helpers
def bone(src: Bone, **changes: Any) -> Bone:
    return src.model_copy(update=changes)


def with_bone(rig: RigFingerprint, name: str, **changes: Any) -> RigFingerprint:
    return rig.model_copy(
        update={"bones": [bone(b, **changes) if b.name == name else b for b in rig.bones]}
    )


def shifted_matrix(rig: RigFingerprint, name: str, delta: float) -> RigFingerprint:
    target = next(b for b in rig.bones if b.name == name)
    matrix = [list(row) for row in target.matrix_local]
    matrix[1][3] += delta
    return with_bone(rig, name, matrix_local=matrix)


def shifted_head(rig: RigFingerprint, name: str, delta: float) -> RigFingerprint:
    target = next(b for b in rig.bones if b.name == name)
    x, y, z = target.head
    return with_bone(rig, name, head=(x + delta, y, z))


def extra_bone(rig: RigFingerprint, name: str = "Tail", parent: str = "Root") -> RigFingerprint:
    new = Bone(
        name=name,
        parent=parent,
        head=(0.0, 0.0, 0.0),
        tail=(0.0, 0.0, 0.2),
        matrix_local=[[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0], [0, 0, 0, 1.0]],
        transform_path=f"{rig.armature_object}/{parent}/{name}",
    )
    bones = list(rig.bones)
    bones.insert(1, new)  # depth-first: Root, Tail, Spine, Head (Tail < Spine alphabetically)
    return rig.model_copy(update={"bones": bones, "bone_count": len(bones)})


def without_bone(rig: RigFingerprint, name: str) -> RigFingerprint:
    bones = [b for b in rig.bones if b.name != name]
    return rig.model_copy(update={"bones": bones, "bone_count": len(bones)})


def result(report: RigFidelityReport, check: str) -> Severity:
    return next(c.result for c in report.checks if c.name == check)


def details(report: RigFidelityReport, check: str) -> list[str]:
    return next(c.details for c in report.checks if c.name == check)


def check(src: RigFingerprint, out: RigFingerprint, mode: RfcMode) -> RigFidelityReport:
    return rig_fidelity_check(src, out, TOL, mode)


# --------------------------------------------------------------------------- identical rigs
@pytest.mark.parametrize("mode", MODES)
def test_identical_rigs_pass_every_check(mode: RfcMode) -> None:
    report = check(sample_rig(), sample_rig(), mode)
    assert report.overall == "pass"
    assert report.mode == mode
    assert all(c.result == "pass" and c.details == [] for c in report.checks)
    expected = CHECK_NAMES if mode == "model" else CHECK_NAMES[:-1]
    assert tuple(c.name for c in report.checks) == expected
    assert report.messages("fail") == []
    assert report.messages("warn") == []


def test_default_tolerances_are_normative() -> None:
    assert (TOL.position_abs, TOL.matrix_abs, TOL.allow_extra_actions) == (1e-4, 1e-4, True)


# --------------------------------------------------------------------------- armature object
@pytest.mark.parametrize("mode", MODES)
def test_armature_object_name_mismatch_fails(mode: RfcMode) -> None:
    report = check(sample_rig(), sample_rig("Rig"), mode)
    assert result(report, "armature_object") == "fail"
    assert details(report, "armature_object") == ["armature object 'Rig' != 'Armature'"]
    # transform paths change with the object name too
    assert result(report, "transform_paths") == "fail"
    assert report.overall == "fail"


# --------------------------------------------------------------------------- bone names
@pytest.mark.parametrize("mode", MODES)
def test_missing_bone_fails_in_both_modes(mode: RfcMode) -> None:
    report = check(sample_rig(), without_bone(sample_rig(), "Head"), mode)
    assert result(report, "bone_names") == "fail"
    assert details(report, "bone_names") == ["missing bones: Head"]
    assert report.overall == "fail"
    # the other checks only look at common bones and still pass
    for name in ("parent_map", "depth_first_order", "transform_paths", "matrix_local", "head_tail"):
        assert result(report, name) == "pass", name


def test_extra_bone_fails_in_model_mode_and_warns_in_animation_mode() -> None:
    model = check(sample_rig(), extra_bone(sample_rig()), "model")
    assert result(model, "bone_names") == "fail"
    assert details(model, "bone_names") == ["extra bones: Tail"]
    assert model.overall == "fail"

    anim = check(sample_rig(), extra_bone(sample_rig()), "animation")
    assert result(anim, "bone_names") == "warn"
    assert result(anim, "depth_first_order") == "pass"  # common bones keep their order
    assert anim.overall == "pass_with_warnings"
    assert anim.messages("warn") == ["bone_names: extra bones: Tail"]


def test_missing_and_extra_together_is_a_fail_with_both_details() -> None:
    out = extra_bone(without_bone(sample_rig(), "Head"))
    report = check(sample_rig(), out, "animation")
    assert result(report, "bone_names") == "fail"
    assert details(report, "bone_names") == ["missing bones: Head", "extra bones: Tail"]


# --------------------------------------------------------------------------- parent map / paths
@pytest.mark.parametrize("mode", MODES)
def test_reparented_bone_fails_parent_map_and_paths(mode: RfcMode) -> None:
    out = with_bone(sample_rig(), "Head", parent="Root", transform_path="Armature/Root/Head")
    report = check(sample_rig(), out, mode)
    assert result(report, "parent_map") == "fail"
    assert details(report, "parent_map") == ["Head: parent 'Root' != 'Spine'"]
    assert result(report, "transform_paths") == "fail"
    assert details(report, "transform_paths") == [
        "Head: path 'Armature/Root/Head' != 'Armature/Root/Spine/Head'"
    ]
    assert report.overall == "fail"


# --------------------------------------------------------------------------- order
def test_order_change_fails_model_and_warns_animation() -> None:
    src = sample_rig()
    out = src.model_copy(update={"bones": [src.bones[0], src.bones[2], src.bones[1]]})
    model = check(src, out, "model")
    assert result(model, "depth_first_order") == "fail"
    assert details(model, "depth_first_order") == [
        "depth-first order Root,Head,Spine != Root,Spine,Head"
    ]
    anim = check(src, out, "animation")
    assert result(anim, "depth_first_order") == "warn"
    assert anim.overall == "pass_with_warnings"


# --------------------------------------------------------------------------- matrix_local
@pytest.mark.parametrize(
    ("delta", "expected_model", "expected_anim"),
    [(1e-4, "pass", "pass"), (1.5e-4, "fail", "warn"), (0.5, "fail", "warn")],
)
def test_matrix_local_tolerance_boundary(
    delta: float, expected_model: Severity, expected_anim: Severity
) -> None:
    out = shifted_matrix(sample_rig(), "Spine", delta)
    assert result(check(sample_rig(), out, "model"), "matrix_local") == expected_model
    assert result(check(sample_rig(), out, "animation"), "matrix_local") == expected_anim


def test_matrix_local_detail_names_bone_cell_and_delta() -> None:
    out = shifted_matrix(sample_rig(), "Spine", 2.5e-3)
    report = check(sample_rig(), out, "animation")
    assert details(report, "matrix_local") == [
        "Spine: matrix_local[1][3] delta 2.500e-03 > 1.0e-04"
    ]
    assert report.overall == "pass_with_warnings"
    assert report.messages("warn") == [
        "matrix_local: Spine: matrix_local[1][3] delta 2.500e-03 > 1.0e-04"
    ]


def test_matrix_tolerance_is_configurable() -> None:
    out = shifted_matrix(sample_rig(), "Spine", 1e-3)
    loose = rig_fidelity_check(sample_rig(), out, RfcTolerances(matrix_abs=1e-2), "model")
    assert result(loose, "matrix_local") == "pass"
    strict = rig_fidelity_check(sample_rig(), out, RfcTolerances(matrix_abs=1e-6), "model")
    assert result(strict, "matrix_local") == "fail"


# --------------------------------------------------------------------------- head / tail
@pytest.mark.parametrize(
    ("delta", "expected_model", "expected_anim"),
    [(1e-4, "pass", "pass"), (2e-4, "fail", "warn")],
)
def test_head_tail_tolerance_boundary(
    delta: float, expected_model: Severity, expected_anim: Severity
) -> None:
    out = shifted_head(sample_rig(), "Head", delta)
    assert result(check(sample_rig(), out, "model"), "head_tail") == expected_model
    assert result(check(sample_rig(), out, "animation"), "head_tail") == expected_anim


def test_head_and_tail_are_reported_separately() -> None:
    out = with_bone(sample_rig(), "Root", head=(0.01, 0.0, 0.0), tail=(0.0, 0.0, 0.52))
    report = check(sample_rig(), out, "model")
    assert details(report, "head_tail") == [
        "Root: head delta 1.000e-02 > 1.0e-04",
        "Root: tail delta 2.000e-02 > 1.0e-04",
    ]
    assert result(report, "head_tail") == "fail"
    assert result(report, "matrix_local") == "pass"


# --------------------------------------------------------------------------- actions
def test_missing_source_action_fails_model_mode_only() -> None:
    out = sample_rig().model_copy(update={"actions": []})
    model = check(sample_rig(), out, "model")
    assert result(model, "actions") == "fail"
    assert details(model, "actions") == ["missing actions: idle"]
    assert model.overall == "fail"
    anim = check(sample_rig(), out, "animation")
    assert "actions" not in {c.name for c in anim.checks}
    assert anim.overall == "pass"


def test_actions_compare_prefix_stripped_names() -> None:
    src = sample_rig().model_copy(update={"actions": ["Armature|Armature|idle"]})
    out = sample_rig().model_copy(update={"actions": ["idle"]})
    assert result(check(src, out, "model"), "actions") == "pass"
    assert result(check(out, src, "model"), "actions") == "pass"


def test_extra_actions_pass_by_default_and_warn_when_disallowed() -> None:
    out = sample_rig().model_copy(update={"actions": ["idle", "walk"]})
    assert result(check(sample_rig(), out, "model"), "actions") == "pass"
    strict = rig_fidelity_check(
        sample_rig(), out, RfcTolerances(allow_extra_actions=False), "model"
    )
    assert result(strict, "actions") == "warn"
    assert details(strict, "actions") == ["extra actions: walk"]
    assert strict.overall == "pass_with_warnings"


def test_missing_and_extra_actions_is_a_fail() -> None:
    out = sample_rig().model_copy(update={"actions": ["walk"]})
    strict = rig_fidelity_check(
        sample_rig(), out, RfcTolerances(allow_extra_actions=False), "model"
    )
    assert result(strict, "actions") == "fail"
    assert details(strict, "actions") == ["missing actions: idle", "extra actions: walk"]


# --------------------------------------------------------------------------- rollup
def test_overall_rollup_fail_beats_warn() -> None:
    out = shifted_matrix(extra_bone(sample_rig()), "Spine", 1.0)
    anim = check(sample_rig(), out, "animation")
    assert {result(anim, "bone_names"), result(anim, "matrix_local")} == {"warn"}
    assert anim.overall == "pass_with_warnings"
    worse = check(sample_rig("Rig"), out, "animation")
    assert worse.overall == "fail"
    assert worse.messages("fail")[0].startswith("armature_object: ")
    assert len(worse.messages("warn")) == 2


def test_report_round_trips_through_json() -> None:
    report = check(sample_rig(), shifted_head(sample_rig(), "Head", 1.0), "animation")
    again = RigFidelityReport.model_validate_json(report.model_dump_json())
    assert again == report
    assert again.overall == "pass_with_warnings"
