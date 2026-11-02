"""Rig Fidelity Check (plan 12 §1.4) — pure comparison of two rig fingerprints.

``src`` is the rig recorded by ``prepare`` (the vendor model as imported); ``out`` is the rig
reported by ``export_rigged`` (mode ``model``) or by ``roundtrip_animation`` for a companion clip
(mode ``animation``). Each check yields ``pass`` | ``warn`` | ``fail`` with human-readable details
(bone name + delta); the overall verdict rolls up to ``pass`` | ``pass_with_warnings`` | ``fail``.
The caller (``pipeline/stages/export.py``) turns ``fail`` into ``RigFidelityError``.

Severity table (plan 12 §1.4):

| check                | model | animation                                    |
|----------------------|-------|----------------------------------------------|
| ``armature_object``  | fail  | fail                                         |
| ``bone_names``       | fail  | missing → fail, extra → warn                 |
| ``parent_map``       | fail  | fail (common bones)                          |
| ``depth_first_order``| fail  | warn (common bones)                          |
| ``transform_paths``  | fail  | fail (common bones)                          |
| ``matrix_local``     | fail  | warn                                         |
| ``head_tail``        | fail  | warn                                         |
| ``actions``          | fail  | not checked                                  |

The ``actions`` check compares import-prefix-stripped names (appendix B §8.1): every source
action must be present in the output; extra output actions pass unless
``RfcTolerances.allow_extra_actions`` is off, in which case they warn.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from pydantic import Field

from grounded_pipeline.blender.protocol import Bone, RigFingerprint
from grounded_pipeline.domain._base import FrozenModel

Severity = Literal["pass", "warn", "fail"]
RfcMode = Literal["model", "animation"]
Overall = Literal["pass", "pass_with_warnings", "fail"]

CHECK_NAMES: tuple[str, ...] = (
    "armature_object",
    "bone_names",
    "parent_map",
    "depth_first_order",
    "transform_paths",
    "matrix_local",
    "head_tail",
    "actions",
)
"""Check names in report order; ``actions`` is omitted in ``animation`` mode."""

_ON_MISMATCH: dict[str, dict[RfcMode, Severity]] = {
    "armature_object": {"model": "fail", "animation": "fail"},
    "parent_map": {"model": "fail", "animation": "fail"},
    "depth_first_order": {"model": "fail", "animation": "warn"},
    "transform_paths": {"model": "fail", "animation": "fail"},
    "matrix_local": {"model": "fail", "animation": "warn"},
    "head_tail": {"model": "fail", "animation": "warn"},
}
_MATRIX_SIZE = 4


@dataclass(frozen=True)
class RfcTolerances:
    """Numeric tolerances of the check (plan 12 §1.4, NORMATIVE defaults)."""

    position_abs: float = 1e-4
    """Max absolute deviation of bone head/tail coordinates (model units)."""
    matrix_abs: float = 1e-4
    """Max absolute deviation per element of ``matrix_local``."""
    allow_extra_actions: bool = True
    """Output actions the source lacks are fine (``True``) or a warning (``False``)."""


class RfcCheck(FrozenModel):
    """Outcome of one check."""

    name: str
    result: Severity
    details: list[str] = Field(default_factory=list)


class RigFidelityReport(FrozenModel):
    """All checks of one comparison plus the rolled-up verdict."""

    mode: RfcMode
    overall: Overall
    checks: list[RfcCheck]

    def messages(self, severity: Severity) -> list[str]:
        """``"<check>: <detail>"`` lines for every check with ``severity``."""
        return [
            f"{check.name}: {detail}"
            for check in self.checks
            if check.result == severity
            for detail in (check.details or ["mismatch"])
        ]


def rig_fidelity_check(
    src: RigFingerprint,
    out: RigFingerprint,
    tol: RfcTolerances,
    mode: RfcMode = "model",
) -> RigFidelityReport:
    """Compare ``out`` against ``src`` (see the module docstring for the severity table).

    Args:
        src: Source rig (``prepare_report.rig``).
        out: Exported model rig or round-tripped clip rig.
        tol: Numeric tolerances; comparisons are inclusive (``delta <= tol`` passes).
        mode: ``model`` (strict) or ``animation`` (lenient on order/transforms, extra bones).

    Returns:
        The report; ``overall == "fail"`` when any check failed.
    """
    src_bones = {b.name: b for b in src.bones}
    out_bones = {b.name: b for b in out.bones}
    common = [b.name for b in src.bones if b.name in out_bones]
    checks = [
        _check("armature_object", mode, _armature_object(src, out)),
        _bone_names(src_bones, out_bones, mode),
        _check("parent_map", mode, _parent_map(src_bones, out_bones, common)),
        _check("depth_first_order", mode, _order(src, out, set(common))),
        _check("transform_paths", mode, _paths(src_bones, out_bones, common)),
        _check("matrix_local", mode, _matrices(src_bones, out_bones, common, tol.matrix_abs)),
        _check("head_tail", mode, _positions(src_bones, out_bones, common, tol.position_abs)),
    ]
    if mode == "model":
        checks.append(_actions(src.actions, out.actions, tol.allow_extra_actions))
    return RigFidelityReport(mode=mode, overall=_overall(checks), checks=checks)


# --------------------------------------------------------------------------- checks
def _check(name: str, mode: RfcMode, details: list[str]) -> RfcCheck:
    result: Severity = "pass" if not details else _ON_MISMATCH[name][mode]
    return RfcCheck(name=name, result=result, details=details)


def _armature_object(src: RigFingerprint, out: RigFingerprint) -> list[str]:
    if src.armature_object == out.armature_object:
        return []
    return [f"armature object {out.armature_object!r} != {src.armature_object!r}"]


def _bone_names(src: dict[str, Bone], out: dict[str, Bone], mode: RfcMode) -> RfcCheck:
    missing = sorted(set(src) - set(out))
    extra = sorted(set(out) - set(src))
    details: list[str] = []
    if missing:
        details.append("missing bones: " + ", ".join(missing))
    if extra:
        details.append("extra bones: " + ", ".join(extra))
    result: Severity = "pass"
    if missing or (extra and mode == "model"):
        result = "fail"
    elif extra:
        result = "warn"
    return RfcCheck(name="bone_names", result=result, details=details)


def _parent_map(src: dict[str, Bone], out: dict[str, Bone], common: Iterable[str]) -> list[str]:
    return [
        f"{name}: parent {out[name].parent!r} != {src[name].parent!r}"
        for name in common
        if out[name].parent != src[name].parent
    ]


def _order(src: RigFingerprint, out: RigFingerprint, common: set[str]) -> list[str]:
    src_order = [b.name for b in src.bones if b.name in common]
    out_order = [b.name for b in out.bones if b.name in common]
    if src_order == out_order:
        return []
    return [f"depth-first order {','.join(out_order)} != {','.join(src_order)}"]


def _paths(src: dict[str, Bone], out: dict[str, Bone], common: Iterable[str]) -> list[str]:
    return [
        f"{name}: path {out[name].transform_path!r} != {src[name].transform_path!r}"
        for name in common
        if out[name].transform_path != src[name].transform_path
    ]


def _matrices(
    src: dict[str, Bone], out: dict[str, Bone], common: Iterable[str], tol: float
) -> list[str]:
    details: list[str] = []
    for name in common:
        worst, where = 0.0, (0, 0)
        for row in range(_MATRIX_SIZE):
            for col in range(_MATRIX_SIZE):
                delta = abs(out[name].matrix_local[row][col] - src[name].matrix_local[row][col])
                if delta > worst:
                    worst, where = delta, (row, col)
        if worst > tol:
            details.append(
                f"{name}: matrix_local[{where[0]}][{where[1]}] delta {worst:.3e} > {tol:.1e}"
            )
    return details


def _positions(
    src: dict[str, Bone], out: dict[str, Bone], common: Iterable[str], tol: float
) -> list[str]:
    details: list[str] = []
    for name in common:
        for label, a, b in (
            ("head", src[name].head, out[name].head),
            ("tail", src[name].tail, out[name].tail),
        ):
            delta = max(abs(x - y) for x, y in zip(a, b, strict=True))
            if delta > tol:
                details.append(f"{name}: {label} delta {delta:.3e} > {tol:.1e}")
    return details


def _actions(src: list[str], out: list[str], allow_extra: bool) -> RfcCheck:
    src_names = {_strip(a) for a in src}
    out_names = {_strip(a) for a in out}
    missing = sorted(src_names - out_names)
    extra = sorted(out_names - src_names)
    details: list[str] = []
    result: Severity = "pass"
    if missing:
        details.append("missing actions: " + ", ".join(missing))
        result = "fail"
    if extra and not allow_extra:
        details.append("extra actions: " + ", ".join(extra))
        if result == "pass":
            result = "warn"
    return RfcCheck(name="actions", result=result, details=details)


def _strip(action: str) -> str:
    """``Armature|Armature|idle`` → ``idle`` (appendix B §8.1); already-clean names pass through."""
    return action.rsplit("|", 1)[-1]


def _overall(checks: list[RfcCheck]) -> Overall:
    results = {c.result for c in checks}
    if "fail" in results:
        return "fail"
    if "warn" in results:
        return "pass_with_warnings"
    return "pass"


__all__ = [
    "CHECK_NAMES",
    "Overall",
    "RfcCheck",
    "RfcMode",
    "RfcTolerances",
    "RigFidelityReport",
    "Severity",
    "rig_fidelity_check",
]
