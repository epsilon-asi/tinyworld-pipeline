"""Rig fingerprint (plan 09 §2.7). Pure: works on ``BoneInfo`` lists, never on ``bpy``.

The same function serves ``prepare_model`` (source fingerprint, before/after guard),
``export_rigged`` and ``roundtrip_animation`` (rig fidelity check, plan 12 §1.4).
"""

from __future__ import annotations

import hashlib

from .protocol import Bone, RigFingerprint
from .scene_api import BoneInfo

MATRIX_DECIMALS = 5
"""``matrix_local`` elements are rounded to 1e-5 before hashing (float32 export noise)."""


def clean_action_name(name: str) -> str:
    """Strip FBX import prefixes: ``Armature|Armature|idle`` → ``idle`` (appendix B §8.1)."""
    return name.rsplit("|", 1)[-1]


def ordered_bones(bones: list[BoneInfo]) -> list[BoneInfo]:
    """Depth-first order: root bones sorted by name, children sorted by name.

    A bone whose parent is not in ``bones`` is treated as a root (defensive; cannot happen for
    a bone collection read from one armature).
    """
    by_name = {bone.name: bone for bone in bones}
    children: dict[str | None, list[BoneInfo]] = {}
    for bone in bones:
        parent = bone.parent if bone.parent in by_name else None
        children.setdefault(parent, []).append(bone)
    out: list[BoneInfo] = []
    stack = sorted(children.get(None, []), key=lambda b: b.name, reverse=True)
    while stack:
        bone = stack.pop()
        out.append(bone)
        stack.extend(sorted(children.get(bone.name, []), key=lambda b: b.name, reverse=True))
    return out


def transform_paths(armature_object: str, ordered: list[BoneInfo]) -> dict[str, str]:
    """Unity-style ``Armature/Root/Spine`` path per bone (``ordered`` must be parent-first)."""
    paths: dict[str, str] = {}
    for bone in ordered:
        parent_path = paths.get(bone.parent or "", armature_object)
        paths[bone.name] = f"{parent_path}/{bone.name}"
    return paths


def fingerprint_sha256(ordered: list[BoneInfo]) -> str:
    """SHA-256 over ``name|parent|`` + ``matrix_local`` rounded to 1e-5, one line per bone."""
    digest = hashlib.sha256()
    for bone in ordered:
        cells = ",".join(_fmt(value) for row in bone.matrix_local for value in row)
        digest.update(f"{bone.name}|{bone.parent or ''}|{cells}\n".encode())
    return digest.hexdigest()


def rig_fingerprint(
    armature_object: str, armature_data: str, bones: list[BoneInfo], actions: list[str]
) -> RigFingerprint:
    """Build the :class:`RigFingerprint` contract value.

    Args:
        armature_object: Armature *object* name (first path component).
        armature_data: Armature *data-block* name.
        bones: Bones in any order.
        actions: Object-level action names (already prefix-stripped by the scene).

    Returns:
        The fingerprint with bones in depth-first, name-sorted order.
    """
    ordered = ordered_bones(bones)
    paths = transform_paths(armature_object, ordered)
    return RigFingerprint(
        armature_object=armature_object,
        armature_data=armature_data,
        bone_count=len(ordered),
        bones=[
            Bone(
                name=b.name,
                parent=b.parent,
                head=b.head,
                tail=b.tail,
                matrix_local=[[float(v) for v in row] for row in b.matrix_local],
                transform_path=paths[b.name],
            )
            for b in ordered
        ],
        actions=list(actions),
        fingerprint_sha256=fingerprint_sha256(ordered),
    )


def _fmt(value: float) -> str:
    return f"{round(value, MATRIX_DECIMALS) + 0.0:.{MATRIX_DECIMALS}f}"  # "+ 0.0" folds -0.0
