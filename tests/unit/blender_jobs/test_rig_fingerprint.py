"""``rig_fingerprint``: order stability, path building, hash sensitivity (plan 09 §2.7)."""

from __future__ import annotations

import random
from dataclasses import replace

from fakes.fake_scene import three_bone_chain
from tinyworld_pipeline.blender_jobs.rig_fingerprint import (
    clean_action_name,
    fingerprint_sha256,
    ordered_bones,
    rig_fingerprint,
    transform_paths,
)
from tinyworld_pipeline.blender_jobs.scene_api import BoneInfo

IDENTITY = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]


def bone(name: str, parent: str | None, tx: float = 0.5) -> BoneInfo:
    matrix = [list(row) for row in IDENTITY]
    matrix[0][3] = tx
    return BoneInfo(name, parent, (tx, 0.0, 0.0), (tx, 1.0, 0.0), matrix)


def branching_rig() -> list[BoneInfo]:
    """``Root`` → {``ArmL`` → ``HandL``, ``ArmR``}, plus a second root ``Prop``."""
    return [
        bone("HandL", "ArmL"),
        bone("Prop", None),
        bone("ArmR", "Root"),
        bone("Root", None),
        bone("ArmL", "Root"),
    ]


def test_ordered_bones_depth_first_sorted() -> None:
    assert [b.name for b in ordered_bones(branching_rig())] == [
        "Prop",
        "Root",
        "ArmL",
        "HandL",
        "ArmR",
    ]


def test_order_is_stable_under_shuffled_input() -> None:
    bones = branching_rig()
    reference = rig_fingerprint("Armature", "Armature", bones, ["idle"])
    rng = random.Random(7)  # noqa: S311 - shuffling test input, not cryptography
    for _ in range(10):
        shuffled = list(bones)
        rng.shuffle(shuffled)
        assert rig_fingerprint("Armature", "Armature", shuffled, ["idle"]) == reference


def test_transform_paths_mirror_unity() -> None:
    ordered = ordered_bones(three_bone_chain())
    assert transform_paths("Armature", ordered) == {
        "Root": "Armature/Root",
        "Spine": "Armature/Root/Spine",
        "Head": "Armature/Root/Spine/Head",
    }
    fp = rig_fingerprint("Rig", "RigData", three_bone_chain(), [])
    assert [b.transform_path for b in fp.bones] == [
        "Rig/Root",
        "Rig/Root/Spine",
        "Rig/Root/Spine/Head",
    ]
    assert fp.armature_object == "Rig"
    assert fp.armature_data == "RigData"
    assert fp.bone_count == 3
    assert fp.actions == []
    assert fp.bones[1].head == (0.0, 0.0, 0.5)
    assert fp.bones[1].parent == "Root"


def test_orphan_parent_is_treated_as_root() -> None:
    ordered = ordered_bones([bone("Child", "Missing"), bone("A", None)])
    assert [b.name for b in ordered] == ["A", "Child"]
    assert transform_paths("Arm", ordered)["Child"] == "Arm/Child"


def test_sha_changes_for_matrix_change_above_1e_5() -> None:
    base = three_bone_chain()
    moved = list(base)
    matrix = [list(row) for row in base[1].matrix_local]
    matrix[2][3] += 1e-4
    moved[1] = replace(base[1], matrix_local=matrix)
    assert fingerprint_sha256(ordered_bones(moved)) != fingerprint_sha256(ordered_bones(base))


def test_sha_ignores_matrix_change_below_1e_5_and_negative_zero() -> None:
    base = three_bone_chain()
    jittered = []
    for b in base:
        matrix = [[v + 1e-7 for v in row] for row in b.matrix_local]
        matrix[0][1] = -0.0
        jittered.append(replace(b, matrix_local=matrix))
    assert fingerprint_sha256(ordered_bones(jittered)) == fingerprint_sha256(ordered_bones(base))


def test_sha_is_sensitive_to_names_and_parents() -> None:
    base = fingerprint_sha256(ordered_bones(three_bone_chain()))
    renamed = [replace(b, name="Neck" if b.name == "Head" else b.name) for b in three_bone_chain()]
    reparented = [
        replace(b, parent="Root" if b.name == "Head" else b.parent) for b in three_bone_chain()
    ]
    assert fingerprint_sha256(ordered_bones(renamed)) != base
    assert fingerprint_sha256(ordered_bones(reparented)) != base
    assert len(base) == 64


def test_head_tail_do_not_affect_the_sha_but_appear_in_bones() -> None:
    base = three_bone_chain()
    moved = [replace(b, head=(9.0, 9.0, 9.0)) for b in base]
    assert fingerprint_sha256(ordered_bones(moved)) == fingerprint_sha256(ordered_bones(base))
    assert rig_fingerprint("A", "A", moved, []).bones[0].head == (9.0, 9.0, 9.0)


def test_clean_action_name() -> None:
    assert clean_action_name("Armature|Armature|idle") == "idle"
    assert clean_action_name("walk") == "walk"
    assert clean_action_name("Key|Armature|idle") == "idle"
