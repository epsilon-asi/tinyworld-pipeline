"""Blender-side job registry.

``entry.py`` dispatches on ``JobRequest.job`` through :data:`JOBS`. Each job module exposes
``run(params: dict) -> (data: dict, warnings: list[str])`` and imports ``bpy`` lazily inside ``run``
so this package imports under plain CPython (unit tests, import-linter).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from . import bake_textures, export_rigged, prepare_model, probe, roundtrip_animation

JobFn = Callable[[dict[str, Any]], tuple[dict[str, Any], list[str]]]
"""``params`` → ``(data, warnings)``. Expected failures raise ``errors.JobFailure``."""

JOBS: dict[str, JobFn] = {
    "probe": probe.run,
    "prepare_model": prepare_model.run,
    "bake_textures": bake_textures.run,
    "export_rigged": export_rigged.run,
    "roundtrip_animation": roundtrip_animation.run,
}
"""Job name → implementation. Later work packages register their jobs here."""
