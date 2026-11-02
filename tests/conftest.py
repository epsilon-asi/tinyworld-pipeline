"""Shared pytest configuration.

Fixtures for fakes (scene, Blender runner, texture provider, clocks) are added by WP-16.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

_DEFAULT_MAC_BLENDER = Path("/Applications/Blender.app/Contents/MacOS/Blender")


def _blender_executable() -> Path | None:
    env = os.environ.get("GP_BLENDER_EXE")
    if env and Path(env).exists():
        return Path(env)
    if _DEFAULT_MAC_BLENDER.exists():
        return _DEFAULT_MAC_BLENDER
    return None


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip Blender and live-Tripo suites automatically when their prerequisites are absent."""
    blender = _blender_executable()
    have_key = bool(os.environ.get("TRIPO_API_KEY"))
    skip_blender = pytest.mark.skip(reason="no Blender executable (set GP_BLENDER_EXE)")
    skip_tripo = pytest.mark.skip(reason="TRIPO_API_KEY not set")
    for item in items:
        if "blender" in item.keywords and blender is None:
            item.add_marker(skip_blender)
        if "tripo_live" in item.keywords and not have_key:
            item.add_marker(skip_tripo)


@pytest.fixture(scope="session")
def blender_exe() -> Path:
    """Path to a real Blender executable; skips the test when none is available."""
    exe = _blender_executable()
    if exe is None:
        pytest.skip("no Blender executable")
    return exe
