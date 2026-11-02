"""Fixtures for the ``library/*`` unit tests; builders live in :mod:`helpers`."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from tinyworld_pipeline.library.discovery import Library

from .helpers import FIXTURE_LIBRARY, LibraryBuilder


@pytest.fixture
def builder(tmp_path: Path) -> LibraryBuilder:
    return LibraryBuilder(tmp_path / "lib")


@pytest.fixture
def fixture_root(tmp_path: Path) -> Path:
    dest = tmp_path / "library_minimal"
    shutil.copytree(FIXTURE_LIBRARY, dest)
    return dest


@pytest.fixture
def fixture_library(fixture_root: Path) -> Library:
    return Library.load(fixture_root)
