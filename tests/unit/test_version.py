"""Bootstrap placeholder test: the package imports and exposes a version."""

from __future__ import annotations

import re

import tinyworld_pipeline


def test_version_is_semver() -> None:
    assert re.fullmatch(r"\d+\.\d+\.\d+", tinyworld_pipeline.__version__)
