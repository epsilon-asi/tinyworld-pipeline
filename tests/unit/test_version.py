"""Bootstrap placeholder test: the package imports and exposes a version."""

from __future__ import annotations

import re

import grounded_pipeline


def test_version_is_semver() -> None:
    assert re.fullmatch(r"\d+\.\d+\.\d+", grounded_pipeline.__version__)
