"""Identifier types (plan 06 §1)."""

from __future__ import annotations

import re
from typing import Annotated

from pydantic import StringConstraints

MODEL_ID_PATTERN = r"^[a-z0-9][a-z0-9_]{1,63}$"
RUN_ID_PATTERN = r"^[0-9A-HJKMNP-TV-Z]{26}$"  # Crockford base32 ULID
SHA256_PATTERN = r"^[0-9a-f]{64}$"

ModelId = Annotated[str, StringConstraints(pattern=MODEL_ID_PATTERN)]
RunId = Annotated[str, StringConstraints(pattern=RUN_ID_PATTERN)]
Sha256 = Annotated[str, StringConstraints(pattern=SHA256_PATTERN)]

_MODEL_ID_RE = re.compile(MODEL_ID_PATTERN)


def is_valid_model_id(value: str) -> bool:
    """Return whether ``value`` is a legal ``model_id`` (folder name, file prefix, log key)."""
    return _MODEL_ID_RE.fullmatch(value) is not None
