"""Built-in defaults (layer 1 of plan 07 §1).

The pydantic field defaults on the section models are the single source of truth; this module only
exposes a ready-made instance and the ``library.toml`` section order.
"""

from __future__ import annotations

from tinyworld_pipeline.domain.library_config import SECTION_MODELS, LibraryConfig

DEFAULTS = LibraryConfig()
SECTION_NAMES: tuple[str, ...] = tuple(SECTION_MODELS)
