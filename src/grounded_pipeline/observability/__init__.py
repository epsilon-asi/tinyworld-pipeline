"""Observability: structlog configuration, secret redaction, console output, log retention.

Importable from every layer; depends on nothing internal except :mod:`grounded_pipeline.domain`
(plan 03 §2). Submodules: :mod:`.events`, :mod:`.logging`, :mod:`.console`, :mod:`.pruning`.
"""

from __future__ import annotations
