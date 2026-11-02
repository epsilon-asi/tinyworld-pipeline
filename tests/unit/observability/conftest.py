"""Fixtures isolating structlog / stdlib logging state between observability tests."""

from __future__ import annotations

import logging
from collections.abc import Iterator

import pytest
import structlog

from tinyworld_pipeline.observability.logging import THIRD_PARTY_LOGGERS, get_logging_handle

RUN_ID = "01HZX3V6Y2K8Q4M9N7P5R1S0TA"
SECRET = "sk-live-0123456789abcdwxyz"


@pytest.fixture(autouse=True)
def _isolate_logging() -> Iterator[None]:
    """Close the active handle and reset structlog, contextvars and logger levels afterwards."""
    saved_levels = {name: logging.getLogger(name).level for name in THIRD_PARTY_LOGGERS}
    root_level = logging.getLogger().level
    structlog.contextvars.clear_contextvars()
    # A test elsewhere in this worker process (e.g. the CLI suite) may have left a handle open;
    # close it first so handler-count assertions start from a clean root logger.
    stale = get_logging_handle()
    if stale is not None:
        stale.close()
    yield
    handle = get_logging_handle()
    if handle is not None:
        handle.close()
    structlog.reset_defaults()
    structlog.contextvars.clear_contextvars()
    for name, level in saved_levels.items():
        logging.getLogger(name).setLevel(level)
    logging.getLogger().setLevel(root_level)
