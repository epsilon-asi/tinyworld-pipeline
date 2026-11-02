from __future__ import annotations

import re

from tinyworld_pipeline.pipeline.timing import (
    Stopwatch,
    SystemClock,
    SystemSleeper,
    compact_utc_timestamp,
    utc_now_iso,
)

from .conftest import FakeClock

ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$")


def test_system_clock_is_monotonic() -> None:
    clock = SystemClock()
    a = clock.monotonic()
    b = clock.monotonic()
    assert b >= a


def test_system_sleeper_ignores_non_positive_durations() -> None:
    sleeper = SystemSleeper()
    sleeper.sleep(0)
    sleeper.sleep(-1)  # returns immediately


def test_utc_now_iso_format() -> None:
    assert ISO_RE.match(utc_now_iso())


def test_compact_timestamp_is_filesystem_safe() -> None:
    stamp = compact_utc_timestamp()
    assert re.fullmatch(r"\d{8}T\d{12}", stamp)


def test_stopwatch_uses_injected_clock() -> None:
    clock = FakeClock(10.0)
    watch = Stopwatch(clock)
    assert watch.elapsed_s == 0.0
    clock.advance(1.2345)
    assert watch.elapsed_s in {1.234, 1.235}
    assert watch.stop() == watch.elapsed_s
    clock.advance(5)
    assert watch.elapsed_s < 2  # frozen after stop
    assert watch.stop() == watch.elapsed_s  # idempotent
    watch.restart()
    assert watch.elapsed_s == 0.0
    clock.advance(0.5)
    assert watch.elapsed_s == 0.5


def test_stopwatch_defaults_to_system_clock() -> None:
    assert Stopwatch().elapsed_s >= 0.0
