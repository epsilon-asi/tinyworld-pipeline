from __future__ import annotations

import threading
from collections.abc import Iterable

import pytest
from structlog.testing import capture_logs

from tinyworld_pipeline.domain.errors import (
    CancelledError,
    TripoTaskFailedError,
    TripoTaskTimeoutError,
)
from tinyworld_pipeline.domain.ports import CancelToken
from tinyworld_pipeline.tripo.models import TaskDetail
from tinyworld_pipeline.tripo.polling import RateLimiter, next_interval, wait_for_task

from .conftest import FakeClock, FakeSleeper, load_fixture


def _detail(name: str) -> TaskDetail:
    return TaskDetail.model_validate(load_fixture(name)["data"])


class ScriptedGetter:
    """``get_task`` that replays a sequence and then repeats the last detail forever."""

    def __init__(self, names: Iterable[str]) -> None:
        self._details = [_detail(n) for n in names]
        self.calls = 0

    def get_task(self, task_id: str) -> TaskDetail:
        self.calls += 1
        index = min(self.calls - 1, len(self._details) - 1)
        return self._details[index]


# ------------------------------------------------------------------------------------ RateLimiter
def test_pace_enforces_min_spacing_with_constant_clock() -> None:
    clock = FakeClock()
    sleeper = FakeSleeper()  # does not advance the clock
    limiter = RateLimiter(2, 0.25, clock=clock, sleeper=sleeper)
    limiter.pace()
    limiter.pace()
    limiter.pace()
    assert sleeper.calls == [0.25, 0.5]


def test_pace_with_advancing_clock_sleeps_only_the_gap() -> None:
    clock = FakeClock()
    sleeper = FakeSleeper(clock)
    limiter = RateLimiter(2, 1.0, clock=clock, sleeper=sleeper)
    limiter.pace()
    clock.advance(0.4)
    limiter.pace()
    assert sleeper.calls == [pytest.approx(0.6)]
    clock.advance(5)
    limiter.pace()  # long idle: no wait
    assert len(sleeper.calls) == 1


def test_pace_disabled_when_interval_zero() -> None:
    sleeper = FakeSleeper()
    limiter = RateLimiter(1, 0.0, clock=FakeClock(), sleeper=sleeper)
    for _ in range(3):
        limiter.pace()
    assert sleeper.calls == []


def test_slot_cap_blocks_second_thread_until_release() -> None:
    limiter = RateLimiter(1, 0.0, clock=FakeClock(), sleeper=FakeSleeper())
    assert limiter.max_concurrent == 1
    acquired = threading.Event()
    order: list[str] = []

    def second() -> None:
        with limiter.slot():
            order.append("second")
        acquired.set()

    limiter.acquire_slot()
    order.append("first")
    thread = threading.Thread(target=second)
    thread.start()
    assert not acquired.wait(0.1)  # blocked while we hold the only slot
    limiter.release_slot()
    assert acquired.wait(2.0)
    thread.join()
    assert order == ["first", "second"]


def test_acquire_slot_timeout_and_over_release() -> None:
    limiter = RateLimiter(1, 0.0, clock=FakeClock(), sleeper=FakeSleeper())
    assert limiter.acquire_slot(timeout_s=0.01) is True
    assert limiter.acquire_slot(timeout_s=0.01) is False
    limiter.release_slot()
    with pytest.raises(ValueError, match="released too many times"):
        limiter.release_slot()


def test_rate_limiter_rejects_bad_arguments() -> None:
    with pytest.raises(ValueError, match="max_concurrent"):
        RateLimiter(0, 0.1, clock=FakeClock(), sleeper=FakeSleeper())
    with pytest.raises(ValueError, match="min_interval_s"):
        RateLimiter(1, -0.1, clock=FakeClock(), sleeper=FakeSleeper())


# ---------------------------------------------------------------------------------- next_interval
def test_next_interval_schedule() -> None:
    assert next_interval(2.0, 0.0) == 2.0
    assert next_interval(2.0, 59.9) == 2.0
    assert next_interval(2.0, 60.0) == 3.0
    assert next_interval(6.75, 100.0) == 10.0
    assert next_interval(10.0, 500.0) == 10.0


# --------------------------------------------------------------------------------- wait_for_task
def _wait(
    getter: ScriptedGetter,
    *,
    interval_s: float = 2.0,
    timeout_s: float = 900.0,
    cancel: CancelToken | None = None,
    on_progress: list[TaskDetail] | None = None,
    clock: FakeClock | None = None,
    sleeper: FakeSleeper | None = None,
) -> TaskDetail:
    clock = clock or FakeClock()
    sleeper = sleeper or FakeSleeper(clock)
    return wait_for_task(
        getter,
        "task_x",
        interval_s=interval_s,
        timeout_s=timeout_s,
        cancel=cancel or CancelToken(),
        clock=clock,
        sleeper=sleeper,
        on_progress=on_progress.append if on_progress is not None else None,
    )


def test_queued_running_success_reports_progress_and_sleeps_between_polls() -> None:
    getter = ScriptedGetter(["task_queued", "task_running", "task_success"])
    seen: list[TaskDetail] = []
    clock = FakeClock()
    sleeper = FakeSleeper(clock)
    with capture_logs() as logs:
        detail = _wait(getter, on_progress=seen, clock=clock, sleeper=sleeper)
    assert detail.is_success
    assert detail.output is not None
    assert detail.output.normalize_scale == 0.5
    assert [d.status for d in seen] == ["queued", "running", "success"]
    assert sleeper.calls == [2.0, 2.0]
    assert getter.calls == 3
    assert [e["event"] for e in logs].count("tripo.poll") == 3
    assert all(e["task_id"] == "task_x" for e in logs)


@pytest.mark.parametrize("name", ["task_failed", "task_banned", "task_expired"])
def test_terminal_failure_statuses_raise_task_failed(name: str) -> None:
    getter = ScriptedGetter([name])
    with pytest.raises(TripoTaskFailedError) as info:
        _wait(getter)
    detail = _detail(name)
    assert info.value.detail == {
        "status": detail.status,
        "error_code": detail.error_code,
        "error_message": detail.error_message,
    }
    assert detail.status in str(info.value)
    if detail.error_message:
        assert detail.error_message in str(info.value)


def test_unknown_status_keeps_polling() -> None:
    getter = ScriptedGetter(["task_unknown", "task_unknown", "task_success"])
    detail = _wait(getter)
    assert detail.is_success
    assert getter.calls == 3


def test_timeout_raises_task_timeout_and_never_oversleeps() -> None:
    getter = ScriptedGetter(["task_running"])
    clock = FakeClock()
    sleeper = FakeSleeper(clock)
    with pytest.raises(TripoTaskTimeoutError) as info:
        _wait(getter, interval_s=2.0, timeout_s=5.0, clock=clock, sleeper=sleeper)
    assert sleeper.calls == [2.0, 2.0, 1.0]
    assert info.value.detail == {"status": "running", "timeout_s": 5.0, "polls": 4}
    assert info.value.hint is not None


def test_cancel_mid_poll_raises_cancelled() -> None:
    getter = ScriptedGetter(["task_queued", "task_running", "task_success"])
    cancel = CancelToken()
    seen: list[TaskDetail] = []

    def on_progress(detail: TaskDetail) -> None:
        seen.append(detail)
        if len(seen) == 2:
            cancel.set()

    clock = FakeClock()
    sleeper = FakeSleeper(clock)
    with pytest.raises(CancelledError) as info:
        wait_for_task(
            getter,
            "task_x",
            interval_s=2.0,
            timeout_s=900.0,
            cancel=cancel,
            clock=clock,
            sleeper=sleeper,
            on_progress=on_progress,
        )
    assert info.value.exit_code == 130
    assert getter.calls == 2  # the third poll never happened


def test_cancel_before_first_poll() -> None:
    cancel = CancelToken()
    cancel.set()
    getter = ScriptedGetter(["task_success"])
    with pytest.raises(CancelledError):
        _wait(getter, cancel=cancel)
    assert getter.calls == 0


def test_backoff_schedule_grows_after_sixty_seconds_up_to_ten() -> None:
    getter = ScriptedGetter(["task_running"])
    clock = FakeClock()
    sleeper = FakeSleeper(clock)
    with pytest.raises(TripoTaskTimeoutError):
        _wait(getter, interval_s=2.0, timeout_s=120.0, clock=clock, sleeper=sleeper)
    # 30 polls at 2 s cover the first minute, then x1.5 per poll capped at 10 s.
    assert sleeper.calls[:30] == [2.0] * 30
    assert sleeper.calls[30:35] == [3.0, 4.5, 6.75, 10.0, 10.0]
    assert sum(sleeper.calls) == pytest.approx(120.0)
    assert max(sleeper.calls) == 10.0


def test_wait_for_task_rejects_non_positive_interval() -> None:
    with pytest.raises(ValueError, match="interval_s"):
        _wait(ScriptedGetter(["task_success"]), interval_s=0)
