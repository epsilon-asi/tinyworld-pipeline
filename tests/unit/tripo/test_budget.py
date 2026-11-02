from __future__ import annotations

import threading

import pytest

from grounded_pipeline.domain.errors import BudgetExceededError, ConfigError
from grounded_pipeline.domain.library_config import TripoSection
from grounded_pipeline.tripo.budget import CreditBudget, Reservation, estimate_credits

TABLE = TripoSection().credit_table


def test_reserve_commit_release_cycle() -> None:
    budget = CreditBudget(100)
    assert budget.limit == 100
    first = budget.reserve(10)
    second = budget.reserve(20)
    assert budget.reserved == 30
    assert budget.spent == 0
    assert budget.remaining == 70

    budget.commit(first, 12.5)  # actual replaces the estimate
    assert budget.spent == 12.5
    assert budget.reserved == 20

    budget.release(second)
    assert budget.reserved == 0
    assert budget.spent == 12.5
    assert budget.remaining == 87.5


def test_reserve_raises_when_total_would_exceed_limit() -> None:
    budget = CreditBudget(30)
    budget.commit(budget.reserve(10), 10)
    budget.reserve(10)
    with pytest.raises(BudgetExceededError) as info:
        budget.reserve(11)
    err = info.value
    assert err.exit_code == 5
    assert err.detail == {"estimate": 11, "spent": 10.0, "reserved": 10.0, "limit": 30.0}
    assert err.hint
    budget.reserve(10)  # exactly reaching the limit is allowed


@pytest.mark.parametrize("limit", [None, 0, 0.0])
def test_none_or_zero_limit_is_unlimited(limit: float | None) -> None:
    budget = CreditBudget(limit)
    assert budget.limit is None
    assert budget.remaining is None
    ticket = budget.reserve(1_000_000)
    budget.commit(ticket, 999_999)
    assert budget.spent == 999_999


def test_invalid_amounts_and_double_settlement() -> None:
    budget = CreditBudget(10)
    with pytest.raises(ValueError, match="estimate"):
        budget.reserve(-1)
    ticket = budget.reserve(5)
    with pytest.raises(ValueError, match="actual"):
        budget.commit(ticket, -1)
    budget.release(ticket)
    with pytest.raises(ValueError, match="already settled"):
        budget.release(ticket)
    with pytest.raises(ValueError, match="already settled"):
        budget.commit(ticket, 1)
    with pytest.raises(ValueError, match="unknown"):
        budget.commit(Reservation(id=999, estimate=1), 1)


def test_concurrent_reservations_never_exceed_limit() -> None:
    budget = CreditBudget(100)
    granted: list[Reservation] = []
    lock = threading.Lock()
    start = threading.Barrier(2)

    def worker() -> None:
        start.wait()
        for _ in range(60):
            try:
                ticket = budget.reserve(3)
            except BudgetExceededError:
                continue
            with lock:
                granted.append(ticket)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(granted) == 33  # floor(100 / 3)
    assert budget.reserved == pytest.approx(99)
    assert budget.spent == 0
    for ticket in granted:
        budget.commit(ticket, 3)
    assert budget.spent == pytest.approx(99)
    assert budget.reserved == 0


def test_estimate_credits_uses_table_and_style_surcharge() -> None:
    assert estimate_credits("standard", style_image=False, table=TABLE) == 10
    assert estimate_credits("fast", style_image=False, table=TABLE) == 10
    assert estimate_credits("detailed", style_image=False, table=TABLE) == 20
    assert estimate_credits("extreme", style_image=True, table=TABLE) == 30  # default surcharge 0
    custom = {"standard": 10.0, "style_image": 5.0}
    assert estimate_credits("standard", style_image=True, table=custom) == 15
    assert estimate_credits("standard", style_image=True, table={"standard": 4.0}) == 4


def test_estimate_credits_unknown_quality_is_a_config_error() -> None:
    with pytest.raises(ConfigError) as info:
        estimate_credits("ultra", style_image=False, table=TABLE)
    assert info.value.detail["known"] == ["detailed", "extreme", "fast", "standard"]
    assert info.value.hint is not None
