"""Per-run credit budget and the configurable credit estimate table (plan 10 §2, §5; appendix A §6).

The budget is pessimistic: a texture task first *reserves* its estimate, then either *commits*
the real ``credits_consumed`` reported by Tripo or *releases* the reservation when the task
failed before billing. ``spent + reserved`` can never exceed ``limit``.
"""

from __future__ import annotations

import itertools
import threading
from collections.abc import Mapping
from dataclasses import dataclass

from grounded_pipeline.domain.errors import BudgetExceededError, ConfigError

STYLE_IMAGE_KEY = "style_image"


@dataclass(frozen=True)
class Reservation:
    """A ticket returned by :meth:`CreditBudget.reserve`; pass it back to commit or release."""

    id: int
    estimate: float


class CreditBudget:
    """Thread-safe credit ledger for one run.

    Args:
        limit: Maximum credits the run may spend (``spent + reserved``). ``None`` or ``0`` means
            unlimited (``TripoSection.max_credits_per_run`` defaults to ``0``).
    """

    def __init__(self, limit: float | None) -> None:
        self._limit: float | None = None if not limit else float(limit)
        self._lock = threading.Lock()
        self._spent = 0.0
        self._open: dict[int, float] = {}
        self._ids = itertools.count(1)

    @property
    def limit(self) -> float | None:
        """The configured limit, or ``None`` when unlimited."""
        return self._limit

    @property
    def spent(self) -> float:
        """Credits committed so far (sum of actual ``credits_consumed``)."""
        with self._lock:
            return self._spent

    @property
    def reserved(self) -> float:
        """Credits currently reserved by in-flight tasks."""
        with self._lock:
            return sum(self._open.values())

    @property
    def remaining(self) -> float | None:
        """``limit - spent - reserved``, or ``None`` when unlimited."""
        with self._lock:
            if self._limit is None:
                return None
            return self._limit - self._spent - sum(self._open.values())

    def reserve(self, estimate: float) -> Reservation:
        """Reserve ``estimate`` credits.

        Raises:
            BudgetExceededError: When ``spent + reserved + estimate`` would exceed the limit.
            ValueError: For a negative estimate.
        """
        if estimate < 0:
            msg = f"estimate must be >= 0, got {estimate}"
            raise ValueError(msg)
        with self._lock:
            open_total = sum(self._open.values())
            if self._limit is not None and self._spent + open_total + estimate > self._limit:
                raise BudgetExceededError(
                    f"credit budget exceeded: need {estimate:g}, "
                    f"spent {self._spent:g} + reserved {open_total:g} of limit {self._limit:g}",
                    detail={
                        "estimate": estimate,
                        "spent": self._spent,
                        "reserved": open_total,
                        "limit": self._limit,
                    },
                    hint="Raise [tripo].max_credits_per_run or run fewer models.",
                )
            ticket = Reservation(id=next(self._ids), estimate=float(estimate))
            self._open[ticket.id] = ticket.estimate
            return ticket

    def commit(self, reservation: Reservation, actual: float) -> None:
        """Replace ``reservation`` with the real ``actual`` credits consumed.

        Raises:
            ValueError: When the reservation is unknown (already committed/released) or
                ``actual`` is negative.
        """
        if actual < 0:
            msg = f"actual must be >= 0, got {actual}"
            raise ValueError(msg)
        with self._lock:
            self._pop(reservation)
            self._spent += float(actual)

    def release(self, reservation: Reservation) -> None:
        """Drop ``reservation`` without spending (task failed before billing).

        Raises:
            ValueError: When the reservation is unknown (already committed/released).
        """
        with self._lock:
            self._pop(reservation)

    def _pop(self, reservation: Reservation) -> None:
        if reservation.id not in self._open:
            msg = f"unknown or already settled reservation #{reservation.id}"
            raise ValueError(msg)
        del self._open[reservation.id]


def estimate_credits(quality: str, *, style_image: bool, table: Mapping[str, float]) -> float:
    """Estimate the credits of one texture task from the configurable credit table.

    Args:
        quality: ``texture_quality`` value (``fast``/``standard``/``detailed``/``extreme``).
        style_image: Whether a style image accompanies the prompt (adds ``table["style_image"]``,
            ``0`` when absent from the table).
        table: ``[tripo.credit_table]`` (defaults: fast 10, standard 10, detailed 20, extreme 30,
            style_image 0).

    Returns:
        Estimated credits (never negative).

    Raises:
        ConfigError: When ``quality`` has no entry in the table.
    """
    if quality not in table:
        raise ConfigError(
            f"no credit estimate for texture quality {quality!r}",
            detail={"quality": quality, "known": sorted(k for k in table if k != STYLE_IMAGE_KEY)},
            hint=f"Add [tripo.credit_table].{quality} to library.toml.",
        )
    total = float(table[quality])
    if style_image:
        total += float(table.get(STYLE_IMAGE_KEY, 0.0))
    return max(0.0, total)


__all__ = ["STYLE_IMAGE_KEY", "CreditBudget", "Reservation", "estimate_credits"]
