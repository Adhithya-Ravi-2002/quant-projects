"""Avellaneda-Stoikov reservation price and quotes, in price space and logit space.

Units: prices are cents on (0, 100); time_remaining is in seconds and sigma is
volatility per sqrt(second) in the model's price space (cents, or logit units).
gamma must be calibrated for the same time scaling: the inventory penalty
q * gamma * sigma^2 * tau grows linearly with the horizon, so a gamma tuned on a
short horizon produces a very large skew when tau is days to resolution.

The `mid` input may be the plain mid-price or the micro-price.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class Quote:
    """Continuous (unrounded) quote in cents."""

    bid: float
    ask: float
    reservation: float


@dataclass(frozen=True, slots=True)
class TickQuote:
    """Quote rounded to the tick grid, in cents."""

    bid: int
    ask: int


class QuotingModel(Protocol):
    def quote(
        self, mid: float, inventory: float, sigma: float, time_remaining: float
    ) -> Quote: ...


def _as_quote(
    s: float, inventory: float, gamma: float, k: float, sigma: float, tau: float
) -> tuple[float, float, float]:
    risk = gamma * sigma**2 * tau
    reservation = s - inventory * risk
    half_spread = (risk + (2 / gamma) * math.log1p(gamma / k)) / 2
    return reservation, reservation - half_spread, reservation + half_spread


class _ASModel:
    def __init__(self, gamma: float, k: float) -> None:
        # Comparisons are written as `not x > 0` so that NaN is rejected too.
        if not gamma > 0:
            raise ValueError(f"gamma must be > 0, got {gamma}")
        if not k > 0:
            raise ValueError(f"k must be > 0, got {k}")
        self.gamma = gamma
        self.k = k

    def quote(
        self, mid: float, inventory: float, sigma: float, time_remaining: float
    ) -> Quote:
        if not 0 < mid < 100:
            raise ValueError(f"mid must be in (0, 100), got {mid}")
        if not math.isfinite(inventory):
            raise ValueError(f"inventory must be finite, got {inventory}")
        if not sigma >= 0:
            raise ValueError(f"sigma must be >= 0, got {sigma}")
        if not time_remaining >= 0:
            raise ValueError(f"time_remaining must be >= 0, got {time_remaining}")
        reservation, bid, ask = _as_quote(
            self._to_space(mid), inventory, self.gamma, self.k, sigma, time_remaining
        )
        return Quote(
            bid=self._to_cents(bid),
            ask=self._to_cents(ask),
            reservation=self._to_cents(reservation),
        )

    def _to_space(self, cents: float) -> float:
        raise NotImplementedError

    def _to_cents(self, value: float) -> float:
        raise NotImplementedError


class PriceSpaceAS(_ASModel):
    """A-S applied directly to cents. sigma is in cents, k in 1/cents.

    Quotes are not bounded to (0, 100); near the edges they can leave it.
    """

    def _to_space(self, cents: float) -> float:
        return cents

    def _to_cents(self, value: float) -> float:
        return value


class LogitSpaceAS(_ASModel):
    """A-S applied to x = ln(p / (1 - p)), p = cents / 100; quotes mapped back by sigmoid.

    sigma, gamma and k are in logit units and are not interchangeable with
    PriceSpaceAS parameters. Quotes stay inside (0, 100) except under extreme
    saturation, where the sigmoid rounds to 0 or 1 in floating point.
    """

    def _to_space(self, cents: float) -> float:
        p = cents / 100
        return math.log(p / (1 - p))

    def _to_cents(self, value: float) -> float:
        return 100 * _sigmoid(value)


def _sigmoid(x: float) -> float:
    # Branching on sign keeps exp() from overflowing for large |x|, which a long
    # horizon or large inventory can produce.
    if x >= 0:
        return 1 / (1 + math.exp(-x))
    z = math.exp(x)
    return z / (1 + z)


def round_to_ticks(quote: Quote, tick: int = 1, lo: int = 1, hi: int = 99) -> TickQuote:
    """Floor the bid and ceil the ask to the tick grid, clamped to [lo, hi], with bid < ask."""
    if tick < 1:
        raise ValueError(f"tick must be >= 1, got {tick}")
    if not 0 < lo < hi < 100:
        raise ValueError(f"need 0 < lo < hi < 100, got lo={lo}, hi={hi}")
    # With lo < hi, both being multiples of tick guarantees hi - lo >= tick, so the
    # bid >= ask fix-up below always has room.
    if lo % tick or hi % tick:
        raise ValueError(f"lo and hi must be multiples of tick {tick}, got lo={lo}, hi={hi}")

    bid = min(max(math.floor(quote.bid / tick) * tick, lo), hi)
    ask = min(max(math.ceil(quote.ask / tick) * tick, lo), hi)
    # A continuous bid < ask always survives floor/ceil, so bid >= ask here can only
    # come from both sides being clamped to the same bound. Keep the pinned side and
    # move the other one tick inside, giving the nearest valid two-sided quote.
    if bid >= ask:
        if ask == hi:
            bid = hi - tick
        else:
            ask = lo + tick
    return TickQuote(bid=bid, ask=ask)
