"""Order Flow Imbalance (OFI) from consecutive L2 book states."""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from prediction_market_mm.orderbook import PriceLevel, Side

Depth = tuple[Sequence[PriceLevel], Sequence[PriceLevel]]


def level_contribution(prev: PriceLevel | None, curr: PriceLevel | None, side: Side) -> float:
    """OFI contribution at one level on one side. None means the level is absent."""
    # An absent level behaves as the worst possible price with size 0, so a level
    # appearing is an improvement and a level disappearing is a worsening.
    if prev is None:
        return curr.size if curr is not None else 0.0
    if curr is None:
        return -prev.size
    # Bids improve upward, asks downward; the sign flip lets one comparison serve both.
    direction = 1 if side is Side.BID else -1
    move = direction * (curr.price - prev.price)
    if move > 0:
        return curr.size
    if move < 0:
        return -prev.size
    return curr.size - prev.size


@dataclass(frozen=True, slots=True)
class OFITick:
    """Per-level contributions for one transition, index 0 = best level."""

    bid: tuple[float, ...]
    ask: tuple[float, ...]

    @property
    def single_level(self) -> float:
        return self.bid[0] - self.ask[0]

    @property
    def multi_level(self) -> float:
        return sum(self.bid) - sum(self.ask)


class OFICalculator:
    """Streaming OFI over the top `levels` levels.

    Levels are compared by index (level i at t-1 vs level i at t), following
    Xu, Gould & Howison. A price level that shifts index without changing
    therefore produces offsetting contributions at the levels it moves between.

    OFI is path-dependent, so the states fed in should be batch-boundary states,
    not intermediate states within one exchange batch.
    """

    def __init__(self, levels: int = 10) -> None:
        if levels < 1:
            raise ValueError(f"levels must be >= 1, got {levels}")
        self.levels = levels
        self._prev: tuple[tuple[PriceLevel, ...], tuple[PriceLevel, ...]] | None = None

    def update(self, bids: Sequence[PriceLevel], asks: Sequence[PriceLevel]) -> OFITick | None:
        """Feed the next book state (best level first). Returns None for the first state."""
        curr = (tuple(bids[: self.levels]), tuple(asks[: self.levels]))
        prev, self._prev = self._prev, curr
        if prev is None:
            return None
        return OFITick(
            bid=self._side(prev[0], curr[0], Side.BID),
            ask=self._side(prev[1], curr[1], Side.ASK),
        )

    def reset(self) -> None:
        """Forget the previous state, e.g. after a resnapshot following corrupt data."""
        self._prev = None

    def _side(
        self, prev: Sequence[PriceLevel], curr: Sequence[PriceLevel], side: Side
    ) -> tuple[float, ...]:
        return tuple(
            level_contribution(
                prev[i] if i < len(prev) else None,
                curr[i] if i < len(curr) else None,
                side,
            )
            for i in range(self.levels)
        )


def compute_ofi(states: Iterable[Depth], levels: int = 10) -> list[OFITick]:
    """OFI for each consecutive pair of (bids, asks) states; n states yield n - 1 ticks."""
    calc = OFICalculator(levels)
    ticks: list[OFITick] = []
    for bids, asks in states:
        tick = calc.update(bids, asks)
        if tick is not None:
            ticks.append(tick)
    return ticks


class RollingOFI:
    """Sum of OFI values over the trailing window (t - window_ms, t]."""

    def __init__(self, window_ms: int = 500) -> None:
        if window_ms <= 0:
            raise ValueError(f"window_ms must be positive, got {window_ms}")
        self.window_ms = window_ms
        self._values: deque[tuple[int, float]] = deque()
        # Shared by add() and total(): eviction is destructive, so neither may
        # move time backwards or an evicted value could belong in a later answer.
        self._now_ms: int | None = None

    def add(self, timestamp_ms: int, value: float) -> None:
        self._advance(timestamp_ms)
        self._values.append((timestamp_ms, value))

    def total(self, as_of_ms: int | None = None) -> float:
        """Windowed sum as of `as_of_ms` (default: the latest time seen)."""
        if as_of_ms is None:
            if self._now_ms is None:
                return 0.0
            as_of_ms = self._now_ms
        self._advance(as_of_ms)
        cutoff = as_of_ms - self.window_ms
        while self._values and self._values[0][0] <= cutoff:
            self._values.popleft()
        # Recomputed rather than kept as a running sum, which would accumulate
        # float drift with fractional sizes; windows hold few ticks, so it is cheap.
        return math.fsum(v for _, v in self._values)

    def _advance(self, timestamp_ms: int) -> None:
        if self._now_ms is not None and timestamp_ms < self._now_ms:
            raise ValueError(f"timestamp {timestamp_ms} is before {self._now_ms}")
        self._now_ms = timestamp_ms
