"""L2 order book state for a single market."""

from __future__ import annotations

from bisect import bisect_left, insort
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from numbers import Integral


class Side(Enum):
    BID = "bid"
    ASK = "ask"


@dataclass(frozen=True, slots=True)
class PriceLevel:
    price: int
    size: float


@dataclass(frozen=True, slots=True)
class BookUpdate:
    """Incremental diff. A size of 0 removes the level."""

    side: Side
    price: int
    size: float


def _validate_price(price: int) -> int:
    # Prices are integer ticks so that dict lookups on diffs are exact. Integral
    # (rather than int) admits numpy integers coming from pandas/Parquet.
    if not isinstance(price, Integral) or isinstance(price, bool):
        raise TypeError(f"price must be an integer tick, got {price!r}")
    return int(price)


class _BookSide:
    def __init__(self, descending: bool) -> None:
        self._descending = descending
        self._sizes: dict[int, float] = {}
        # Always ascending; bids are read from the end.
        self._prices: list[int] = []

    def set(self, price: int, size: float) -> None:
        price = _validate_price(price)
        if size < 0:
            raise ValueError(f"size must be non-negative, got {size!r}")
        if size == 0:
            # Removing an absent level is a no-op so that replayed diffs are idempotent.
            if self._sizes.pop(price, None) is not None:
                del self._prices[bisect_left(self._prices, price)]
            return
        if price not in self._sizes:
            insort(self._prices, price)
        self._sizes[price] = size

    def best(self) -> PriceLevel | None:
        if not self._prices:
            return None
        price = self._prices[-1] if self._descending else self._prices[0]
        return PriceLevel(price, self._sizes[price])

    def levels(self, n: int) -> list[PriceLevel]:
        prices = self._prices[::-1][:n] if self._descending else self._prices[:n]
        return [PriceLevel(p, self._sizes[p]) for p in prices]

    def __len__(self) -> int:
        return len(self._prices)


class OrderBook:
    """L2 book for one market.

    All received levels are kept internally; `max_levels` caps only what is
    exposed via `depth()`. Truncating internal state would lose levels that
    should move into the top N when a better level is removed.
    """

    def __init__(self, max_levels: int = 10) -> None:
        if max_levels < 1:
            raise ValueError(f"max_levels must be >= 1, got {max_levels}")
        self.max_levels = max_levels
        self._bids = _BookSide(descending=True)
        self._asks = _BookSide(descending=False)

    @classmethod
    def from_snapshot(
        cls,
        bids: Iterable[tuple[int, float]],
        asks: Iterable[tuple[int, float]],
        max_levels: int = 10,
    ) -> OrderBook:
        book = cls(max_levels)
        book.apply_snapshot(bids, asks)
        return book

    def apply_snapshot(
        self, bids: Iterable[tuple[int, float]], asks: Iterable[tuple[int, float]]
    ) -> None:
        """Replace all book state. Levels may be given in any order."""
        # Built separately and swapped in so a malformed snapshot leaves the
        # existing state untouched.
        new_bids = self._build_side(bids, descending=True)
        new_asks = self._build_side(asks, descending=False)
        self._bids, self._asks = new_bids, new_asks

    @staticmethod
    def _build_side(levels: Iterable[tuple[int, float]], descending: bool) -> _BookSide:
        side = _BookSide(descending)
        seen: set[int] = set()
        for price, size in levels:
            price = _validate_price(price)
            if price in seen:
                raise ValueError(f"duplicate price {price} in snapshot")
            seen.add(price)
            side.set(price, size)
        return side

    def apply_update(self, update: BookUpdate) -> None:
        # Crossing is deliberately not rejected here: it can be a legitimate
        # transient state mid-batch. Callers check is_crossed() at batch boundaries.
        side = self._bids if update.side is Side.BID else self._asks
        side.set(update.price, update.size)

    def best_bid(self) -> PriceLevel | None:
        return self._bids.best()

    def best_ask(self) -> PriceLevel | None:
        return self._asks.best()

    def depth(self, n: int | None = None) -> tuple[list[PriceLevel], list[PriceLevel]]:
        """Top `n` levels per side (default `max_levels`), bids descending, asks ascending."""
        if n is None:
            n = self.max_levels
        if not 0 <= n <= self.max_levels:
            raise ValueError(f"n must be in [0, {self.max_levels}], got {n}")
        return self._bids.levels(n), self._asks.levels(n)

    def is_crossed(self) -> bool:
        """True if best bid >= best ask (locked counts as crossed)."""
        bid, ask = self.best_bid(), self.best_ask()
        return bid is not None and ask is not None and bid.price >= ask.price
