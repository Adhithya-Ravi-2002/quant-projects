"""Kalshi orderbook_delta channel: message parsing and book reconstruction.

Kalshi publishes only YES bids and NO bids, as fixed-point dollar strings with
ascending price order. A NO bid at p is a YES ask at 100 - p (in cents), so the
book is kept in YES terms throughout. Only whole-cent (linear_cent) markets are
supported; a sub-cent price is rejected.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Any

from prediction_market_mm.ofi import OFICalculator, OFITick
from prediction_market_mm.orderbook import BookUpdate, OrderBook, PriceLevel, Side


def dollars_to_cents(price: str) -> int:
    """Convert a fixed-point dollar string such as "0.0800" to integer cents."""
    try:
        cents = Decimal(price) * 100
    except InvalidOperation as exc:
        raise ValueError(f"invalid price {price!r}") from exc
    if cents != cents.to_integral_value():
        raise ValueError(f"sub-cent price {price!r} is not supported")
    if not 0 < cents < 100:
        raise ValueError(f"price {price!r} is outside (0, 1) dollars")
    return int(cents)


def parse_size(size: str) -> float:
    """Parse a resting size, which must be finite and non-negative."""
    value = parse_delta(size)
    if value < 0:
        raise ValueError(f"negative size {size!r}")
    return value


def parse_delta(delta: str) -> float:
    """Parse a size change, which must be finite and may be negative."""
    try:
        value = Decimal(delta)
    except InvalidOperation as exc:
        raise ValueError(f"invalid size {delta!r}") from exc
    if not value.is_finite():
        raise ValueError(f"non-finite size {delta!r}")
    return float(value)


@dataclass(frozen=True, slots=True)
class KalshiSnapshot:
    """Snapshot in Kalshi terms: YES and NO bids as (cents, size)."""

    sid: int
    seq: int
    market_ticker: str
    yes: tuple[tuple[int, float], ...]
    no: tuple[tuple[int, float], ...]


@dataclass(frozen=True, slots=True)
class KalshiDelta:
    """Size change at one price, in Kalshi terms. `delta` is a change, not a new size."""

    sid: int
    seq: int
    market_ticker: str
    side: str
    price: int
    delta: float
    ts_ms: int | None


def parse_message(raw: Mapping[str, Any]) -> KalshiSnapshot | KalshiDelta | None:
    """Parse one decoded WebSocket message. Returns None for non-orderbook message types."""
    kind = raw.get("type")
    if kind not in ("orderbook_snapshot", "orderbook_delta"):
        return None
    try:
        msg = raw["msg"]
        sid, seq, ticker = int(raw["sid"]), int(raw["seq"]), str(msg["market_ticker"])
        if kind == "orderbook_snapshot":
            return KalshiSnapshot(
                sid=sid,
                seq=seq,
                market_ticker=ticker,
                yes=_parse_levels(msg.get("yes_dollars_fp", [])),
                no=_parse_levels(msg.get("no_dollars_fp", [])),
            )
        side = msg["side"]
        if side not in ("yes", "no"):
            raise ValueError(f"unknown side {side!r}")
        ts_ms = msg.get("ts_ms")
        return KalshiDelta(
            sid=sid,
            seq=seq,
            market_ticker=ticker,
            side=side,
            price=dollars_to_cents(msg["price_dollars"]),
            delta=parse_delta(msg["delta_fp"]),
            ts_ms=int(ts_ms) if ts_ms is not None else None,
        )
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError(f"malformed {kind} message: {exc!r}") from exc


def _parse_levels(levels: Iterable[Iterable[str]]) -> tuple[tuple[int, float], ...]:
    return tuple((dollars_to_cents(p), parse_size(s)) for p, s in levels)


def to_book_levels(
    yes: Iterable[tuple[int, float]], no: Iterable[tuple[int, float]]
) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    """Map Kalshi YES/NO bids to (bids, asks): YES bid p -> bid p, NO bid p -> ask 100 - p."""
    return [(p, s) for p, s in yes], [(100 - p, s) for p, s in no]


def to_book_side(side: str, price: int) -> tuple[Side, int]:
    return (Side.BID, price) if side == "yes" else (Side.ASK, 100 - price)


class MachineState(Enum):
    AWAITING_SNAPSHOT = "awaiting_snapshot"
    LIVE = "live"


@dataclass(slots=True)
class DataQualityCounters:
    """Event counts reported as data-quality metrics (SPEC-EVL-01)."""

    snapshots: int = 0
    batches: int = 0
    seq_gaps: int = 0
    out_of_order: int = 0
    crossed_at_boundary: int = 0
    negative_size: int = 0
    malformed: int = 0
    dropped_while_invalid: int = 0
    wrong_sid: int = 0
    wrong_ticker: int = 0

    @property
    def invalidations(self) -> int:
        return (
            self.seq_gaps
            + self.out_of_order
            + self.crossed_at_boundary
            + self.negative_size
            + self.malformed
        )


@dataclass(frozen=True, slots=True)
class BatchResult:
    """Book state at a batch boundary.

    `ofi` is None for a snapshot, which is the new OFI baseline. `ts_ms` is None
    for a snapshot or a delta without a timestamp.
    """

    seq: int
    ts_ms: int | None
    bids: tuple[PriceLevel, ...]
    asks: tuple[PriceLevel, ...]
    ofi: OFITick | None
    provisional: bool = False


@dataclass(slots=True)
class _OpenBatch:
    ts_ms: int | None
    last_seq: int


class KalshiBookMachine:
    """Reconstructs one market's book from one subscription and emits batch-boundary states.

    Batch boundaries: consecutive deltas sharing a ts_ms form one batch. A batch
    closes canonically only when a delta with a different ts_ms arrives, or a
    snapshot arrives; a delta without ts_ms is a batch on its own. This uses only
    recorded fields, so a replay of the raw messages reproduces the canonical
    stream exactly. The last batch of a stream stays open until something closes it.

    `end_batch()` gives the live loop a provisional view of the open batch during
    quiet periods. It does not close the batch, touch the OFI state, or apply the
    crossed-book check, so it can never change the canonical stream: a late delta
    with the same ts_ms still joins the batch.

    is_crossed() is checked only when a batch closes. A crossed book at a boundary,
    a seq gap or regression, a delta driving a size negative, or a malformed
    message invalidates the book: the open batch is discarded, OFI is reset, and
    deltas are dropped until a fresh snapshot arrives.
    """

    def __init__(self, sid: int, market_ticker: str, levels: int = 10) -> None:
        self.sid = sid
        self.market_ticker = market_ticker
        self.levels = levels
        self.book = OrderBook(max_levels=levels)
        self.counters = DataQualityCounters()
        self.state = MachineState.AWAITING_SNAPSHOT
        self._ofi = OFICalculator(levels)
        self._expected_seq: int | None = None
        self._open: _OpenBatch | None = None
        self._last_depth: tuple[list[PriceLevel], list[PriceLevel]] | None = None

    @property
    def needs_snapshot(self) -> bool:
        return self.state is MachineState.AWAITING_SNAPSHOT

    def process(self, raw: Mapping[str, Any]) -> list[BatchResult]:
        """Process one decoded message; returns canonical results closed by it (usually 0 or 1)."""
        try:
            message = parse_message(raw)
        except ValueError:
            self.counters.malformed += 1
            self._invalidate()
            return []
        if message is None:
            return []
        if message.sid != self.sid:
            self.counters.wrong_sid += 1
            return []
        if message.market_ticker != self.market_ticker:
            self.counters.wrong_ticker += 1
            return []
        if isinstance(message, KalshiSnapshot):
            return self._on_snapshot(message)
        return self._on_delta(message)

    def end_batch(self) -> BatchResult | None:
        """Provisional state of the open batch for live use only; see the class docstring.

        Returns None if there is no open batch, the book is invalid, or the book is
        currently crossed.
        """
        if self._open is None or self.needs_snapshot or self.book.is_crossed():
            return None
        assert self._last_depth is not None
        scratch = OFICalculator(self.levels)
        scratch.update(*self._last_depth)
        bids, asks = self.book.depth()
        return BatchResult(
            seq=self._open.last_seq,
            ts_ms=self._open.ts_ms,
            bids=tuple(bids),
            asks=tuple(asks),
            ofi=scratch.update(bids, asks),
            provisional=True,
        )

    def _on_snapshot(self, snap: KalshiSnapshot) -> list[BatchResult]:
        # Built before closing the open batch so that a snapshot rejected here is
        # handled exactly like one rejected during parsing: the open batch is
        # discarded, not closed.
        try:
            book = OrderBook.from_snapshot(*to_book_levels(snap.yes, snap.no), max_levels=self.levels)
        except ValueError:
            self.counters.malformed += 1
            self._invalidate()
            return []
        results = self._close_batch()
        self.book = book
        self.counters.snapshots += 1
        self._expected_seq = snap.seq + 1
        # A crossed snapshot cannot be a transient mid-batch state; it points to a
        # bad snapshot or a price-mapping error, so it is never accepted.
        if self.book.is_crossed():
            self.counters.crossed_at_boundary += 1
            self._invalidate()
            return results
        self.state = MachineState.LIVE
        # Reset even on a snapshot received while LIVE: the jump in state may hide
        # missed deltas, so no OFI tick may span it.
        self._ofi.reset()
        bids, asks = self.book.depth()
        self._ofi.update(bids, asks)
        self._last_depth = (bids, asks)
        results.append(
            BatchResult(seq=snap.seq, ts_ms=None, bids=tuple(bids), asks=tuple(asks), ofi=None)
        )
        return results

    def _on_delta(self, delta: KalshiDelta) -> list[BatchResult]:
        if self.needs_snapshot:
            self.counters.dropped_while_invalid += 1
            return []
        assert self._expected_seq is not None
        if delta.seq != self._expected_seq:
            # The missing message may belong to the open batch, so the open batch
            # is discarded rather than closed.
            if delta.seq > self._expected_seq:
                self.counters.seq_gaps += 1
            else:
                self.counters.out_of_order += 1
            self._invalidate()
            return []
        self._expected_seq += 1

        results: list[BatchResult] = []
        if self._open is not None and (delta.ts_ms is None or delta.ts_ms != self._open.ts_ms):
            results = self._close_batch()
            if self.needs_snapshot:
                return results

        side, price = to_book_side(delta.side, delta.price)
        # Rounded to the feed's 2-decimal size precision so that float error cannot
        # leave a tiny residual where the level should have been removed.
        new_size = round(self.book.size_at(side, price) + delta.delta, 2)
        if new_size < 0:
            self.counters.negative_size += 1
            self._invalidate()
            return results
        self.book.apply_update(BookUpdate(side, price, new_size))
        self._open = _OpenBatch(ts_ms=delta.ts_ms, last_seq=delta.seq)

        if delta.ts_ms is None:
            results.extend(self._close_batch())
        return results

    def _close_batch(self) -> list[BatchResult]:
        if self._open is None:
            return []
        batch, self._open = self._open, None
        if self.book.is_crossed():
            self.counters.crossed_at_boundary += 1
            self._invalidate()
            return []
        bids, asks = self.book.depth()
        tick = self._ofi.update(bids, asks)
        self._last_depth = (bids, asks)
        self.counters.batches += 1
        return [
            BatchResult(
                seq=batch.last_seq, ts_ms=batch.ts_ms, bids=tuple(bids), asks=tuple(asks), ofi=tick
            )
        ]

    def _invalidate(self) -> None:
        self.state = MachineState.AWAITING_SNAPSHOT
        self._open = None
        self._expected_seq = None
        self._last_depth = None
        self._ofi.reset()
