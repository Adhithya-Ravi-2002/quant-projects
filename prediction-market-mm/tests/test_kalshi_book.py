from dataclasses import replace
from typing import Any

import pytest

from prediction_market_mm.kalshi_book import (
    BatchResult,
    KalshiBookMachine,
    MachineState,
    dollars_to_cents,
    to_book_levels,
)
from prediction_market_mm.orderbook import PriceLevel, Side

L = PriceLevel
SID = 2
TICKER = "FED-23DEC-T3.00"


def snapshot(
    seq: int,
    yes: list[list[str]],
    no: list[list[str]],
    sid: int = SID,
    ticker: str = TICKER,
) -> dict[str, Any]:
    return {
        "type": "orderbook_snapshot",
        "sid": sid,
        "seq": seq,
        "msg": {
            "market_ticker": ticker,
            "market_id": "9b0f6b43-5b68-4f9f-9f02-9a2d1b8ac1a1",
            "yes_dollars_fp": yes,
            "no_dollars_fp": no,
        },
    }


def delta(
    seq: int,
    side: str,
    price: str,
    change: str,
    ts_ms: int | None,
    sid: int = SID,
    ticker: str = TICKER,
) -> dict[str, Any]:
    msg: dict[str, Any] = {
        "market_ticker": ticker,
        "market_id": "9b0f6b43-5b68-4f9f-9f02-9a2d1b8ac1a1",
        "price_dollars": price,
        "delta_fp": change,
        "side": side,
    }
    if ts_ms is not None:
        msg["ts_ms"] = ts_ms
    return {"type": "orderbook_delta", "sid": sid, "seq": seq, "msg": msg}


# The snapshot from Kalshi's docs.
# YES bids map directly: 0.08 -> bid 8 x 300, 0.22 -> bid 22 x 333
# NO bids become asks at 100 - p: 0.54 -> ask 46 x 20, 0.56 -> ask 44 x 146
DOC_SNAPSHOT = snapshot(
    seq=2,
    yes=[["0.0800", "300.00"], ["0.2200", "333.00"]],
    no=[["0.5400", "20.00"], ["0.5600", "146.00"]],
)


@pytest.fixture
def machine() -> KalshiBookMachine:
    m = KalshiBookMachine(sid=SID, market_ticker=TICKER)
    m.process(DOC_SNAPSHOT)
    return m


def best(result: BatchResult) -> tuple[PriceLevel, PriceLevel]:
    return result.bids[0], result.asks[0]


# --- Parsing and conversion ---


def test_dollars_to_cents() -> None:
    # "0.0800" * 100 = 8.00 -> 8; "0.2200" -> 22; "0.99" -> 99
    assert dollars_to_cents("0.0800") == 8
    assert dollars_to_cents("0.2200") == 22
    assert dollars_to_cents("0.99") == 99


@pytest.mark.parametrize("price", ["0.0850", "1.0000", "0.0000", "-0.10", "abc"])
def test_dollars_to_cents_rejects(price: str) -> None:
    # 0.0850 -> 8.5 cents (sub-cent); 1.00 and 0.00 are outside (0, 100) cents
    with pytest.raises(ValueError):
        dollars_to_cents(price)


def test_to_book_levels_maps_no_bids_to_asks() -> None:
    # NO 54 -> ask 100 - 54 = 46; NO 56 -> ask 100 - 56 = 44
    bids, asks = to_book_levels([(8, 300.0), (22, 333.0)], [(54, 20.0), (56, 146.0)])
    assert bids == [(8, 300.0), (22, 333.0)]
    assert asks == [(46, 20.0), (44, 146.0)]


def test_snapshot_conversion() -> None:
    m = KalshiBookMachine(sid=SID, market_ticker=TICKER)
    (result,) = m.process(DOC_SNAPSHOT)
    # bids descending: 22 x 333, 8 x 300; asks ascending: 44 x 146, 46 x 20
    # Best NO bid 56 gives best YES ask 100 - 56 = 44; spread 44 - 22 = 22 cents
    assert result.bids == (L(22, 333.0), L(8, 300.0))
    assert result.asks == (L(44, 146.0), L(46, 20.0))
    assert result.ofi is None
    assert result.seq == 2 and result.ts_ms is None
    assert m.state is MachineState.LIVE
    assert m.counters.snapshots == 1


def test_non_orderbook_message_ignored(machine: KalshiBookMachine) -> None:
    before = replace(machine.counters)
    assert machine.process({"type": "subscribed", "id": 1, "msg": {"channel": "orderbook_delta", "sid": 2}}) == []
    assert machine.counters == before
    assert machine.state is MachineState.LIVE


# --- Deltas ---


def test_delta_adds_changes_and_removes_levels(machine: KalshiBookMachine) -> None:
    assert machine.process(delta(3, "yes", "0.2300", "50.00", 1000)) == []  # add bid 23: 0 + 50 = 50
    assert machine.process(delta(4, "no", "0.5600", "-46.00", 1000)) == []  # ask 44: 146 - 46 = 100
    assert machine.process(delta(5, "no", "0.5400", "-20.00", 1000)) == []  # ask 46: 20 - 20 = 0, removed
    (result,) = machine.process(delta(6, "yes", "0.0800", "1.00", 2000))    # new ts_ms closes batch 1000
    assert result.ts_ms == 1000 and result.seq == 5
    assert result.bids == (L(23, 50.0), L(22, 333.0), L(8, 300.0))
    assert result.asks == (L(44, 100.0),)
    # OFI vs the snapshot baseline, best level:
    #   bid 22 -> 23 improves: +50;  ask 44 -> 44 unchanged: 100 - 146 = -46
    #   single-level OFI = 50 - (-46) = 96
    assert result.ofi is not None
    assert result.ofi.single_level == 96.0
    # the ts 2000 delta is applied but its batch is still open
    assert machine.book.size_at(Side.BID, 8) == 301.0


def test_delta_to_negative_size_invalidates(machine: KalshiBookMachine) -> None:
    # NO 0.54 is ask 46 x 20: 20 - 25 = -5
    assert machine.process(delta(3, "no", "0.5400", "-25.00", 1000)) == []
    assert machine.needs_snapshot
    assert machine.counters.negative_size == 1


def test_fractional_sizes_remove_level_exactly() -> None:
    m = KalshiBookMachine(sid=SID, market_ticker=TICKER)
    m.process(snapshot(2, yes=[["0.1000", "0.30"]], no=[["0.8000", "5.00"]]))
    # In floats, 0.30 - 0.10 - 0.20 = -2.8e-17, which would look negative.
    # Rounded to 2 decimals after each step: 0.30 -> 0.20 -> 0.00, so the level is removed.
    m.process(delta(3, "yes", "0.1000", "-0.10", 1000))
    m.process(delta(4, "yes", "0.1000", "-0.20", 1000))
    assert m.counters.negative_size == 0
    assert m.book.size_at(Side.BID, 10) == 0.0
    assert m.book.best_bid() is None


def test_delta_without_ts_is_its_own_batch(machine: KalshiBookMachine) -> None:
    (result,) = machine.process(delta(3, "yes", "0.2200", "7.00", None))
    # bid 22: 333 + 7 = 340; unchanged price, OFI = 340 - 333 = +7
    assert result.ts_ms is None
    assert result.bids[0] == L(22, 340.0)
    assert result.ofi is not None and result.ofi.single_level == 7.0


def test_snapshot_closes_open_batch_first(machine: KalshiBookMachine) -> None:
    machine.process(delta(3, "yes", "0.2300", "50.00", 1000))
    results = machine.process(snapshot(4, yes=[["0.3000", "5.00"]], no=[["0.6000", "7.00"]]))
    # 1: batch ts 1000 (bid 23 x 50 on top); 2: the new snapshot (bid 30, ask 100 - 60 = 40)
    assert [r.ts_ms for r in results] == [1000, None]
    assert results[0].bids[0] == L(23, 50.0)
    assert best(results[1]) == (L(30, 5.0), L(40, 7.0))
    assert results[1].ofi is None


# --- Sequence handling ---


def test_seq_gap_invalidates_and_discards_open_batch(machine: KalshiBookMachine) -> None:
    machine.process(delta(3, "yes", "0.2300", "50.00", 1000))
    # expected 4, got 5: gap. Seq 4 might have belonged to batch 1000, so it is discarded.
    assert machine.process(delta(5, "yes", "0.2300", "10.00", 2000)) == []
    assert machine.needs_snapshot
    assert machine.counters.seq_gaps == 1
    assert machine.counters.batches == 0
    # further deltas are dropped until a snapshot arrives
    assert machine.process(delta(6, "yes", "0.2300", "10.00", 3000)) == []
    assert machine.counters.dropped_while_invalid == 1


def test_seq_regression_invalidates(machine: KalshiBookMachine) -> None:
    machine.process(delta(3, "yes", "0.2300", "50.00", 1000))
    assert machine.process(delta(3, "yes", "0.2300", "50.00", 1000)) == []  # duplicate seq 3
    assert machine.needs_snapshot
    assert machine.counters.out_of_order == 1


def test_deltas_before_first_snapshot_are_dropped() -> None:
    m = KalshiBookMachine(sid=SID, market_ticker=TICKER)
    assert m.process(delta(1, "yes", "0.2300", "50.00", 1000)) == []
    assert m.counters.dropped_while_invalid == 1
    assert m.needs_snapshot


def test_recovery_after_fresh_snapshot(machine: KalshiBookMachine) -> None:
    machine.process(delta(5, "yes", "0.2300", "50.00", 1000))  # expected 3: gap
    assert machine.needs_snapshot

    (snap,) = machine.process(snapshot(10, yes=[["0.3000", "5.00"]], no=[["0.6000", "7.00"]]))
    # bid 30 x 5; ask 100 - 60 = 40 x 7
    assert best(snap) == (L(30, 5.0), L(40, 7.0))
    assert machine.state is MachineState.LIVE

    machine.process(delta(11, "yes", "0.3000", "5.00", 3000))  # seq continues from snapshot: 10 + 1
    (result,) = machine.process(delta(12, "yes", "0.3000", "1.00", 4000))
    # OFI baseline is the new snapshot (OFI was reset), not the pre-gap book:
    #   bid 30 -> 30 unchanged: 10 - 5 = +5;  ask 40 -> 40 unchanged: 7 - 7 = 0
    #   single-level OFI = 5 - 0 = 5
    assert result.ofi is not None and result.ofi.single_level == 5.0
    assert machine.counters.snapshots == 2
    assert machine.counters.seq_gaps == 1


# --- Crossed-book policy ---


def test_crossed_at_boundary_invalidates(machine: KalshiBookMachine) -> None:
    # YES bid at 45 >= best ask 44: crossed, and nothing un-crosses it within batch 1000
    machine.process(delta(3, "yes", "0.4500", "10.00", 1000))
    assert machine.process(delta(4, "yes", "0.0800", "1.00", 2000)) == []  # boundary
    assert machine.needs_snapshot
    assert machine.counters.crossed_at_boundary == 1
    assert machine.counters.batches == 0


def test_transient_cross_within_batch_does_not_invalidate(machine: KalshiBookMachine) -> None:
    machine.process(delta(3, "yes", "0.4500", "10.00", 1000))    # bid 45 vs ask 44: crossed
    assert machine.book.is_crossed()
    machine.process(delta(4, "no", "0.5600", "-146.00", 1000))   # ask 44: 146 - 146 = 0, removed
    assert not machine.book.is_crossed()                         # best ask now 46 > 45
    (result,) = machine.process(delta(5, "yes", "0.0800", "1.00", 2000))
    assert machine.state is MachineState.LIVE
    assert machine.counters.crossed_at_boundary == 0
    assert best(result) == (L(45, 10.0), L(46, 20.0))
    # OFI: bid 22 -> 45 improves: +10; ask 44 -> 46 worsens: -146
    #      single-level OFI = 10 - (-146) = 156
    assert result.ofi is not None and result.ofi.single_level == 156.0


def test_crossed_snapshot_is_rejected() -> None:
    m = KalshiBookMachine(sid=SID, market_ticker=TICKER)
    # YES bid 50; NO bid 55 -> ask 100 - 55 = 45 < 50: crossed
    assert m.process(snapshot(2, yes=[["0.5000", "1.00"]], no=[["0.5500", "1.00"]])) == []
    assert m.needs_snapshot
    assert m.counters.crossed_at_boundary == 1


# --- Subscription identity ---


def test_wrong_sid_and_ticker_rejected(machine: KalshiBookMachine) -> None:
    assert machine.process(delta(3, "yes", "0.2300", "50.00", 1000, sid=99)) == []
    assert machine.process(delta(3, "yes", "0.2300", "50.00", 1000, ticker="OTHER")) == []
    assert machine.counters.wrong_sid == 1
    assert machine.counters.wrong_ticker == 1
    # rejected messages do not consume seq: our own seq 3 is still accepted
    machine.process(delta(3, "yes", "0.2300", "50.00", 1000))
    assert machine.counters.seq_gaps == 0
    assert machine.book.size_at(Side.BID, 23) == 50.0
    assert machine.state is MachineState.LIVE


def test_malformed_message_invalidates(machine: KalshiBookMachine) -> None:
    bad = delta(3, "yes", "0.2300", "50.00", 1000)
    del bad["msg"]["price_dollars"]
    assert machine.process(bad) == []
    assert machine.counters.malformed == 1
    assert machine.needs_snapshot


def test_sub_cent_price_counts_as_malformed(machine: KalshiBookMachine) -> None:
    machine.process(delta(3, "yes", "0.2350", "50.00", 1000))
    assert machine.counters.malformed == 1
    assert machine.needs_snapshot


@pytest.mark.parametrize("size", ["NaN", "Infinity", "-1.00"])
def test_bad_snapshot_size_is_malformed(machine: KalshiBookMachine, size: str) -> None:
    bad = snapshot(3, yes=[["0.2200", size]], no=[["0.5600", "146.00"]])
    assert machine.process(bad) == []
    assert machine.counters.malformed == 1
    assert machine.needs_snapshot


@pytest.mark.parametrize("change", ["NaN", "Infinity", "-Infinity"])
def test_non_finite_delta_is_malformed(machine: KalshiBookMachine, change: str) -> None:
    assert machine.process(delta(3, "yes", "0.2200", change, 1000)) == []
    assert machine.counters.malformed == 1
    assert machine.needs_snapshot


def test_negative_delta_price_is_malformed(machine: KalshiBookMachine) -> None:
    assert machine.process(delta(3, "yes", "-1.00", "5.00", 1000)) == []
    assert machine.counters.malformed == 1
    assert machine.needs_snapshot


def test_negative_delta_amount_is_valid(machine: KalshiBookMachine) -> None:
    # delta_fp is a change, and Kalshi's documented example is "-54.00": bid 22 is 333 - 1 = 332
    machine.process(delta(3, "yes", "0.2200", "-1.00", 1000))
    assert machine.counters.malformed == 0
    assert machine.book.size_at(Side.BID, 22) == 332.0


def test_snapshot_with_duplicate_price_is_malformed(machine: KalshiBookMachine) -> None:
    machine.process(delta(3, "yes", "0.2300", "50.00", 1000))  # open batch
    dup = snapshot(4, yes=[["0.2200", "1.00"], ["0.2200", "2.00"]], no=[["0.5600", "1.00"]])
    assert machine.process(dup) == []  # open batch discarded, not closed
    assert machine.counters.malformed == 1
    assert machine.counters.batches == 0
    assert machine.needs_snapshot


def test_snapshot_with_negative_size_is_malformed(machine: KalshiBookMachine) -> None:
    neg = snapshot(3, yes=[["0.2200", "-5.00"]], no=[["0.5600", "1.00"]])
    assert machine.process(neg) == []
    assert machine.counters.malformed == 1
    assert machine.needs_snapshot


# --- Canonical vs provisional output ---


def test_end_batch_does_not_change_canonical_stream() -> None:
    stream = [
        DOC_SNAPSHOT,
        delta(3, "yes", "0.2300", "50.00", 1000),
        delta(4, "no", "0.5600", "-46.00", 1000),  # late same-ts delta
        delta(5, "yes", "0.0800", "1.00", 2000),
    ]
    plain = KalshiBookMachine(sid=SID, market_ticker=TICKER)
    canonical_plain = [r for msg in stream for r in plain.process(msg)]

    idle = KalshiBookMachine(sid=SID, market_ticker=TICKER)
    canonical_idle: list[BatchResult] = []
    for i, msg in enumerate(stream):
        canonical_idle.extend(idle.process(msg))
        if i == 1:
            # Idle period after seq 3: provisional view of batch 1000 so far.
            provisional = idle.end_batch()
            assert provisional is not None and provisional.provisional
            # bid 22 -> 23 improves: +50; ask 44 unchanged: 146 - 146 = 0; OFI = 50
            assert provisional.ofi is not None and provisional.ofi.single_level == 50.0
            assert idle.end_batch() == provisional  # repeatable, no side effects

    assert canonical_idle == canonical_plain
    # Canonical batch 1000 includes the late delta: ask 44 is 146 - 46 = 100,
    # OFI = 50 - (100 - 146) = 96, not the provisional 50.
    batch = canonical_plain[1]
    assert batch.ts_ms == 1000 and not batch.provisional
    assert batch.asks[0] == L(44, 100.0)
    assert batch.ofi is not None and batch.ofi.single_level == 96.0


def test_end_batch_none_when_nothing_to_report(machine: KalshiBookMachine) -> None:
    assert machine.end_batch() is None  # no open batch
    machine.process(delta(3, "yes", "0.4500", "10.00", 1000))  # crossed mid-batch
    assert machine.end_batch() is None
    assert machine.state is MachineState.LIVE  # provisional check never invalidates
