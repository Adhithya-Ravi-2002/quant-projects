import pytest

from prediction_market_mm.ofi import OFICalculator, RollingOFI, compute_ofi, level_contribution
from prediction_market_mm.orderbook import BookUpdate, OrderBook, PriceLevel, Side

L = PriceLevel


# --- Single-level contribution: bid side ---


def test_bid_price_improves() -> None:
    # 48 -> 49 is higher, so improved: contribution = +new size = +30 (old 100 ignored)
    assert level_contribution(L(48, 100.0), L(49, 30.0), Side.BID) == 30.0


def test_bid_price_unchanged_size_change() -> None:
    # 48 -> 48 unchanged: new - old = 70 - 100 = -30
    assert level_contribution(L(48, 100.0), L(48, 70.0), Side.BID) == -30.0


def test_bid_price_worsens() -> None:
    # 48 removed, new best 47 is lower, so worsened: contribution = -old size = -100
    # (the 200 resting at 47 is ignored)
    assert level_contribution(L(48, 100.0), L(47, 200.0), Side.BID) == -100.0


# --- Single-level contribution: ask side (improvement is a LOWER price) ---


def test_ask_price_improves() -> None:
    # 52 -> 51 is lower, so improved: contribution = +new size = +20
    assert level_contribution(L(52, 80.0), L(51, 20.0), Side.ASK) == 20.0


def test_ask_price_unchanged_size_change() -> None:
    # 52 -> 52 unchanged: new - old = 50 - 80 = -30
    assert level_contribution(L(52, 80.0), L(52, 50.0), Side.ASK) == -30.0


def test_ask_price_worsens() -> None:
    # 52 removed, new best 53 is higher, so worsened: contribution = -old size = -80
    assert level_contribution(L(52, 80.0), L(53, 60.0), Side.ASK) == -80.0


# --- Absent levels: treated as worst price with size 0 ---


def test_level_appears_counts_as_improvement() -> None:
    # absent -> 40x25: +new size = +25
    assert level_contribution(None, L(40, 25.0), Side.BID) == 25.0


def test_level_disappears_counts_as_worsening() -> None:
    # 60x15 -> absent: -old size = -15
    assert level_contribution(L(60, 15.0), None, Side.ASK) == -15.0


def test_level_absent_both_times_is_zero() -> None:
    assert level_contribution(None, None, Side.BID) == 0.0


# --- Combined single-level OFI (the approved 3-tick worked example) ---

WORKED_STATES = [
    ([L(48, 100.0), L(47, 200.0)], [L(52, 80.0), L(53, 60.0)]),  # t0
    ([L(49, 30.0), L(48, 100.0)], [L(52, 50.0), L(53, 60.0)]),   # t1
    ([L(48, 100.0), L(47, 200.0)], [L(51, 20.0), L(52, 50.0)]),  # t2
]


def test_single_level_ofi_worked_example() -> None:
    ticks = compute_ofi(WORKED_STATES, levels=1)
    assert len(ticks) == 2
    # t0->t1: bid 48->49 improves: +30; ask 52->52 unchanged: 50-80 = -30
    #         OFI = 30 - (-30) = +60
    assert ticks[0].bid == (30.0,)
    assert ticks[0].ask == (-30.0,)
    assert ticks[0].single_level == 60.0
    # t1->t2: bid 49->48 worsens: -30; ask 52->51 improves: +20
    #         OFI = -30 - 20 = -50
    assert ticks[1].bid == (-30.0,)
    assert ticks[1].ask == (20.0,)
    assert ticks[1].single_level == -50.0


def test_first_state_yields_no_tick() -> None:
    calc = OFICalculator(levels=1)
    assert calc.update(*WORKED_STATES[0]) is None
    assert calc.update(*WORKED_STATES[1]) is not None


# --- Multi-level OFI ---


def test_multi_level_includes_deeper_level() -> None:
    ticks = compute_ofi(WORKED_STATES, levels=2)
    # t0->t1, level 2: bid 47->48 improves: +100; ask 53->53 unchanged: 60-60 = 0
    #   level-2 OFI = 100 - 0 = +100
    #   multi-level = level 1 (+60) + level 2 (+100) = +160
    assert ticks[0].bid == (30.0, 100.0)
    assert ticks[0].ask == (-30.0, 0.0)
    assert ticks[0].multi_level == 160.0
    # t1->t2, level 2: bid 48->47 worsens: -100; ask 53->52 improves: +50
    #   level-2 OFI = -100 - 50 = -150
    #   multi-level = level 1 (-50) + level 2 (-150) = -200
    assert ticks[1].bid == (-30.0, -100.0)
    assert ticks[1].ask == (20.0, 50.0)
    assert ticks[1].multi_level == -200.0
    # single_level is unaffected by the extra depth
    assert ticks[0].single_level == 60.0
    assert ticks[1].single_level == -50.0


def test_multi_level_only_deep_level_changes() -> None:
    # Best levels identical; only bid level 3 changes size 40 -> 90.
    t0 = ([L(50, 10.0), L(49, 20.0), L(48, 40.0)], [L(55, 10.0)])
    t1 = ([L(50, 10.0), L(49, 20.0), L(48, 90.0)], [L(55, 10.0)])
    (tick,) = compute_ofi([t0, t1], levels=3)
    # level 1: 10-10 = 0; level 2: 20-20 = 0; level 3: 48->48 unchanged, 90-40 = +50
    # asks: level 1 unchanged 10-10 = 0; levels 2,3 absent both times = 0
    # multi-level = (0 + 0 + 50) - (0 + 0 + 0) = +50; single-level = 0 - 0 = 0
    assert tick.bid == (0.0, 0.0, 50.0)
    assert tick.multi_level == 50.0
    assert tick.single_level == 0.0


def test_levels_beyond_n_are_ignored() -> None:
    # Level 3 changes 40 -> 90, but N=2, so it must not contribute.
    t0 = ([L(50, 10.0), L(49, 20.0), L(48, 40.0)], [L(55, 10.0)])
    t1 = ([L(50, 10.0), L(49, 20.0), L(48, 90.0)], [L(55, 10.0)])
    (tick,) = compute_ofi([t0, t1], levels=2)
    assert tick.bid == (0.0, 0.0)
    assert tick.multi_level == 0.0


def test_side_with_fewer_levels_than_n() -> None:
    # Ask side goes from 1 level to 2: level 2 appears (absent -> 56x30).
    t0 = ([L(50, 10.0)], [L(55, 10.0)])
    t1 = ([L(50, 10.0)], [L(55, 10.0), L(56, 30.0)])
    (tick,) = compute_ofi([t0, t1], levels=2)
    # ask level 2 appears: +30. bid level 2 absent both times: 0. All else unchanged: 0.
    # multi-level = (0 + 0) - (0 + 30) = -30
    assert tick.ask == (0.0, 30.0)
    assert tick.multi_level == -30.0


def test_feeds_from_orderbook_depth() -> None:
    book = OrderBook.from_snapshot([(48, 100.0)], [(52, 80.0)])
    calc = OFICalculator(levels=1)
    calc.update(*book.depth(1))
    book.apply_update(BookUpdate(Side.BID, 49, 30.0))
    tick = calc.update(*book.depth(1))
    # bid 48->49 improves: +30; ask unchanged 80-80 = 0; OFI = 30 - 0 = +30
    assert tick is not None
    assert tick.single_level == 30.0


def test_reset_forgets_previous_state() -> None:
    calc = OFICalculator(levels=1)
    calc.update(*WORKED_STATES[0])
    calc.reset()
    assert calc.update(*WORKED_STATES[1]) is None


def test_invalid_levels() -> None:
    with pytest.raises(ValueError):
        OFICalculator(levels=0)


# --- Rolling window ---


def test_rolling_window_sum() -> None:
    r = RollingOFI(window_ms=500)
    r.add(0, 60.0)
    r.add(200, -50.0)
    r.add(450, 30.0)
    # as of 450, window (-50, 450] holds t=0, 200, 450: 60 - 50 + 30 = 40
    assert r.total() == 40.0
    r.add(700, 25.0)
    # as of 700, window (200, 700]: t=0 dropped; t=200 is exactly 500ms old, so
    # excluded by the open left edge. Holds t=450, 700: 30 + 25 = 55
    assert r.total() == 55.0
    # as of 960 (no new tick), window (460, 960] holds only t=700: 25
    assert r.total(as_of_ms=960) == 25.0
    # as of 1300, window (800, 1300] is empty: 0
    assert r.total(as_of_ms=1300) == 0.0


def test_rolling_window_default_is_500ms() -> None:
    r = RollingOFI()
    r.add(0, 10.0)
    r.add(499, 5.0)
    # as of 499, window (-1, 499] holds both: 10 + 5 = 15
    assert r.total() == 15.0
    # as of 500, window (0, 500]: t=0 excluded, leaves 5
    assert r.total(as_of_ms=500) == 5.0


def test_rolling_equal_timestamps_allowed() -> None:
    r = RollingOFI(window_ms=100)
    r.add(10, 1.0)
    r.add(10, 2.0)
    # both at t=10, window (-90, 10]: 1 + 2 = 3
    assert r.total() == 3.0


def test_rolling_empty_is_zero() -> None:
    assert RollingOFI().total() == 0.0


def test_rolling_rejects_time_going_backwards() -> None:
    r = RollingOFI()
    r.add(100, 1.0)
    with pytest.raises(ValueError):
        r.add(99, 1.0)
    r.total(as_of_ms=300)
    with pytest.raises(ValueError):
        r.add(200, 1.0)
    with pytest.raises(ValueError):
        r.total(as_of_ms=250)


def test_rolling_invalid_window() -> None:
    with pytest.raises(ValueError):
        RollingOFI(window_ms=0)
