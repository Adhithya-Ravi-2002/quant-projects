import pytest

from prediction_market_mm.orderbook import BookUpdate, OrderBook, PriceLevel, Side

# Prices in integer cents. Given deliberately out of order to exercise sorting.
BIDS = [(47, 250.0), (48, 100.0), (45, 50.0)]
ASKS = [(55, 300.0), (52, 80.0), (53, 120.0)]


@pytest.fixture
def book() -> OrderBook:
    return OrderBook.from_snapshot(BIDS, ASKS)


def test_snapshot_initialization(book: OrderBook) -> None:
    bids, asks = book.depth()
    assert bids == [PriceLevel(48, 100.0), PriceLevel(47, 250.0), PriceLevel(45, 50.0)]
    assert asks == [PriceLevel(52, 80.0), PriceLevel(53, 120.0), PriceLevel(55, 300.0)]
    assert book.best_bid() == PriceLevel(48, 100.0)
    assert book.best_ask() == PriceLevel(52, 80.0)
    assert not book.is_crossed()


def test_snapshot_replaces_existing_state(book: OrderBook) -> None:
    book.apply_snapshot([(30, 10.0)], [(70, 20.0)])
    assert book.depth() == ([PriceLevel(30, 10.0)], [PriceLevel(70, 20.0)])


def test_empty_book() -> None:
    book = OrderBook()
    assert book.best_bid() is None
    assert book.best_ask() is None
    assert book.depth() == ([], [])
    assert not book.is_crossed()


def test_add_level_inside_spread_becomes_best(book: OrderBook) -> None:
    book.apply_update(BookUpdate(Side.BID, 50, 40.0))
    book.apply_update(BookUpdate(Side.ASK, 51, 15.0))
    assert book.best_bid() == PriceLevel(50, 40.0)
    assert book.best_ask() == PriceLevel(51, 15.0)


def test_add_level_deep_in_book(book: OrderBook) -> None:
    book.apply_update(BookUpdate(Side.BID, 46, 75.0))
    bids, _ = book.depth()
    assert [lvl.price for lvl in bids] == [48, 47, 46, 45]
    assert book.best_bid() == PriceLevel(48, 100.0)


def test_remove_non_best_level(book: OrderBook) -> None:
    book.apply_update(BookUpdate(Side.ASK, 53, 0))
    _, asks = book.depth()
    assert asks == [PriceLevel(52, 80.0), PriceLevel(55, 300.0)]


def test_remove_best_level_promotes_next(book: OrderBook) -> None:
    book.apply_update(BookUpdate(Side.BID, 48, 0))
    book.apply_update(BookUpdate(Side.ASK, 52, 0))
    assert book.best_bid() == PriceLevel(47, 250.0)
    assert book.best_ask() == PriceLevel(53, 120.0)


def test_remove_absent_level_is_noop(book: OrderBook) -> None:
    before = book.depth()
    book.apply_update(BookUpdate(Side.BID, 10, 0))
    assert book.depth() == before


def test_remove_last_level_empties_side() -> None:
    book = OrderBook.from_snapshot([(40, 5.0)], [(60, 5.0)])
    book.apply_update(BookUpdate(Side.BID, 40, 0))
    assert book.best_bid() is None
    assert book.best_ask() == PriceLevel(60, 5.0)


def test_change_existing_level_size(book: OrderBook) -> None:
    book.apply_update(BookUpdate(Side.BID, 47, 10.0))
    book.apply_update(BookUpdate(Side.ASK, 52, 500.0))
    bids, asks = book.depth()
    assert bids == [PriceLevel(48, 100.0), PriceLevel(47, 10.0), PriceLevel(45, 50.0)]
    assert asks[0] == PriceLevel(52, 500.0)
    assert len(bids) == 3 and len(asks) == 3


def test_best_prices_after_update_sequence(book: OrderBook) -> None:
    updates = [
        BookUpdate(Side.BID, 49, 30.0),   # new best bid
        BookUpdate(Side.ASK, 52, 0),      # best ask removed -> 53
        BookUpdate(Side.ASK, 53, 60.0),   # resize new best ask
        BookUpdate(Side.BID, 49, 0),      # new best bid removed -> 48
        BookUpdate(Side.BID, 48, 90.0),   # resize restored best bid
    ]
    for u in updates:
        book.apply_update(u)
    assert book.best_bid() == PriceLevel(48, 90.0)
    assert book.best_ask() == PriceLevel(53, 60.0)


def test_depth_truncates_to_max_levels() -> None:
    bids = [(p, 1.0) for p in range(30, 45)]
    asks = [(p, 1.0) for p in range(55, 70)]
    book = OrderBook.from_snapshot(bids, asks, max_levels=10)
    top_bids, top_asks = book.depth()
    assert [lvl.price for lvl in top_bids] == list(range(44, 34, -1))
    assert [lvl.price for lvl in top_asks] == list(range(55, 65))
    assert len(book.depth(3)[0]) == 3


def test_levels_beyond_max_are_retained_and_promoted() -> None:
    bids = [(p, 1.0) for p in range(1, 5)]
    book = OrderBook.from_snapshot(bids, [], max_levels=3)
    assert [lvl.price for lvl in book.depth()[0]] == [4, 3, 2]
    book.apply_update(BookUpdate(Side.BID, 4, 0))
    assert [lvl.price for lvl in book.depth()[0]] == [3, 2, 1]


def test_deep_level_added_by_update_is_retained_and_promoted() -> None:
    book = OrderBook.from_snapshot([(2, 1.0), (3, 1.0), (4, 1.0)], [], max_levels=3)
    book.apply_update(BookUpdate(Side.BID, 1, 7.0))
    assert [lvl.price for lvl in book.depth()[0]] == [4, 3, 2]
    book.apply_update(BookUpdate(Side.BID, 4, 0))
    # Size 7.0 confirms the promoted level is the one the diff added.
    assert book.depth()[0] == [PriceLevel(3, 1.0), PriceLevel(2, 1.0), PriceLevel(1, 7.0)]


def test_ask_levels_beyond_max_are_retained_and_promoted() -> None:
    asks = [(96, 1.0), (97, 2.0), (98, 3.0), (99, 4.0)]
    book = OrderBook.from_snapshot([], asks, max_levels=3)
    assert [lvl.price for lvl in book.depth()[1]] == [96, 97, 98]
    book.apply_update(BookUpdate(Side.ASK, 96, 0))
    assert book.depth()[1] == [PriceLevel(97, 2.0), PriceLevel(98, 3.0), PriceLevel(99, 4.0)]


def test_is_crossed_and_locked(book: OrderBook) -> None:
    book.apply_update(BookUpdate(Side.BID, 52, 1.0))
    assert book.is_crossed()
    book.apply_update(BookUpdate(Side.BID, 52, 0))
    book.apply_update(BookUpdate(Side.BID, 53, 1.0))
    assert book.is_crossed()


def test_negative_size_rejected(book: OrderBook) -> None:
    with pytest.raises(ValueError):
        book.apply_update(BookUpdate(Side.BID, 47, -1.0))


def test_float_price_rejected(book: OrderBook) -> None:
    with pytest.raises(TypeError):
        book.apply_update(BookUpdate(Side.BID, 0.47, 1.0))  # type: ignore[arg-type]


def test_bad_snapshot_leaves_state_intact(book: OrderBook) -> None:
    before = book.depth()
    with pytest.raises(ValueError):
        book.apply_snapshot([(40, 1.0), (40, 2.0)], [])
    assert book.depth() == before


def test_depth_out_of_range(book: OrderBook) -> None:
    with pytest.raises(ValueError):
        book.depth(11)
