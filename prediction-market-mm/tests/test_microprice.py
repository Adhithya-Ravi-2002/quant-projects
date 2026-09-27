import pytest

from prediction_market_mm.microprice import micro_price
from prediction_market_mm.orderbook import PriceLevel

L = PriceLevel


def test_equal_sizes_equals_mid() -> None:
    # (100*52 + 100*48) / (100 + 100) = (5200 + 4800) / 200 = 10000 / 200 = 50.0
    # mid = (48 + 52) / 2 = 50.0
    assert micro_price(L(48, 100.0), L(52, 100.0)) == 50.0


def test_bid_heavy_leans_toward_ask() -> None:
    # (300*52 + 100*48) / (300 + 100) = (15600 + 4800) / 400 = 20400 / 400 = 51.0
    # mid is 50.0; 51.0 is 1 tick from the ask and 3 from the bid
    result = micro_price(L(48, 300.0), L(52, 100.0))
    assert result == 51.0
    assert abs(result - 52) < abs(result - 48)


def test_ask_heavy_leans_toward_bid() -> None:
    # (100*52 + 300*48) / (100 + 300) = (5200 + 14400) / 400 = 19600 / 400 = 49.0
    # mid is 50.0; 49.0 is 1 tick from the bid and 3 from the ask
    result = micro_price(L(48, 100.0), L(52, 300.0))
    assert result == 49.0
    assert abs(result - 48) < abs(result - 52)


def test_one_side_empty_collapses_to_other_price() -> None:
    # bid size 0: (0*52 + 50*48) / (0 + 50) = 2400 / 50 = 48.0 (the bid price)
    assert micro_price(L(48, 0.0), L(52, 50.0)) == 48.0
    # ask size 0: (50*52 + 0*48) / (50 + 0) = 2600 / 50 = 52.0 (the ask price)
    assert micro_price(L(48, 50.0), L(52, 0.0)) == 52.0


def test_both_sizes_zero_raises() -> None:
    # denominator 0 + 0 = 0: undefined
    with pytest.raises(ValueError):
        micro_price(L(48, 0.0), L(52, 0.0))


def test_negative_size_raises() -> None:
    # 5 + (-5) = 0 would otherwise divide by zero
    with pytest.raises(ValueError):
        micro_price(L(48, 5.0), L(52, -5.0))
