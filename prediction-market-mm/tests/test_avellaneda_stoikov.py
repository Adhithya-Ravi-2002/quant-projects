import math

import pytest

from prediction_market_mm.avellaneda_stoikov import (
    LogitSpaceAS,
    PriceSpaceAS,
    QuotingModel,
    Quote,
    TickQuote,
    round_to_ticks,
)

# Price-space parameters from the hand example.
PRICE = PriceSpaceAS(gamma=0.01, k=1.5)
PRICE_SIGMA = 2.0

# Logit-space parameters matched to PRICE at mid 50, where 1 cent = c = 0.04 logit
# (dx/dp = 1 / (p(1-p)) = 4 per unit probability at p = 0.5):
#   sigma = 2 * c = 0.08, gamma = 0.01 / c = 0.25, k = 1.5 / c = 37.5
LOGIT = LogitSpaceAS(gamma=0.25, k=37.5)
LOGIT_SIGMA = 0.08

MODELS = [(PRICE, PRICE_SIGMA), (LOGIT, LOGIT_SIGMA)]


# --- Hand examples ---


def test_price_space_hand_example() -> None:
    q = PRICE.quote(mid=50, inventory=100, sigma=2, time_remaining=1)
    # gamma * sigma^2 * tau = 0.01 * 4 * 1 = 0.04
    # r = 50 - 100 * 0.04 = 46
    # spread = 0.04 + (2 / 0.01) * ln(1 + 0.01 / 1.5)
    #        = 0.04 + 200 * 0.0066445 = 0.04 + 1.3289085 = 1.3689085
    # bid = 46 - 0.6844543 = 45.3155457, ask = 46 + 0.6844543 = 46.6844543
    assert q.reservation == pytest.approx(46.0)
    assert q.ask - q.bid == pytest.approx(1.3689085, abs=1e-6)
    assert q.bid == pytest.approx(45.3155457, abs=1e-6)
    assert q.ask == pytest.approx(46.6844543, abs=1e-6)


def test_logit_space_hand_example() -> None:
    q = LOGIT.quote(mid=50, inventory=100, sigma=0.08, time_remaining=1)
    # x = ln(0.5 / 0.5) = 0
    # gamma * sigma^2 * tau = 0.25 * 0.0064 * 1 = 0.0016
    # r_x = 0 - 100 * 0.0016 = -0.16
    # spread_x = 0.0016 + (2 / 0.25) * ln(1 + 0.25 / 37.5)
    #          = 0.0016 + 8 * 0.0066445 = 0.0016 + 0.0531563 = 0.0547563
    # bid_x = -0.16 - 0.0273782 = -0.1873782; ask_x = -0.16 + 0.0273782 = -0.1326218
    # cents = 100 / (1 + e^-x):
    #   r   = 100 / (1 + 1.1735109) = 46.0085
    #   bid = 100 / (1 + 1.2060833) = 45.3292
    #   ask = 100 / (1 + 1.1418181) = 46.6893
    assert q.reservation == pytest.approx(46.0085, abs=1e-4)
    assert q.bid == pytest.approx(45.3292, abs=1e-4)
    assert q.ask == pytest.approx(46.6893, abs=1e-4)


def test_models_agree_at_mid_50_with_matched_parameters() -> None:
    # 100 * sigmoid(x) = 50 + 25x - 2.0833x^3 + ...; there is no x^2 term because
    # p = 0.5 is the sigmoid's inflection point. With matched parameters the price
    # model is exactly the linear part, so the gap is about 2.0833 * |x|^3 cents.
    flat_p = PRICE.quote(50, 0, PRICE_SIGMA, 1)
    flat_l = LOGIT.quote(50, 0, LOGIT_SIGMA, 1)
    # q = 0: |bid_x| = |ask_x| = 0.0273782; gap = 2.0833 * 0.0273782^3 = 0.00004
    for a, b in [(flat_p.bid, flat_l.bid), (flat_p.ask, flat_l.ask),
                 (flat_p.reservation, flat_l.reservation)]:
        assert abs(a - b) < 0.01

    long_p = PRICE.quote(50, 100, PRICE_SIGMA, 1)
    long_l = LOGIT.quote(50, 100, LOGIT_SIGMA, 1)
    # q = 100: the gap grows with the inventory offset (cubic in x):
    #   reservation: 2.0833 * 0.16^3      = 0.0085  (46.0085 vs 46.0000)
    #   ask:         2.0833 * 0.1326218^3 = 0.0049  (46.6893 vs 46.6845)
    #   bid:         2.0833 * 0.1873782^3 = 0.0137  (45.3292 vs 45.3155)
    assert abs(long_l.reservation - long_p.reservation) < 0.01
    assert abs(long_l.ask - long_p.ask) < 0.01
    assert abs(long_l.bid - long_p.bid) < 0.015


# --- Inventory behavior ---


@pytest.mark.parametrize(("model", "sigma"), MODELS)
def test_flat_inventory_reservation_equals_mid(model: QuotingModel, sigma: float) -> None:
    # q = 0: r = s - 0 = s (in logit space, sigmoid(logit(p)) = p)
    for mid in (3.0, 50.0, 81.5):
        assert model.quote(mid, 0, sigma, 1).reservation == pytest.approx(mid, abs=1e-9)


@pytest.mark.parametrize(("model", "sigma"), MODELS)
def test_inventory_skew_mirrors_at_mid_50(model: QuotingModel, sigma: float) -> None:
    long = model.quote(50, 100, sigma, 1)
    short = model.quote(50, -100, sigma, 1)
    # price:  r_long = 50 - 4 = 46, r_short = 50 + 4 = 54
    # logit:  r_x = -0.16 and +0.16, sigmoid symmetric about 0: 46.0085 and 53.9915
    assert long.reservation < 50 < short.reservation
    assert long.reservation - 50 == pytest.approx(-(short.reservation - 50))
    # Mirror image: long bid = 100 - short ask (e.g. price: 45.3155 = 100 - 54.6845)
    assert long.bid == pytest.approx(100 - short.ask)
    assert long.ask == pytest.approx(100 - short.bid)


# --- Spread ---


@pytest.mark.parametrize(("model", "sigma"), MODELS)
def test_spread_widens_with_sigma(model: QuotingModel, sigma: float) -> None:
    low = model.quote(50, 0, sigma * 0.5, 1)
    high = model.quote(50, 0, sigma * 1.5, 1)
    # price: sigma 1 -> 0.01*1 + 1.3289085 = 1.3389; sigma 3 -> 0.09 + 1.3289085 = 1.4189
    # logit: sigma 0.04 -> 0.25*0.0016 + 0.0531563 = 0.0535563
    #        sigma 0.12 -> 0.25*0.0144 + 0.0531563 = 0.0567563 (logit units)
    assert high.ask - high.bid > low.ask - low.bid


@pytest.mark.parametrize(("model", "sigma"), MODELS)
def test_spread_widens_with_tau(model: QuotingModel, sigma: float) -> None:
    short = model.quote(50, 0, sigma, 1)
    long = model.quote(50, 0, sigma, 10)
    # price: tau 1 -> 0.04 + 1.3289085 = 1.3689; tau 10 -> 0.4 + 1.3289085 = 1.7289
    # logit: tau 1 -> 0.0016 + 0.0531563 = 0.0547563; tau 10 -> 0.016 + 0.0531563 = 0.0691563
    assert long.ask - long.bid > short.ask - short.bid


# --- Boundary behavior ---


def test_near_boundary_price_space_leaves_range_logit_does_not() -> None:
    price = PRICE.quote(mid=2, inventory=100, sigma=PRICE_SIGMA, time_remaining=1)
    # r = 2 - 100 * 0.04 = -2 (a negative probability)
    assert price.reservation == pytest.approx(-2.0)
    assert price.bid < 0

    logit = LOGIT.quote(mid=2, inventory=100, sigma=LOGIT_SIGMA, time_remaining=1)
    # x = ln(0.02 / 0.98) = -3.8918; r_x = -3.8918 - 0.16 = -4.0518
    # bid_x = -4.0792, ask_x = -4.0244
    # cents: r = 1.7093, bid = 1.6639, ask = 1.7560
    assert 0 < logit.bid < logit.ask < 100
    assert logit.reservation == pytest.approx(1.7093, abs=1e-4)
    assert logit.bid == pytest.approx(1.6639, abs=1e-4)
    assert logit.ask == pytest.approx(1.7560, abs=1e-4)


def test_near_boundary_sub_tick_spread_rounds_to_one_tick() -> None:
    q = LOGIT.quote(mid=2, inventory=100, sigma=LOGIT_SIGMA, time_remaining=1)
    # continuous spread = 1.7560 - 1.6639 = 0.0920, below one tick
    assert q.ask - q.bid < 1
    # floor(1.6639) = 1, ceil(1.7560) = 2: exactly one tick apart
    assert round_to_ticks(q) == TickQuote(bid=1, ask=2)


def test_extreme_logit_does_not_overflow() -> None:
    # r_x = logit(0.5) - 1e6 * 0.25 * 0.0064 * 1e6 = -1.6e9; exp(1.6e9) would overflow
    q = LOGIT.quote(mid=50, inventory=1e6, sigma=LOGIT_SIGMA, time_remaining=1e6)
    assert 0 <= q.bid <= q.ask < 100


def test_logit_round_trip() -> None:
    # q = 0, sigma = 0, tau = 0: spread_x = (2 / 0.25) * ln(1 + 0.25 / 1e6) = 8 * 2.5e-7 = 2e-6
    # so bid and ask sit ~2.3e-5 cents either side of the mid
    q = LogitSpaceAS(gamma=0.25, k=1e6).quote(mid=37.3, inventory=0, sigma=0, time_remaining=0)
    assert q.reservation == pytest.approx(37.3, abs=1e-9)
    assert q.bid == pytest.approx(37.3, abs=1e-4)
    assert q.ask == pytest.approx(37.3, abs=1e-4)
    assert q.bid < 37.3 < q.ask


# --- round_to_ticks ---


def test_round_floors_bid_and_ceils_ask() -> None:
    # floor(45.3155) = 45, ceil(46.6845) = 47
    assert round_to_ticks(Quote(45.3155, 46.6845, 46.0)) == TickQuote(45, 47)


def test_round_integer_inputs_unchanged() -> None:
    assert round_to_ticks(Quote(45.0, 47.0, 46.0)) == TickQuote(45, 47)


def test_round_with_larger_tick() -> None:
    # tick 5: floor(46.2 / 5) * 5 = 9 * 5 = 45; ceil(53.1 / 5) * 5 = 11 * 5 = 55
    assert round_to_ticks(Quote(46.2, 53.1, 50.0), tick=5, lo=5, hi=95) == TickQuote(45, 55)


def test_round_clamps_one_side() -> None:
    # floor(0.4) = 0 -> clamped to 1; ceil(3.2) = 4
    assert round_to_ticks(Quote(0.4, 3.2, 1.8)) == TickQuote(1, 4)
    # floor(96.5) = 96; ceil(99.3) = 100 -> clamped to 99
    assert round_to_ticks(Quote(96.5, 99.3, 97.9)) == TickQuote(96, 99)


def test_round_both_clamped_high_separates() -> None:
    # floor(99.4) = 99, ceil(100.6) = 101 -> 99: bid = ask = 99
    # ask is pinned at hi, so bid moves down one tick: (98, 99)
    assert round_to_ticks(Quote(99.4, 100.6, 100.0)) == TickQuote(98, 99)


def test_round_both_clamped_low_separates() -> None:
    # Price-space quote at mid 2, q 100: bid -2.6845, ask -1.3155
    # floor(-2.6845) = -3 -> 1, ceil(-1.3155) = -1 -> 1: bid = ask = 1
    # bid is pinned at lo, so ask moves up one tick: (1, 2)
    q = PRICE.quote(mid=2, inventory=100, sigma=PRICE_SIGMA, time_remaining=1)
    assert round_to_ticks(q) == TickQuote(1, 2)


# --- Validation ---


@pytest.mark.parametrize("mid", [0, 100, -1, 101, math.nan])
def test_invalid_mid(mid: float) -> None:
    with pytest.raises(ValueError):
        PRICE.quote(mid, 0, 1, 1)


@pytest.mark.parametrize("inventory", [math.nan, math.inf, -math.inf])
def test_invalid_inventory(inventory: float) -> None:
    for model in (PRICE, LOGIT):
        with pytest.raises(ValueError):
            model.quote(50, inventory, 1, 1)


@pytest.mark.parametrize("gamma", [0, -0.1, math.nan])
def test_invalid_gamma(gamma: float) -> None:
    for cls in (PriceSpaceAS, LogitSpaceAS):
        with pytest.raises(ValueError):
            cls(gamma=gamma, k=1.5)


@pytest.mark.parametrize("k", [0, -1, math.nan])
def test_invalid_k(k: float) -> None:
    for cls in (PriceSpaceAS, LogitSpaceAS):
        with pytest.raises(ValueError):
            cls(gamma=0.01, k=k)


@pytest.mark.parametrize("sigma", [-0.1, math.nan])
def test_invalid_sigma(sigma: float) -> None:
    for model in (PRICE, LOGIT):
        with pytest.raises(ValueError):
            model.quote(50, 0, sigma, 1)


@pytest.mark.parametrize("tau", [-1, math.nan])
def test_invalid_time_remaining(tau: float) -> None:
    for model in (PRICE, LOGIT):
        with pytest.raises(ValueError):
            model.quote(50, 0, 1, tau)


def test_zero_sigma_and_tau_are_valid() -> None:
    # sigma = 0, tau = 0: no inventory term, spread = (2 / 0.01) * ln(1 + 0.01 / 1.5) = 1.3289
    q = PRICE.quote(50, 100, 0, 0)
    assert q.reservation == 50.0
    assert q.ask - q.bid == pytest.approx(1.3289085, abs=1e-6)


@pytest.mark.parametrize(
    ("tick", "lo", "hi"),
    [
        (0, 1, 99),   # tick < 1
        (1, 0, 99),   # lo not > 0
        (1, 1, 100),  # hi not < 100
        (1, 50, 50),  # lo not < hi
        (5, 1, 95),   # lo not a multiple of tick
        (5, 5, 97),   # hi not a multiple of tick
    ],
)
def test_invalid_round_params(tick: int, lo: int, hi: int) -> None:
    with pytest.raises(ValueError):
        round_to_ticks(Quote(40.0, 60.0, 50.0), tick=tick, lo=lo, hi=hi)
